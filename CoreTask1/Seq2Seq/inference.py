import numpy as np
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
CHECKPOINT_PATH = "./checkpoints/bestTransformerSeq2Seq.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# vocab / config (must match training)
PAD, BOS, EOS, UNK = 0,1,2,3
D_MODEL = 96
N_HEADS = 4
MAX_LEN = 50

# vocab needs to be rebuilt from training data to match checkpoint
try:
    import pandas as pd
    splits = {'train': 'small/train-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/google/code_x_glue_cc_code_refinement/" + splits["train"])
    all_tok = [k for i,j in zip(df['buggy'],df['fixed']) for k in i.split() + j.split()]
    vocab = {'<PAD>':0, '<BOS>':1, '<EOS>':2, '<UNK>':3}
    vocab.update({t:i+4 for i,t in enumerate(sorted(set(all_tok)))})
    inv_vocab = {i:t for t,i in vocab.items()}
    VOCAB_SIZE = len(vocab)
except Exception as e:
    print(f"could not load dataset: {e}")
    raise

def encode(text):
    ids = [BOS] + [vocab.get(t, UNK) for t in text.split()] + [EOS]
    return ids[:MAX_LEN] + [PAD] * max(0, MAX_LEN - len(ids))

def decode(ids):
    return ' '.join(inv_vocab.get(i, '?') for i in ids if i not in (PAD,BOS,EOS))


class SwiGLU(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2,dim=-1)
        return x1 * nn.functional.silu(x2)
class GEGLU(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2,dim=-1)
        return x1 * nn.functional.gelu(x2)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, causal=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.dk = d_model // n_heads
        self.causal = causal

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        batch, seq_len_q, _ = q.shape
        seq_len_k = k.shape[1]
        H = self.n_heads
        dk = self.dk
        Q = self.W_q(q).view(batch, seq_len_q, H, dk).transpose(1, 2)   # (B, H, seq_len_q, dk)
        K = self.W_k(k).view(batch, seq_len_k, H, dk).transpose(1, 2)   # (B, H, seq_len_k, dk)
        V = self.W_v(v).view(batch, seq_len_k, H, dk).transpose(1, 2)   # (B, H, seq_len_k, dk)
        attn_scores = (Q @ K.transpose(-2,-1)) / (dk**0.5)  # (B, H, seq_len_q, seq_len_k)
        if self.causal:
            causal_mask = torch.triu(torch.ones(seq_len_q, seq_len_q, device=q.device), diagonal=1).bool() # (seq_len_q, seq_len_q) only for decoder selfattn
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf')) # (B, H, seq_len_q, seq_len_k)
        # padding mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, seq_len_k)
            attn_scores = attn_scores.masked_fill(mask, float('-inf')) # (B, H, seq_len_q, seq_len_k) 

        weights = torch.softmax(attn_scores, dim=-1)  # (B, H, seq_len_q, seq_len_k)
        out = weights @ V                             # (B, H, seq_len_q, dk)
        out = out.transpose(1,2).contiguous().view(batch,seq_len_q,-1)  # (B, seq_len_q, d_model)
        return self.W_o(out), weights                                   # (B, seq_len_q, d_model)


class EncBlock(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, hidden=256, drop=0.2,activation=nn.ReLU):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, causal=False)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        if activation == "swiglu" or activation =="geglu":
            act = SwiGLU() if activation == "swiglu" else GEGLU()
            self.ff = nn.Sequential(nn.Linear(d_model, hidden*2), act,nn.Dropout(drop), nn.Linear(hidden, d_model))
        else:
            act = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'gelu': nn.GELU()}.get(activation, nn.ReLU())
            self.ff = nn.Sequential(nn.Linear(d_model, hidden), act,nn.Dropout(drop), nn.Linear(hidden, d_model))
  
        self.drop = nn.Dropout(drop)
    def forward(self, x, mask=None):
        x_norm = self.ln1(x)                              #(B,seq_len,d_model)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, mask)  #(B,seq_len,d_model)
        x = x + self.drop(attn_out)                       #(B,seq_len,d_model)
        x = x + self.drop(self.ff(self.ln2(x)))       
        return x


class DecBlock(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, hidden=256,drop=0.1,activation=nn.ReLU):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, causal=True)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, causal=False)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        if activation == "swiglu" or activation =="geglu":
            act = SwiGLU() if activation == "swiglu" else GEGLU()
            self.ff = nn.Sequential(nn.Linear(d_model, hidden*2), act,nn.Dropout(drop), nn.Linear(hidden, d_model))
        else:
            act = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'gelu': nn.GELU()}.get(activation, nn.ReLU())
            self.ff = nn.Sequential(nn.Linear(d_model, hidden), act,nn.Dropout(drop), nn.Linear(hidden, d_model))
        self.drop = nn.Dropout(drop)
    def forward(self, x, enc_out, enc_mask=None, tgt_mask=None):
        # x: (B, tgt_len, d_model), enc_out: (B, seq_len, d_model), enc_mask: (B, seq_len)
        # masked self-attention (causal)
        x_norm = self.ln1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, tgt_mask)
        x = x + self.drop(attn_out)
        # cross-attention (no causal)
        cross_out, cw = self.cross_attn(self.ln2(x), enc_out, enc_out, enc_mask) # (B,tgt_len, d_model), (B, H, seq_len_q,seq_len_k)
        x = x + self.drop(cross_out)
        x = x + self.drop(self.ff(self.ln3(x)))
        return x, cw                #(B,tgt_len,d_model)


class SinPE(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()*(-np.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerSeq2Seq(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, n_enc=2, n_dec=2, hidden=256,drop=0.1
                ,tie_embedding=False
                ,activation='relu'):
        super().__init__()
        self.src_emb = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD)
        if tie_embedding:
            self.tgt_emb.weight = self.src_emb.weight
        self.pe = SinPE(d_model)
        self.enc_blocks = nn.ModuleList([EncBlock(d_model, n_heads, hidden,drop=drop, activation=activation) for _ in range(n_enc)])
        self.dec_blocks = nn.ModuleList([DecBlock(d_model, n_heads, hidden, drop=drop,activation=activation) for _ in range(n_dec)])
        self.head = nn.Linear(d_model, VOCAB_SIZE)

    def encode(self, src):
        mask = (src == PAD)         # (B, seq_len)
        x = self.pe(self.src_emb(src))   #(B, seq_len, d_model)
        for blk in self.enc_blocks:
            x = blk(x, mask)    
        return x, mask                  # (B, seq_len, d_model), (B, seq_len)

    def decode(self, tgt, enc, enc_mask):
        tgt_pad_mask = (tgt == PAD)                # (B, tgt_len)
        x = self.pe(self.tgt_emb(tgt))  #(B, tgt_len, d_model)
        cross_attns = []
        for blk in self.dec_blocks:
            x, cw = blk(x, enc, enc_mask, tgt_pad_mask)        # (B, tgt_len, d_model), (B, H, seq_len_q, seq_len_k)
            cross_attns.append(cw)
        return self.head(x), cross_attns        # (B, tgt_len, VOCAB_SIZE)

    def forward(self, src, tgt):
        enc, enc_mask = self.encode(src)
        out, _ = self.decode(tgt[:,:-1], enc, enc_mask)  # shift right by dropping last so that no position sees itself in decode
        dummy = torch.zeros(tgt.size(0), 1, out.size(-1), device=tgt.device)
        return torch.cat([dummy, out], dim=1)               # pad col 0 so logits[:,1:] still aligns


model = TransformerSeq2Seq(hidden=128,d_model=D_MODEL,n_heads=N_HEADS).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
model.eval()

@torch.no_grad()
def greedy_decode(model, src, max_len=MAX_LEN):
    model.eval()
    tokens = [BOS]
    enc, enc_mask = model.encode(src)
    for _ in range(max_len - 1):
        tgt_in = torch.tensor([tokens], dtype=torch.long, device=src.device)
        logits, _ = model.decode(tgt_in, enc, enc_mask)
        nxt = logits[0, -1].argmax().item()  
        tokens.append(nxt)
        if nxt == EOS:
            break
    return tokens

def predict(buggy_code):
    src = torch.tensor([encode(buggy_code)], dtype=torch.long, device=device)
    pred_ids = greedy_decode(model, src)
    pred_str = decode(pred_ids)
    print(f"buggy: {buggy_code}")
    print(f"pred:  {pred_str}")

    return pred_str


buggy = input("Enter buggy code snippet: ")
predict(buggy)