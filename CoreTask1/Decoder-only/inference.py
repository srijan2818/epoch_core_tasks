import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from collections import Counter
CHECKPOINT_PATH = "./checkpoints/bestDecoderOnlyTransformer.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# TRAINING CONFIG SET  dont change
D_MODEL  = 256
N_HEADS  = 4
N_LAYERS = 12
HIDDEN   = 256
SEQ_LEN  = 128

# vocab needs to be rebuilt from source so stoi/itos match training)
df = pd.read_csv("hf://datasets/merve/poetry/poetry.csv")

raw_txt = df['content'].str.cat()
raw_txt = raw_txt.replace('\r\n', '\n')

SPECIAL_MAP = [('{oe}', 'œ'),('{\"e}', 'ë'),('{.}', 'ye'),('{i}{_o}', 'io'),('{_o}', 'o'),('{i}', 'i')]
for old, new in SPECIAL_MAP:
    raw_txt = raw_txt.replace(old, new)

all_poems = []
for poem in df['content']:
    poem = poem.replace('\r\n', '\n')
    for old, new in SPECIAL_MAP:
        poem = poem.replace(old, new)
    poem = poem.strip()
    if poem:
        all_poems.append(poem)

def tokenize(text):
    tokens, word = [], ''
    for c in text:
        if c == '\n':
            if word:
                tokens.append(word)
                word = ''
            tokens.append('<EOL>')
        elif c.isalpha() or c == "'":
            word += c
        elif c.isspace():
            if word:
                tokens.append(word)
                word = ''
        else:
            if word:
                tokens.append(word)
                word = ''
            tokens.append(c)
    if word:
        tokens.append(word)
    return tokens

vocab = ['<BOS>', '<EOS>', '<EOL>'] + sorted(set(tokenize('\n'.join(all_poems))))
stoi = {t: i for i, t in enumerate(vocab)}
itos = {i: t for t, i in stoi.items()}
vocab_size = len(vocab)
bos_idx, eos_idx = 0, 1

def encode(text):
    return [bos_idx] + [stoi[t] for t in tokenize(text)] + [eos_idx]

def decode(ids):
    out, prev_word = [], False
    for i in ids:
        if i in (bos_idx, eos_idx):
            continue
        t = itos[i]
        if t == '<EOL>':
            out.append('\n')
            prev_word = False
        elif len(t) == 1 and not t.isalnum():
            out.append(t)
            prev_word = False
        else:
            if prev_word:
                out.append(' ')
            out.append(t)
            prev_word = True
    return ''.join(out)

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model = D_MODEL, n_heads = N_HEADS):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.dk = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model,bias=False)
        self.W_k = nn.Linear(d_model, d_model,bias=False)
        self.W_v = nn.Linear(d_model, d_model,bias=False)
        self.W_o = nn.Linear(d_model, d_model,bias=False)

    def forward(self, x):
        batch, seq_len,_ = x.shape
        H = self.n_heads
        dk = self.dk
        Q = self.W_q(x).view(batch,seq_len,H,dk).transpose(1,2)
        K = self.W_k(x).view(batch,seq_len,H,dk).transpose(1,2)
        V = self.W_v(x).view(batch,seq_len,H,dk).transpose(1,2)
        attn_scores = (Q @ K.transpose(-2,-1)) / (dk**0.5)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        weights = torch.softmax(attn_scores, dim=-1)
        out = weights @ V
        out = out.transpose(1,2).contiguous().view(batch,seq_len,-1)
        return self.W_o(out), weights

class TransformerBlock(nn.Module):
        def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, hidden=256):
             super().__init__()
             self.attn = MaskedMultiHeadAttention(d_model,n_heads)
             self.ln1 = nn.LayerNorm(d_model)
             self.ln2 = nn.LayerNorm(d_model)
             self.ff = nn.Sequential(nn.Linear(d_model,hidden),nn.ReLU(),nn.Linear(hidden,d_model))

        def forward(self, x):
            attn_out, w = self.attn(x)
            x = self.ln1(x+attn_out)
            x = x + self.ff(self.ln2(x))
            return x, w

class DecoderOnlyTransformer(nn.Module):
        def __init__(self, d_model = D_MODEL, n_heads = N_HEADS, n_classes=vocab_size,
                     n_layers = 4,
                     hidden = 256,
                     use_pe = True):
            super().__init__()

            self.embed = nn.Embedding(vocab_size, d_model)
            self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, hidden) for _ in range(n_layers)])
            self.use_pe = use_pe
            self.head = nn.Linear(d_model, n_classes)
            if use_pe:
                 self.pe = nn.Embedding(SEQ_LEN,d_model)

        def _embed(self, x):
            x = self.embed(x)
            if self.use_pe:
                pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
                x = x + self.pe(pos)
            return x

        def forward(self, x):
            x = self._embed(x)
            for block in self.blocks:
                x, _ = block(x)
            return self.head(x)

        def forward_with_weights(self, x):
            x = self._embed(x)
            all_weights = []
            for block in self.blocks:
                x, w = block(x)
                all_weights.append(w)
            return self.head(x), all_weights

model = DecoderOnlyTransformer(n_layers=N_LAYERS, d_model=D_MODEL, n_classes=vocab_size).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
model.eval()

@torch.no_grad()
def generate(model, prompt, max_new=200, temperature=1.0, top_p=1.0, rep_penalty=1.0):
    model.eval()
    ids = encode(prompt)[:-1]
    ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new):
        logits = model(ids[:, -SEQ_LEN:])[0, -1] # (1, curr_len, vocab) taking last position's logits

        # repetition penalty with penalty (<1 encourage, >1 penalize)
        for tok in set(ids[0, -20:].tolist()):
            logits[tok] /= rep_penalty

        # low tmp -> sharper distribution of token probabilities and high temp -> flatter
        if temperature == 0:
            next_tok = logits.argmax().item()  # greedy
        else:
            logits /= temperature
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
                remove = probs.cumsum(0) - probs > top_p
                sorted_logits[remove] = float('-inf')
                logits[sorted_idx] = sorted_logits
            next_tok = torch.multinomial(torch.nn.functional.softmax(logits, dim=-1), 1).item()

        if next_tok == eos_idx:
            break
        ids = torch.cat([ids, torch.tensor([[next_tok]], device=device)], dim=1)

    return decode(ids[0].tolist())

prompt = input("Enter prompt: ")
print('Top-p 0.9 + rep penalty 1.3')
print('-'*60)
print(generate(model, prompt, temperature=0.8, top_p=0.9, rep_penalty=1.3))