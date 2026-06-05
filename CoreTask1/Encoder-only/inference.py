import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
CHECKPOINT_PATH = "./checkpoints/bestEncoderOnlyTransformer.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
SHOW_ATTENTION = 1
if SHOW_ATTENTION:
    try:
        from IPython.display import display, HTML
        from bertviz import head_view
    except ImportError:
        print("bertviz not installed")
        SHOW_ATTENTION = False

# model/config (training config dont change)
d_model = 64
val_cnt = 1000          # values in [0, 999] - 1000 values
seq_len = 10            # each sequence is of length 10
n_heads = 4
n_layers = 4
hidden = 256

def normalise(x, method="minmax"):
    if method == "minmax":                                       # sequence wise normalisation
        min = x.min(dim=-1, keepdim=True).values                 # x is (B, 10), and keepdim to keep the 2nd dimension (B, 1)
        max = x.max(dim=-1, keepdim=True).values
        x = (x-min)/torch.max(max-min,torch.scalar_tensor(1e-6))
    elif method == "zscore":
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x-mean)/torch.max(std,torch.scalar_tensor(1e-6))
    return x

class NormProjectionEmbed(nn.Module):
    def __init__(self, d_model=d_model, norm="minmax"):
        super().__init__()
        self.norm = norm
        self.proj = nn.Linear(1, d_model)                   # Linear layer with common (W,b) for all values
    def forward(self, x):
        x = x.float()
        x = normalise(x, method=self.norm)
        return self.proj(x.unsqueeze(-1))                   # (B, 10) -> (B, 10, 1) -> (B, 10, d_model)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=d_model, n_heads=n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x):
        batch, seq_len, _ = x.shape
        H = self.n_heads
        dk = self.d_k
        Q = self.W_q(x).view(batch,seq_len,H,dk).transpose(1,2)
        K = self.W_k(x).view(batch,seq_len,H,dk).transpose(1,2)
        V = self.W_v(x).view(batch,seq_len,H,dk).transpose(1,2)

        attn_scores = (Q @ K.transpose(-2,-1)) / (dk**0.5)
        weights = torch.softmax(attn_scores,dim=-1)
        out = weights @ V
        out = out.transpose(1,2).contiguous().view(batch,seq_len,-1)
        return self.W_o(out), weights

class TransformerBlock(nn.Module):
    def __init__(self, d_model=d_model, n_heads=n_heads, hidden = 256):
        super().__init__()
        self.attn = MultiHeadAttention(d_model,n_heads)
        self.ff = nn.Sequential(nn.Linear(d_model,hidden), nn.ReLU(), nn.Linear(hidden,d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        attn_out, w = self.attn(x)
        x = self.ln1(x + attn_out)
        x = x + self.ff(self.ln2(x))
        return x, w

class EncoderOnlyTransformer(nn.Module):
    def __init__(self, embedder=NormProjectionEmbed(norm="zscore"), d_model = d_model, n_heads=n_heads, n_classes = seq_len,
                 n_layers = 2,
                 hidden = 256,
                 use_pe = False,
                 ):
        super().__init__()
        self.embed = embedder
        self.use_pe = use_pe
        self.blocks = nn.ModuleList([TransformerBlock(d_model,n_heads,hidden) for _ in range(n_layers)])
        self.head = nn.Linear(d_model,n_classes)

        if use_pe:
            self.pe = nn.Embedding(seq_len,d_model)

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


model = EncoderOnlyTransformer(embedder=NormProjectionEmbed(norm="zscore"), d_model=d_model, n_heads=n_heads, n_layers=n_layers, n_classes=seq_len,).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
model.eval()

def act_ranks(arr):
    return np.argsort(np.argsort(arr)).tolist()

def predict_ranks(seq, show_attention=SHOW_ATTENTION):
    if len(seq) != seq_len:
        raise ValueError(f"Expected {seq_len} numbers, got {len(seq)}")
    x = torch.tensor([seq], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, all_weights = model.forward_with_weights(x)
        pred = logits.argmax(-1).squeeze(0).cpu().numpy().tolist()
    out = {"input": seq, "predicted_ranks": pred, "target": act_ranks(seq),}
    print("Input seq  :", out["input"])
    print("Predicted ranks :", out["predicted_ranks"])
    print("Target ranks :", out["target"])

    if show_attention:
            layer_idx = -1
            sort_idx = np.argsort(seq)
            sort_seq = sorted(seq)

            weights = all_weights[layer_idx][0].cpu().numpy()
            fig, axes = plt.subplots(n_heads//2, 2, figsize=(15, 15))
            axes = axes.flatten()
            tick_labels = [str(v) for v in sort_seq]

            for h in range(n_heads):
                sns.heatmap(weights[h][sort_idx, :][:, sort_idx], ax=axes[h], xticklabels=tick_labels, yticklabels=tick_labels, cmap='Blues', vmin=0, vmax=1, annot=True, fmt='.2f')
                axes[h].set_title(f'Head {h+1}')
                axes[h].set_xlabel('Key')
                axes[h].set_ylabel('Query')

            plt.suptitle(f'Attention weights')
            plt.tight_layout()
            plt.show()

            return out


sample_seq = eval(input("Enter List :"))
predict_ranks(sample_seq)