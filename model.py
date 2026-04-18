import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear layers for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # 1. Linear projection and split into heads
        # Shape: [batch, heads, seq_len, d_k]
        Q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2. Scaled Dot-Product Attention
        # Energy shape: [batch, heads, seq_len, seq_len]
        energy = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # Mask is [batch, 1, 1, seq_len] - broadcast to energy shape
            energy = energy.masked_fill(mask == 0, float('-1e9'))

        attention = F.softmax(energy, dim=-1)

        # 3. Multiply by V and concatenate
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.fc_out(out)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Sub-layer 1: Attention + Residual + Norm
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Sub-layer 2: Feed Forward + Residual + Norm
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size=5, d_model=64, num_heads=4, d_ff=128, num_layers=1, use_pe=True):
        super().__init__()
        self.use_pe = use_pe
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])

        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x, mask):
        # Prepare mask for MHA [batch, 1, 1, seq_len]
        mha_mask = mask.unsqueeze(1).unsqueeze(2)

        out = self.embedding(x)
        if self.use_pe:
            out = self.pos_encoding(out)

        for layer in self.layers:
            out = layer(out, mha_mask)

        # Global Mean Pooling: only average non-padding tokens
        # Create a mask that is [batch, seq_len, 1]
        mask_expanded = mask.unsqueeze(-1)
        sum_embeddings = torch.sum(out * mask_expanded, dim=1)
        num_active = torch.sum(mask_expanded, dim=1)
        pooled = sum_embeddings / num_active

        return torch.sigmoid(self.classifier(pooled))



