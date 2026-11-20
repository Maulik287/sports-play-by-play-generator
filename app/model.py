"""Transformer and GPT Model components from scratch in PyTorch."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPTEmbedding(nn.Module):
    """Token and positional embedding layer."""
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        if t > self.max_len:
            raise ValueError(f"Sequence length {t} exceeds maximum length {self.max_len}")
        positions = torch.arange(0, t, dtype=torch.long, device=idx.device)
        tok_emb = self.token_embeddings(idx)           # (b, t, d_model)
        pos_emb = self.position_embeddings(positions)  # (t, d_model)
        x = tok_emb + pos_emb
        return self.dropout(x)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention mechanism."""
    def __init__(self, d_model: int, n_heads: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Key, Query, Value projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: lower triangular matrix of ones
        self.register_buffer("tril", torch.tril(torch.ones(max_len, max_len)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, c = x.shape

        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)  # (b, nh, t, hd)
        k = self.k_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)  # (b, nh, t, hd)
        v = self.v_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)  # (b, nh, t, hd)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # (b, nh, t, t)
        scores = scores.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        y = weights @ v  # (b, nh, t, hd)
        y = y.transpose(1, 2).contiguous().view(b, t, c)  # Re-assemble head outputs
        out = self.resid_dropout(self.out_proj(y))
        return out, weights


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    def __init__(self, d_model: int, d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Transformer decoder block with causal self-attention and MLP."""
    def __init__(self, d_model: int, n_heads: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_len=max_len, dropout=dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.ln_1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x


class SportsGPT(nn.Module):
    """Autoregressive Sports Commentary GPT model."""
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_len: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding = GPTEmbedding(vocab_size, d_model, max_len=max_len, dropout=dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_len=max_len, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embedding.token_embeddings.weight

        # Parameter initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, t = idx.shape
        x = self.embedding(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (b, t, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 40,
        temperature: float = 0.8,
        top_k: int | None = 10,
        eos_token_id: int | None = None
    ) -> torch.Tensor:
        """Autoregressively sample tokens given conditioning prefix."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_len else idx[:, -self.max_len:]
            logits, _ = self(idx_cond)
            # Pick logits at final step
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)

            if eos_token_id is not None and (next_idx == eos_token_id).all():
                break

        return idx
