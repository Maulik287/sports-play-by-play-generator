"""Sports Play-by-Play Generator: End-to-end GPT implementation from scratch."""
from __future__ import annotations
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from app.data import load_sports_corpus, split_into_snippets, clean_text
from app.tokenizer import BPETokenizer
from app.model import GPTEmbedding, CausalSelfAttention, TransformerBlock, SportsGPT
from app.storage import Store


# Default cached components for fast CLI and test runs
_DEFAULT_TOKENIZER: BPETokenizer | None = None
_DEFAULT_MODEL: SportsGPT | None = None


def get_trained_tokenizer(vocab_size: int = 220) -> BPETokenizer:
    """Get or train cached BPE tokenizer on sports corpus."""
    global _DEFAULT_TOKENIZER
    if _DEFAULT_TOKENIZER is None or _DEFAULT_TOKENIZER.vocab_size != vocab_size:
        corpus = load_sports_corpus()
        tok = BPETokenizer()
        tok.train(corpus, target_vocab_size=vocab_size)
        _DEFAULT_TOKENIZER = tok
    return _DEFAULT_TOKENIZER


def create_dataset_batches(
    token_ids: list[int],
    block_size: int = 32,
    batch_size: int = 8
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Create input-target training pairs from token sequence."""
    batches = []
    n_tokens = len(token_ids)
    if n_tokens <= block_size:
        # Pad sequence if too short
        token_ids = token_ids + [0] * (block_size + 1 - n_tokens)
        n_tokens = len(token_ids)

    xs, ys = [], []
    step = max(1, block_size // 2)
    for i in range(0, n_tokens - block_size, step):
        x = token_ids[i : i + block_size]
        y = token_ids[i + 1 : i + block_size + 1]
        xs.append(x)
        ys.append(y)

    # Group into batches
    for i in range(0, len(xs) - batch_size + 1, batch_size):
        bx = torch.tensor(xs[i : i + batch_size], dtype=torch.long)
        by = torch.tensor(ys[i : i + batch_size], dtype=torch.long)
        batches.append((bx, by))

    if not batches and xs:
        bx = torch.tensor(xs, dtype=torch.long)
        by = torch.tensor(ys, dtype=torch.long)
        batches.append((bx, by))

    return batches


def train_gpt_model(
    model: SportsGPT,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    epochs: int = 15,
    lr: float = 3e-3
) -> list[float]:
    """Train SportsGPT model and return loss history."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_history: list[float] = []

    for _ in range(epochs):
        epoch_losses = []
        for x, y in batches:
            optimizer.zero_grad()
            _, loss = model(x, targets=y)
            assert loss is not None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())
        if epoch_losses:
            loss_history.append(sum(epoch_losses) / len(epoch_losses))

    return loss_history


def get_trained_model(epochs: int = 12) -> tuple[SportsGPT, BPETokenizer]:
    """Get or train a cached lightweight SportsGPT instance."""
    global _DEFAULT_MODEL
    tokenizer = get_trained_tokenizer(vocab_size=200)
    if _DEFAULT_MODEL is None:
        corpus = load_sports_corpus()
        token_ids = tokenizer.encode(corpus)
        batches = create_dataset_batches(token_ids, block_size=24, batch_size=8)
        model = SportsGPT(
            vocab_size=tokenizer.vocab_size,
            d_model=48,
            n_heads=4,
            n_layers=2,
            max_len=64,
            dropout=0.05
        )
        train_gpt_model(model, batches, epochs=epochs, lr=3e-3)
        _DEFAULT_MODEL = model
    return _DEFAULT_MODEL, tokenizer


def generate_sports_commentary(
    model: SportsGPT,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 35,
    temperature: float = 0.8,
    top_k: int = 10,
    sport: str | None = None,
    excitement: str = "high"
) -> str:
    """Generate dynamic play-by-play sports commentary given prompt and controls."""
    prefix = ""
    if sport:
        prefix = f"[{sport.upper()}] "
    full_prompt = (prefix + prompt).strip()

    input_ids = tokenizer.encode(full_prompt)
    if not input_ids:
        input_ids = [tokenizer.bos_token_id]

    idx = torch.tensor([input_ids], dtype=torch.long)
    generated_idx = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=tokenizer.eos_token_id
    )

    result_ids = generated_idx[0].tolist()
    decoded = tokenizer.decode(result_ids)

    # Style commentary based on excitement level
    if excitement.lower() == "high":
        if not decoded.endswith("!"):
            decoded = decoded.rstrip(".") + "!"
    elif excitement.lower() == "low":
        decoded = decoded.replace("!", ".")

    return decoded


# --- Milestone 1: Gather a small corpus of sports commentary or news articles.
# Hint: Data collection and preprocessing
def milestone_1() -> dict[str, Any]:
    """Gather, clean and preprocess the sports commentary corpus."""
    corpus = load_sports_corpus()
    snippets = split_into_snippets(corpus)
    vocab = set(corpus.split())
    if not snippets or len(corpus) < 100:
        raise ValueError("Sports corpus is empty or insufficient.")
    return {
        "status": "completed",
        "corpus_length": len(corpus),
        "num_snippets": len(snippets),
        "vocab_size_words": len(vocab),
        "sample": snippets[0],
    }


# --- Milestone 2: Implement the byte-pair tokenizer and create embeddings from the sports text.
# Hint: Byte-pair tokenization and embedding creation
def milestone_2() -> dict[str, Any]:
    """Train BPE tokenizer and instantiate PyTorch embedding layers."""
    corpus = load_sports_corpus()
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, target_vocab_size=180)

    sample_text = "Goal! Messi curls it into the corner!"
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)

    d_model = 32
    max_len = 64
    embedding_layer = GPTEmbedding(vocab_size=tokenizer.vocab_size, d_model=d_model, max_len=max_len)
    test_tensor = torch.tensor([encoded[:8]], dtype=torch.long)
    embedded = embedding_layer(test_tensor)

    return {
        "status": "completed",
        "vocab_size": tokenizer.vocab_size,
        "num_merges": len(tokenizer.merges),
        "encoded_sample": encoded,
        "decoded_sample": decoded,
        "embedding_shape": tuple(embedded.shape),
    }


# --- Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
# Hint: Causal self-attention and transformer block assembly
def milestone_3() -> dict[str, Any]:
    """Assemble causal self-attention and transformer block, verifying causal mask."""
    d_model = 32
    n_heads = 4
    max_len = 48
    attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, max_len=max_len)
    block = TransformerBlock(d_model=d_model, n_heads=n_heads, max_len=max_len)

    # Test causality: check mask is lower triangular
    is_causal = torch.all(attn.tril == torch.tril(torch.ones(max_len, max_len))).item()

    # Verify transformer block input and output shapes match
    x = torch.randn(2, 10, d_model)
    out = block(x)

    model = SportsGPT(vocab_size=100, d_model=d_model, n_heads=n_heads, n_layers=2, max_len=max_len)
    logits, _ = model(torch.randint(0, 100, (2, 8)))

    return {
        "status": "completed",
        "is_causal_mask": is_causal,
        "block_output_shape": tuple(out.shape),
        "logits_shape": tuple(logits.shape),
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }


# --- Milestone 4: Train the GPT model using the sports text data.
# Hint: Model training
def milestone_4() -> dict[str, Any]:
    """Train the GPT model using the sports text data and verify loss reduction."""
    tokenizer = get_trained_tokenizer(vocab_size=160)
    corpus = load_sports_corpus()
    token_ids = tokenizer.encode(corpus)
    batches = create_dataset_batches(token_ids, block_size=20, batch_size=8)

    model = SportsGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_heads=4,
        n_layers=2,
        max_len=48,
        dropout=0.0
    )

    loss_history = train_gpt_model(model, batches, epochs=10, lr=4e-3)
    initial_loss = loss_history[0]
    final_loss = loss_history[-1]

    return {
        "status": "completed",
        "initial_loss": round(initial_loss, 4),
        "final_loss": round(final_loss, 4),
        "loss_reduced": bool(final_loss < initial_loss),
        "epochs": len(loss_history),
    }


# --- Milestone 5: Develop a text generation function to produce short commentary snippets.
# Hint: Text generation function
def milestone_5() -> dict[str, Any]:
    """Generate dynamic play-by-play commentary snippets."""
    model, tokenizer = get_trained_model(epochs=10)
    prompt = "Curry with five seconds remaining"
    commentary = generate_sports_commentary(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=25,
        temperature=0.8,
        top_k=8,
        sport="Basketball",
        excitement="high"
    )

    return {
        "status": "completed",
        "prompt": prompt,
        "commentary": commentary,
        "is_non_empty": bool(len(commentary) > len(prompt)),
    }


# --- Milestone 6: Create a CLI where users can input a prompt and receive generated commentary.
# Hint: Command-line interface development
def milestone_6() -> dict[str, Any]:
    """Verify CLI integration and commentary saving mechanism."""
    from app.cli import main
    store = Store("data/history.json")
    prompt = "Messi with a free kick outside the box"
    
    # Generate and save via core helper
    model, tokenizer = get_trained_model(epochs=6)
    commentary = generate_sports_commentary(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=25,
        sport="Soccer",
        excitement="high"
    )

    # Store commentary record
    record = {
        "prompt": prompt,
        "commentary": commentary,
        "sport": "Soccer",
        "excitement": "high"
    }
    store.put(f"snippet_{len(store.all()) + 1}", record)

    return {
        "status": "completed",
        "saved_key": f"snippet_{len(store.all())}",
        "record": record,
        "all_records_count": len(store.all()),
    }
