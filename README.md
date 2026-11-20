# Sports Play-by-Play Generator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/8bgv75burt/badge.svg)](https://ask.thili.ai/b/8bgv75burt)

This project, 'Sports Play-by-Play Generator,' delivers a command-line tool for sports fans. It generates dynamic commentary snippets for fictional game scenarios, leveraging a GPT model built and trained from scratch on sports news and commentary across soccer, basketball, football, cricket, and tennis.

## What you are building

**Goal.** You will build a command-line assistant that generates short, dynamic commentary snippets for fictional game scenarios, trained on real sports news articles.

**Done when.** The project is done when you have successfully applied the byte-pair tokenizer and embeddings, implemented causal self-attention, assembled the transformer block, and trained your GPT model from scratch to generate coherent sports commentary snippets via a command-line interface.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Gather a small corpus of sports commentary or news articles.**  
      Hint: Data collection and preprocessing (`app/data.py`, `data/sports_corpus.txt`)
- [x] **2. Implement the byte-pair tokenizer and create embeddings from the sports text.**  
      Hint: Byte-pair tokenization and embedding creation (`app/tokenizer.py`, `app/model.py`)
- [x] **3. Build the causal self-attention mechanism and assemble the transformer block.**  
      Hint: Causal self-attention and transformer block assembly (`app/model.py`)
- [x] **4. Train the GPT model using the sports text data.**  
      Hint: Model training (`app/core.py`, `app/model.py`)
- [x] **5. Develop a text generation function to produce short commentary snippets.**  
      Hint: Text generation function (`app/core.py`)
- [x] **6. Create a CLI where users can input a prompt and receive generated commentary.**  
      Hint: Command-line interface development (`app/cli.py`)

## Architecture Overview

1. **Corpus & Preprocessing (`app/data.py`)**:
   - Curated multi-sport dataset encompassing dynamic play-by-play moments from soccer, basketball, football, cricket, and tennis.
   - Text normalization, whitespace cleaning, and play snippet extraction.

2. **Byte-Pair Encoding (BPE) Tokenizer (`app/tokenizer.py`)**:
   - Implemented from scratch in pure Python with vocabulary extraction and iterative bigram frequency merging.
   - Handles special tokens (`<pad>`, `<unk>`, `<bos>`, `<eos>`).
   - Supports serialization to and deserialization from JSON.

3. **Causal Transformer & GPT Architecture (`app/model.py`)**:
   - **`GPTEmbedding`**: Token embedding combined with learned positional embeddings.
   - **`CausalSelfAttention`**: Multi-head self-attention with lower-triangular causal masking preventing attention to future positions.
   - **`FeedForward`**: Position-wise two-layer MLP with GELU non-linearity.
   - **`TransformerBlock`**: Pre-LayerNorm decoder block with residual skip connections.
   - **`SportsGPT`**: Full autoregressive GPT model with weight tying and cross-entropy loss computation.

4. **Generation & Controls (`app/core.py`)**:
   - Autoregressive generation with temperature scaling and top-k filtering.
   - Support for sport tags (`--sport soccer/basketball/football/cricket/tennis`).
   - Dynamic excitement modulation (`--excitement high/medium/low`).

5. **CLI & History Persistence (`app/cli.py`, `app/storage.py`)**:
   - Rich CLI to generate commentary, inspect history, train custom weights, and validate milestones.

## Run it

You need Python 3.12+ and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

### CLI Commands

- **Check setup:**
  ```bash
  python -m app.cli hello
  ```

- **Generate sports commentary:**
  ```bash
  python -m app.cli generate "Step back three at the buzzer" --sport basketball --excitement high --save
  ```

- **View saved commentary history:**
  ```bash
  python -m app.cli history
  ```

- **Train the model:**
  ```bash
  python -m app.cli train --epochs 12 --vocab-size 180
  ```

- **Validate all milestones:**
  ```bash
  python -m app.cli milestones
  ```

## Stretch ideas Implemented

- **Sport / Scenario Targeting**: Choose among multiple sports (`soccer`, `basketball`, `football`, `cricket`, `tennis`).
- **Excitement Modulation**: Adjust commentary tone between `low`, `medium`, and `high` energy.
- **Persistent History Storage**: Automatically stores generated snippets and metadata in `data/history.json` through `Store`.
