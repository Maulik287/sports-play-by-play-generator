# Sports Play-by-Play Generator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/8bgv75burt/badge.svg)](https://ask.thili.ai/b/8bgv75burt)

This project, 'Sports Play-by-Play Generator,' delivers a command-line tool for sports fans. It generates dynamic commentary snippets for fictional game scenarios, leveraging a GPT model trained on real sports news and commentary from the Media & Sports domain.

## What you are building

**Goal.** You will build a command-line assistant that generates short, dynamic commentary snippets for fictional game scenarios, trained on real sports news articles.

**Done when.** The project is done when you have successfully applied the byte-pair tokenizer and embeddings, implemented causal self-attention, assembled the transformer block, and trained your GPT model from scratch to generate coherent sports commentary snippets via a command-line interface.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Gather a small corpus of sports commentary or news articles.**  
      Hint: Data collection and preprocessing
- [ ] **2. Implement the byte-pair tokenizer and create embeddings from the sports text.**  
      Hint: Byte-pair tokenization and embedding creation
- [ ] **3. Build the causal self-attention mechanism and assemble the transformer block.**  
      Hint: Causal self-attention and transformer block assembly
- [ ] **4. Train the GPT model using the sports text data.**  
      Hint: Model training
- [ ] **5. Develop a text generation function to produce short commentary snippets.**  
      Hint: Text generation function
- [ ] **6. Create a CLI where users can input a prompt and receive generated commentary.**  
      Hint: Command-line interface development

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow users to specify a sport or scenario type (e.g., 'basketball', 'touchdown') to guide the generation more effectively.
- Implement options for controlling the length or 'excitement level' of the generated commentary.
- Enhance the command-line interface with features like saving generated snippets or a history of prompts.
