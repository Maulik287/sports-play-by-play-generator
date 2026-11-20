"""Evaluation and benchmarking module for Sports Play-by-Play Generator."""
import math
import json
from pathlib import Path
import torch
import torch.nn.functional as F

from app.core import get_trained_model, generate_sports_commentary, load_sports_corpus, create_dataset_batches
from app.model import SportsGPT


def evaluate_model_perplexity(model: SportsGPT, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, float]:
    """Calculate Cross-Entropy loss and Perplexity on test batches."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for x, y in batches:
            logits, loss = model(x, targets=y)
            if loss is not None:
                total_loss += loss.item() * y.numel()
                total_tokens += y.numel()
    
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20.0))
    return {
        "cross_entropy_loss": round(avg_loss, 4),
        "perplexity": round(perplexity, 2),
    }


def compute_baselines(vocab_size: int, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, dict[str, float]]:
    """Compute comparison metrics against Uniform Random and Bigram baseline models."""
    # 1. Random baseline: loss is ln(V)
    random_loss = math.log(vocab_size)
    random_ppl = math.exp(random_loss)

    # 2. Bigram empirical frequency baseline
    bigram_counts: dict[int, dict[int, int]] = {}
    unigram_counts: dict[int, int] = {}
    
    for x, y in batches:
        x_list = x.flatten().tolist()
        y_list = y.flatten().tolist()
        for prev_tok, next_tok in zip(x_list, y_list):
            unigram_counts[prev_tok] = unigram_counts.get(prev_tok, 0) + 1
            if prev_tok not in bigram_counts:
                bigram_counts[prev_tok] = {}
            bigram_counts[prev_tok][next_tok] = bigram_counts[prev_tok].get(next_tok, 0) + 1

    bigram_loss = 0.0
    tokens_evaluated = 0
    for x, y in batches:
        x_list = x.flatten().tolist()
        y_list = y.flatten().tolist()
        for prev_tok, next_tok in zip(x_list, y_list):
            count_prev = unigram_counts.get(prev_tok, 0)
            count_pair = bigram_counts.get(prev_tok, {}).get(next_tok, 0)
            # Laplace smoothed probability
            prob = (count_pair + 1.0) / (count_prev + vocab_size)
            bigram_loss += -math.log(prob)
            tokens_evaluated += 1

    avg_bigram_loss = bigram_loss / max(tokens_evaluated, 1)
    bigram_ppl = math.exp(min(avg_bigram_loss, 20.0))

    return {
        "random_baseline": {
            "loss": round(random_loss, 4),
            "perplexity": round(random_ppl, 2),
        },
        "bigram_baseline": {
            "loss": round(avg_bigram_loss, 4),
            "perplexity": round(bigram_ppl, 2),
        }
    }


def run_full_evaluation(save_path: str = "eval_results.json") -> dict:
    """Run evaluation comparing SportsGPT against baselines and export results."""
    model, tokenizer = get_trained_model(epochs=12)
    corpus = load_sports_corpus()
    token_ids = tokenizer.encode(corpus)
    batches = create_dataset_batches(token_ids, block_size=24, batch_size=8)

    gpt_metrics = evaluate_model_perplexity(model, batches)
    baselines = compute_baselines(tokenizer.vocab_size, batches)

    sample_scenarios = [
        {"sport": "soccer", "prompt": "Stoppage time penalty kick", "excitement": "high"},
        {"sport": "basketball", "prompt": "Curry step-back three at the buzzer", "excitement": "high"},
        {"sport": "football", "prompt": "Fourth and goal on the one-yard line", "excitement": "high"},
        {"sport": "cricket", "prompt": "Yorker on the final ball with 4 needed", "excitement": "high"},
        {"sport": "tennis", "prompt": "Baseline rally at championship point", "excitement": "medium"},
    ]

    generated_samples = []
    for sc in sample_scenarios:
        output = generate_sports_commentary(
            model=model,
            tokenizer=tokenizer,
            prompt=sc["prompt"],
            sport=sc["sport"],
            excitement=sc["excitement"],
            max_new_tokens=30
        )
        generated_samples.append({
            "sport": sc["sport"],
            "prompt": sc["prompt"],
            "excitement": sc["excitement"],
            "generated_output": output
        })

    eval_data = {
        "model_architecture": {
            "name": "SportsGPT",
            "d_model": 48,
            "n_heads": 4,
            "n_layers": 2,
            "vocab_size": tokenizer.vocab_size,
            "max_len": 64
        },
        "benchmark_comparison": {
            "random_baseline": baselines["random_baseline"],
            "bigram_baseline": baselines["bigram_baseline"],
            "sports_gpt": gpt_metrics
        },
        "sample_generations": generated_samples
    }

    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(eval_data, indent=2), encoding="utf-8")
    return eval_data


if __name__ == "__main__":
    results = run_full_evaluation()
    print("Evaluation completed successfully.")
    print(json.dumps(results, indent=2))
