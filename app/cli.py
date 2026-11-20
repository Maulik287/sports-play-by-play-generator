"""Command-line entry point: `python -m app.cli --help`."""
from __future__ import annotations
import argparse
import sys
from app.storage import Store


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="sports-play-by-play-generator",
        description="Sports fans get a command-line assistant that generates short, dynamic commentary snippets for fictional game scenarios, trained on real sports news articles."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # hello
    sub.add_parser("hello", help="check that everything is wired up")

    # milestones
    sub.add_parser("milestones", help="run all milestones and check status")

    # generate
    gen_parser = sub.add_parser("generate", help="generate sports commentary snippet from prompt")
    gen_parser.add_argument("prompt", nargs="?", default="Extra time penalty kick", help="game scenario prompt")
    gen_parser.add_argument("--sport", choices=["soccer", "basketball", "football", "cricket", "tennis"], default=None, help="sport type")
    gen_parser.add_argument("--excitement", choices=["low", "medium", "high"], default="high", help="excitement tone")
    gen_parser.add_argument("--max-tokens", type=int, default=35, help="maximum new tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=0.8, help="sampling temperature")
    gen_parser.add_argument("--top-k", type=int, default=10, help="top-k sampling threshold")
    gen_parser.add_argument("--save", action="store_true", help="save generated commentary to history")

    # history
    sub.add_parser("history", help="display saved commentary snippets")

    # train
    train_parser = sub.add_parser("train", help="train SportsGPT model on sports corpus")
    train_parser.add_argument("--epochs", type=int, default=10, help="training epochs")
    train_parser.add_argument("--vocab-size", type=int, default=180, help="BPE vocabulary size")
    train_parser.add_argument("--lr", type=float, default=3e-3, help="learning rate")

    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Sports Play-by-Play Generator is ready. Open README.md for your milestones.")

    elif args.command == "generate":
        from app.core import get_trained_model, generate_sports_commentary
        print(f"Generating commentary for scenario: '{args.prompt}'...")
        model, tokenizer = get_trained_model()
        commentary = generate_sports_commentary(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            sport=args.sport,
            excitement=args.excitement
        )
        print("\n--- Play-by-Play Live Commentary ---")
        print(commentary)
        print("------------------------------------")

        if args.save:
            store = Store("data/history.json")
            item_id = f"item_{len(store.all()) + 1}"
            store.put(item_id, {
                "prompt": args.prompt,
                "sport": args.sport,
                "excitement": args.excitement,
                "commentary": commentary
            })
            print(f"Saved commentary snippet to history as {item_id}.")

    elif args.command == "history":
        store = Store("data/history.json")
        items = store.all()
        if not items:
            print("No saved commentary history found. Generate some with --save!")
        else:
            print(f"--- Commentary History ({len(items)} items) ---")
            for k, v in items.items():
                print(f"[{k}] ({v.get('sport') or 'General'} - {v.get('excitement')} excitement)")
                print(f"Prompt: {v.get('prompt')}")
                print(f"Commentary: {v.get('commentary')}\n")

    elif args.command == "train":
        from app.core import load_sports_corpus, get_trained_tokenizer, create_dataset_batches, SportsGPT, train_gpt_model
        print(f"Training SportsGPT model for {args.epochs} epochs with vocab size {args.vocab_size}...")
        corpus = load_sports_corpus()
        tok = get_trained_tokenizer(vocab_size=args.vocab_size)
        token_ids = tok.encode(corpus)
        batches = create_dataset_batches(token_ids, block_size=24, batch_size=8)
        model = SportsGPT(vocab_size=tok.vocab_size, d_model=48, n_heads=4, n_layers=2, max_len=64)
        losses = train_gpt_model(model, batches, epochs=args.epochs, lr=args.lr)
        print(f"Training complete. Initial loss: {losses[0]:.4f}, Final loss: {losses[-1]:.4f}")

    elif args.command == "milestones":
        from app import core
        print("Running milestones validation...")
        m1 = core.milestone_1()
        print(f"[Milestone 1] Corpus gathered: {m1['num_snippets']} snippets, {m1['corpus_length']} chars.")
        m2 = core.milestone_2()
        print(f"[Milestone 2] BPE Vocab size: {m2['vocab_size']}, Merges: {m2['num_merges']}.")
        m3 = core.milestone_3()
        print(f"[Milestone 3] Transformer blocks assembled. Causal mask valid: {m3['is_causal_mask']}.")
        m4 = core.milestone_4()
        print(f"[Milestone 4] Model trained: loss went from {m4['initial_loss']} to {m4['final_loss']}.")
        m5 = core.milestone_5()
        print(f"[Milestone 5] Commentary generated: {m5['commentary'][:50]}...")
        m6 = core.milestone_6()
        print(f"[Milestone 6] CLI and history storage verified: {m6['saved_key']}.")
        print("\nAll 6 milestones executed successfully!")


if __name__ == "__main__":
    main()
