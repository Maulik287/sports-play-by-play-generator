"""Unit tests for all 6 milestones of the Sports Play-by-Play Generator."""
import pytest
import torch

from app import core
from app.data import load_sports_corpus, split_into_snippets, clean_text
from app.tokenizer import BPETokenizer
from app.model import GPTEmbedding, CausalSelfAttention, TransformerBlock, SportsGPT
from app.storage import Store
from app.cli import main


# Milestone 1: Gather a small corpus of sports commentary or news articles.
def test_milestone_1():
    result = core.milestone_1()
    assert result["status"] == "completed"
    assert result["corpus_length"] > 200
    assert result["num_snippets"] >= 5
    assert result["vocab_size_words"] > 20
    assert len(result["sample"]) > 10

    # Test corpus utilities
    raw_text = "  Goal!   Great strike.\n\n\nAnother play.  "
    cleaned = clean_text(raw_text)
    assert "  " not in cleaned
    snippets = split_into_snippets(cleaned)
    assert len(snippets) == 2


# Milestone 2: Implement the byte-pair tokenizer and create embeddings from the sports text.
def test_milestone_2():
    result = core.milestone_2()
    assert result["status"] == "completed"
    assert result["vocab_size"] > 50
    assert result["num_merges"] > 0
    assert len(result["encoded_sample"]) > 0
    assert "corner" in result["decoded_sample"].lower() or "goal" in result["decoded_sample"].lower()
    assert len(result["embedding_shape"]) == 3

    # Direct tokenizer tests
    tok = BPETokenizer()
    text = "Curry hits the three! Curry scores!"
    tok.train(text, target_vocab_size=60)
    encoded = tok.encode("Curry hits")
    decoded = tok.decode(encoded)
    assert "curry" in decoded.lower()
    assert tok.pad_token_id == 0
    assert tok.unk_token_id == 1

    # Embedding tests
    emb = GPTEmbedding(vocab_size=tok.vocab_size, d_model=32, max_len=64)
    out = emb(torch.tensor([[0, 1, 2]]))
    assert out.shape == (1, 3, 32)


# Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
def test_milestone_3():
    result = core.milestone_3()
    assert result["status"] == "completed"
    assert result["is_causal_mask"] is True
    assert result["block_output_shape"] == (2, 10, 32)
    assert result["logits_shape"] == (2, 8, 100)
    assert result["total_parameters"] > 0

    # Causal self-attention test
    attn = CausalSelfAttention(d_model=32, n_heads=4, max_len=32)
    x = torch.randn(2, 6, 32)
    out, weights = attn(x)
    assert out.shape == (2, 6, 32)
    assert weights.shape == (2, 4, 6, 6)
    # Check causality: upper triangular weights must be 0
    for i in range(6):
        for j in range(i + 1, 6):
            assert torch.all(weights[:, :, i, j] == 0.0)

    # Transformer block test
    block = TransformerBlock(d_model=32, n_heads=4, max_len=32)
    blk_out = block(x)
    assert blk_out.shape == (2, 6, 32)


# Milestone 4: Train the GPT model using the sports text data.
def test_milestone_4():
    result = core.milestone_4()
    assert result["status"] == "completed"
    assert result["loss_reduced"] is True
    assert result["final_loss"] < result["initial_loss"]
    assert result["epochs"] == 10

    # Test dataset batching
    token_ids = list(range(50))
    batches = core.create_dataset_batches(token_ids, block_size=16, batch_size=4)
    assert len(batches) > 0
    bx, by = batches[0]
    assert bx.shape[1] == 16
    assert by.shape[1] == 16
    # Targets are shifted by 1
    assert torch.equal(bx[:, 1:], by[:, :-1])


# Milestone 5: Develop a text generation function to produce short commentary snippets.
def test_milestone_5():
    result = core.milestone_5()
    assert result["status"] == "completed"
    assert len(result["commentary"]) > len(result["prompt"])
    assert result["is_non_empty"] is True

    # Test commentary generation with options
    model, tokenizer = core.get_trained_model(epochs=4)
    comm_soccer = core.generate_sports_commentary(
        model=model,
        tokenizer=tokenizer,
        prompt="Penalty kick",
        sport="soccer",
        excitement="high"
    )
    assert "[SOCCER]" in comm_soccer
    assert comm_soccer.endswith("!")

    comm_low = core.generate_sports_commentary(
        model=model,
        tokenizer=tokenizer,
        prompt="Slow baseline rally",
        sport="tennis",
        excitement="low"
    )
    assert "[TENNIS]" in comm_low
    assert "!" not in comm_low


# Milestone 6: Create a CLI where users can input a prompt and receive generated commentary.
def test_milestone_6(capsys, tmp_path):
    result = core.milestone_6()
    assert result["status"] == "completed"
    assert "snippet_" in result["saved_key"]
    assert result["all_records_count"] >= 1

    # Test CLI commands
    main(["hello"])
    out = capsys.readouterr().out
    assert "Sports Play-by-Play Generator is ready" in out

    main(["generate", "Overtime buzzer beater", "--sport", "basketball", "--excitement", "high", "--save"])
    gen_out = capsys.readouterr().out
    assert "Live Commentary" in gen_out
    assert "Saved commentary snippet" in gen_out

    main(["history"])
    hist_out = capsys.readouterr().out
    assert "Commentary History" in hist_out
    assert "Overtime buzzer beater" in hist_out

    main(["milestones"])
    m_out = capsys.readouterr().out
    assert "All 6 milestones executed successfully" in m_out
