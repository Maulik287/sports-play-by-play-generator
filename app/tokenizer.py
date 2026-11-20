"""Byte-Pair Encoding (BPE) Tokenizer implementation from scratch."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path


class BPETokenizer:
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    def __init__(self, vocab: dict[str, int] | None = None, merges: list[tuple[str, str]] | None = None):
        if vocab is None:
            self.special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
            self.token2id: dict[str, int] = {tok: idx for idx, tok in enumerate(self.special_tokens)}
            self.id2token: dict[int, str] = {idx: tok for tok, idx in self.token2id.items()}
            self.merges: list[tuple[str, str]] = []
        else:
            self.token2id = dict(vocab)
            self.id2token = {idx: tok for tok, idx in self.token2id.items()}
            self.merges = merges or []
            self.special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    @property
    def pad_token_id(self) -> int:
        return self.token2id[self.PAD_TOKEN]

    @property
    def unk_token_id(self) -> int:
        return self.token2id[self.UNK_TOKEN]

    @property
    def bos_token_id(self) -> int:
        return self.token2id[self.BOS_TOKEN]

    @property
    def eos_token_id(self) -> int:
        return self.token2id[self.EOS_TOKEN]

    @staticmethod
    def _get_stats(vocab_words: dict[tuple[str, ...], int]) -> Counter[tuple[str, str]]:
        pairs: Counter[tuple[str, str]] = Counter()
        for word, freq in vocab_words.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    @staticmethod
    def _merge_vocab(pair: tuple[str, str], v_in: dict[tuple[str, ...], int]) -> dict[tuple[str, ...], int]:
        v_out: dict[tuple[str, ...], int] = {}
        bigram = pair
        replacement = "".join(pair)
        for word, freq in v_in.items():
            new_word: list[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == bigram:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            v_out[tuple(new_word)] = freq
        return v_out

    def train(self, text: str, target_vocab_size: int = 250, max_merges: int | None = None) -> BPETokenizer:
        """Train BPE on input text until reaching target_vocab_size or max_merges."""
        # 1. Initialize special tokens
        self.special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        self.token2id = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        self.merges = []

        # 2. Get base characters
        unique_chars = sorted(set(text))
        for ch in unique_chars:
            if ch not in self.token2id:
                idx = len(self.token2id)
                self.token2id[ch] = idx

        # 3. Represent text as frequency of character tuples
        words = text.split()
        vocab_words: dict[tuple[str, ...], int] = Counter(
            tuple(list(w) + ["</w>"]) for w in words
        )
        if "</w>" not in self.token2id:
            self.token2id["</w>"] = len(self.token2id)

        # 4. Iteratively merge most frequent pair
        num_merges = 0
        limit = max_merges if max_merges is not None else max(0, target_vocab_size - len(self.token2id))

        while len(self.token2id) < target_vocab_size and num_merges < limit:
            pairs = self._get_stats(vocab_words)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < 1:
                break
            vocab_words = self._merge_vocab(best_pair, vocab_words)
            new_token = "".join(best_pair)
            if new_token not in self.token2id:
                self.token2id[new_token] = len(self.token2id)
                self.merges.append(best_pair)
            num_merges += 1

        self.id2token = {idx: tok for tok, idx in self.token2id.items()}
        return self

    def _tokenize_word(self, word: str) -> list[str]:
        tokens = list(word) + ["</w>"]
        # Handle single characters that may not be in vocab
        for merge_pair in self.merges:
            p0, p1 = merge_pair
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == p0 and tokens[i + 1] == p1:
                    new_tokens.append(p0 + p1)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text to token IDs."""
        words = text.split()
        token_ids: list[int] = []
        if add_special_tokens:
            token_ids.append(self.bos_token_id)

        for w in words:
            word_tokens = self._tokenize_word(w)
            for tok in word_tokens:
                token_ids.append(self.token2id.get(tok, self.unk_token_id))

        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to human-readable string."""
        tokens = []
        for tid in token_ids:
            tok = self.id2token.get(tid, self.UNK_TOKEN)
            if skip_special_tokens and tok in self.special_tokens:
                continue
            tokens.append(tok)

        merged_str = "".join(tokens)
        decoded = merged_str.replace("</w>", " ")
        return decoded.strip()

    def save(self, path: str | Path) -> None:
        """Save vocabulary and merges to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "token2id": self.token2id,
            "merges": [list(pair) for pair in self.merges]
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """Load vocabulary and merges from JSON."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        merges = [tuple(pair) for pair in data.get("merges", [])]
        return cls(vocab=data.get("token2id", {}), merges=merges)
