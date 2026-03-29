"""
vocabulary.py — Construction du vocabulaire à partir des captions.
"""

import os
import json
from collections import Counter


class Vocabulary:
    """Gère le mapping mot <-> index pour le décodeur."""

    PAD_TOKEN = "<pad>"
    START_TOKEN = "<start>"
    END_TOKEN = "<end>"
    UNK_TOKEN = "<unk>"

    def __init__(self, freq_threshold: int = 5):
        self.freq_threshold = freq_threshold
        self.word2idx = {}
        self.idx2word = {}
        self._init_special_tokens()

    def _init_special_tokens(self):
        self.word2idx = {
            self.PAD_TOKEN: 0,
            self.START_TOKEN: 1,
            self.END_TOKEN: 2,
            self.UNK_TOKEN: 3,
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
        return len(self.word2idx)

    def build_vocabulary(self, sentence_list: list):
        """Construit le vocabulaire à partir d'une liste de phrases."""
        counter = Counter()
        for sentence in sentence_list:
            tokens = self.tokenize(sentence)
            counter.update(tokens)

        idx = len(self.word2idx)
        for word, count in counter.items():
            if count >= self.freq_threshold:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

        print(f"[Vocabulary] {len(self.word2idx)} mots "
              f"(seuil de fréquence = {self.freq_threshold})")

    @staticmethod
    def tokenize(text: str) -> list:
        """Tokenisation simple : lowercase + split sur espaces."""
        return text.lower().strip().split()

    def numericalize(self, text: str) -> list:
        """Convertit une phrase en liste d'indices (avec <start>/<end>)."""
        tokens = self.tokenize(text)
        indices = [self.word2idx[self.START_TOKEN]]
        for token in tokens:
            indices.append(
                self.word2idx.get(token, self.word2idx[self.UNK_TOKEN])
            )
        indices.append(self.word2idx[self.END_TOKEN])
        return indices

    def decode(self, indices: list) -> str:
        """Convertit une liste d'indices en phrase lisible."""
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, self.UNK_TOKEN)
            if word == self.END_TOKEN:
                break
            if word not in (self.START_TOKEN, self.PAD_TOKEN):
                words.append(word)
        return " ".join(words)

    def save(self, path: str):
        data = {
            "freq_threshold": self.freq_threshold,
            "word2idx": self.word2idx,
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[Vocabulary] Sauvegardé dans {path}")

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, "r") as f:
            data = json.load(f)
        vocab = cls(freq_threshold=data["freq_threshold"])
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(v): k for k, v in data["word2idx"].items()}
        print(f"[Vocabulary] Chargé depuis {path} — {len(vocab)} mots")
        return vocab