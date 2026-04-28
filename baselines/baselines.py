"""
baselines.py

Three baselines required by your proposal:

  1. NoChangeBaseline    — always return input unchanged
  2. DictionaryBaseline  — dictionary spell checker only (no grammar)
  3. MostFrequentBaseline — replace unknowns with most frequent vocab word
"""

import re
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.spell_corrector import tokenize, detokenize


class NoChangeBaseline:
    """
    Baseline 1: Always return the input sentence unchanged.

    Purpose: lower bound. Any system that makes corrections should
    beat this on F0.5 if it's getting more right than wrong.
    """
    name = "No-Change Baseline"

    def correct_batch(self, sentences: list) -> list:
        return list(sentences)

    def correct_sentence(self, sentence: str) -> tuple:
        return sentence, []


class DictionaryBaseline:
    """
    Baseline 2: Dictionary-based spell checker.

    Corrects only out-of-vocabulary words by replacing them with the
    most frequency-matched word in the dictionary (edit distance 1 only).
    Does NOT perform grammar correction.

    Simpler than SpellCorrector:
      - Only edit distance 1 (more conservative)
      - No grammar module at all
    """
    name = "Dictionary Spell Checker Baseline"

    def __init__(self):
        self.vocab = set()
        self.word_freq = collections.Counter()
        self.trained = False

    def train(self, sentences: list):
        for sentence in sentences:
            for tok in tokenize(sentence):
                word = tok.lower()
                if re.match(r"^[a-z']+$", word) and len(word) > 1:
                    self.word_freq[word] += 1
                    self.vocab.add(word)
        self.trained = True

    def correct_sentence(self, sentence: str) -> tuple:
        tokens = tokenize(sentence)
        corrected_tokens = []
        corrections_made = []

        for tok in tokens:
            word = tok.lower()
            if (not re.match(r"^[a-zA-Z']+$", tok)
                    or len(tok) <= 1
                    or tok[0].isupper()
                    or word in self.vocab):
                corrected_tokens.append(tok)
                continue

            # Try edit distance 1 only
            best = self._best_edit1(word)
            corrected_tokens.append(best if best else tok)
            if best and best != word:
                corrections_made.append((tok, best))

        return detokenize(corrected_tokens), corrections_made

    def correct_batch(self, sentences: list) -> list:
        return [self.correct_sentence(s)[0] for s in sentences]

    def _best_edit1(self, word: str) -> str:
        letters = "abcdefghijklmnopqrstuvwxyz"
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        edits = set(
            [L + R[1:] for L, R in splits if R] +
            [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1] +
            [L + c + R[1:] for L, R in splits if R for c in letters] +
            [L + c + R for L, R in splits for c in letters]
        )
        candidates = [w for w in edits if w in self.vocab]
        if not candidates:
            return None
        return max(candidates, key=lambda w: self.word_freq.get(w, 0))


class MostFrequentBaseline:
    """
    Baseline 3: Replace unknown words with the most frequent vocabulary word.

    This is intentionally naive — it should perform very poorly and serves
    as a sanity-check lower bound to show that random substitution is worse
    than doing nothing.
    """
    name = "Most-Frequent-Word Baseline"

    def __init__(self):
        self.vocab = set()
        self.most_frequent_word = None
        self.trained = False

    def train(self, sentences: list):
        freq = collections.Counter()
        for sentence in sentences:
            for tok in tokenize(sentence):
                word = tok.lower()
                if re.match(r"^[a-z']+$", word) and len(word) > 1:
                    freq[word] += 1
                    self.vocab.add(word)
        self.most_frequent_word = freq.most_common(1)[0][0] if freq else "the"
        self.trained = True
        print(f"[MostFrequentBaseline] Most frequent word: '{self.most_frequent_word}'")

    def correct_sentence(self, sentence: str) -> tuple:
        tokens = tokenize(sentence)
        corrected_tokens = []
        corrections_made = []

        for tok in tokens:
            word = tok.lower()
            if (not re.match(r"^[a-zA-Z']+$", tok)
                    or len(tok) <= 1
                    or tok[0].isupper()
                    or word in self.vocab):
                corrected_tokens.append(tok)
                continue

            corrected_tokens.append(self.most_frequent_word)
            corrections_made.append((tok, self.most_frequent_word))

        return detokenize(corrected_tokens), corrections_made

    def correct_batch(self, sentences: list) -> list:
        return [self.correct_sentence(s)[0] for s in sentences]


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data.data_loader import DataLoader

    loader = DataLoader(use_real_data=False)
    _, train_tgt = loader.load_train()
    test_src, _ = loader.load_test()

    for cls in [NoChangeBaseline, DictionaryBaseline, MostFrequentBaseline]:
        b = cls()
        if hasattr(b, "train"):
            b.train(train_tgt)
        outputs = b.correct_batch(test_src[:3])
        print(f"\n=== {b.name} ===")
        for src, out in zip(test_src[:3], outputs):
            print(f"  IN:  {src}")
            print(f"  OUT: {out}")
