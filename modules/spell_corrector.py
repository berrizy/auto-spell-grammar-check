"""
spell_corrector.py

Spell correction using:
  1. Levenshtein (edit) distance to generate candidates
  2. Unigram frequency ranking to select the best candidate

Training:
  - Build vocabulary and word frequency counts from the training corpus

Correction:
  - For each token not in vocabulary, find all vocab words within
    edit distance k (default k=2) and pick the most frequent one.
  - Tokens starting with uppercase (potential proper nouns) and
    numeric tokens are skipped.
"""

import re
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def tokenize(text: str) -> list:
    """
    Simple word tokenizer that preserves punctuation as separate tokens.
    No external libraries required.
    """
    # Split on whitespace and common punctuation boundaries
    tokens = re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)
    return tokens


def detokenize(tokens: list) -> str:
    """Rejoin tokens into a sentence with basic punctuation handling."""
    if not tokens:
        return ""
    result = tokens[0]
    for tok in tokens[1:]:
        if tok in {",", ".", "!", "?", ";", ":", "'s", "n't", "'re", "'ve", "'ll", "'d", "'m"}:
            result += tok
        else:
            result += " " + tok
    return result


class SpellCorrector:
    def __init__(self, max_edit_distance: int = 2):
        """
        Args:
            max_edit_distance: Maximum Levenshtein distance for candidates (k).
                               k=1 → very conservative (fewer corrections, higher precision)
                               k=2 → balanced (default)
                               k=3 → aggressive (more corrections, lower precision)
        """
        self.max_edit_distance = max_edit_distance
        self.vocab = set()
        self.word_freq = collections.Counter()
        self.trained = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, sentences: list):
        """
        Build vocabulary and unigram frequency table from training sentences.

        Args:
            sentences: List of correct (target) sentences from training split.
        """
        for sentence in sentences:
            tokens = tokenize(sentence)
            for tok in tokens:
                word = tok.lower()
                if self._is_word(word):
                    self.word_freq[word] += 1
                    self.vocab.add(word)

        self.trained = True
        print(f"[SpellCorrector] Trained on {len(sentences)} sentences.")
        print(f"[SpellCorrector] Vocabulary size: {len(self.vocab):,} words")
        print(f"[SpellCorrector] Top 10 words: {self.word_freq.most_common(10)}")

    # ── Correction ────────────────────────────────────────────────────────────

    def correct_sentence(self, sentence: str) -> tuple:
        """
        Correct spelling errors in a sentence.

        Returns:
            corrected_sentence (str): sentence after spell correction
            corrections_made (list): list of (original, corrected) pairs
        """
        if not self.trained:
            raise RuntimeError("Call train() before correct_sentence()")

        tokens = tokenize(sentence)
        corrected_tokens = []
        corrections_made = []

        for tok in tokens:
            corrected = self._correct_token(tok)
            corrected_tokens.append(corrected)
            if corrected.lower() != tok.lower():
                corrections_made.append((tok, corrected))

        return detokenize(corrected_tokens), corrections_made

    def correct_batch(self, sentences: list) -> list:
        """Correct a list of sentences. Returns list of corrected sentences."""
        return [self.correct_sentence(s)[0] for s in sentences]

    # ── Core algorithm ────────────────────────────────────────────────────────

    def _correct_token(self, token: str) -> str:
        """
        Attempt to correct a single token.
        Returns the original token if no correction is needed or found.
        """
        # Skip punctuation, numbers, and likely proper nouns (capitalized)
        if not self._is_word(token):
            return token
        if token[0].isupper():  # Likely proper noun
            return token
        if token.lower() in self.vocab:  # Already correct
            return token

        # Generate candidates within edit distance k
        candidates = self._get_candidates(token.lower())
        if not candidates:
            return token  # No candidate found → leave unchanged

        # Rank by unigram frequency and return the best
        best = max(candidates, key=lambda w: self.word_freq.get(w, 0))

        # Preserve original capitalisation
        if token[0].isupper():
            best = best.capitalize()
        return best

    def _get_candidates(self, word: str) -> list:
        """
        Return all vocabulary words within max_edit_distance of word.
        Uses iterative expansion: first try distance 1, then 2.
        """
        # Fast path: check edits at distance 1 first (most common corrections)
        edits1 = self._edits_n(word, 1)
        candidates_1 = [w for w in edits1 if w in self.vocab]
        if candidates_1:
            return candidates_1  # Prefer minimal-distance corrections

        if self.max_edit_distance >= 2:
            edits2 = self._edits_n(word, 2)
            candidates_2 = [w for w in edits2 if w in self.vocab]
            if candidates_2:
                return candidates_2

        return []

    def _edits_n(self, word: str, n: int) -> set:
        """Generate all strings at edit distance exactly n from word."""
        if n == 0:
            return {word}
        result = set()
        for w in self._edits1(word):
            result.update(self._edits_n(w, n - 1))
        return result

    def _edits1(self, word: str) -> set:
        """
        Generate all strings at edit distance 1 from word.
        Operations: deletion, transposition, replacement, insertion.
        """
        letters = "abcdefghijklmnopqrstuvwxyz"
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

        deletions     = [L + R[1:]           for L, R in splits if R]
        transposes    = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces      = [L + c + R[1:]       for L, R in splits if R for c in letters]
        inserts       = [L + c + R           for L, R in splits for c in letters]

        return set(deletions + transposes + replaces + inserts)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _is_word(token: str) -> bool:
        """Return True if token is a real alphabetic word (not punctuation/number)."""
        return bool(re.match(r"^[a-zA-Z']+$", token)) and len(token) > 1

    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Standard dynamic-programming Levenshtein distance.
        Used in evaluation to verify candidate distances.
        """
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j],      # deletion
                                       dp[i][j - 1],       # insertion
                                       dp[i - 1][j - 1])   # substitution
        return dp[m][n]


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.data_loader import DataLoader

    loader = DataLoader(use_real_data=False)
    _, train_tgt = loader.load_train()

    corrector = SpellCorrector(max_edit_distance=2)
    corrector.train(train_tgt)

    print("\n=== Spell Correction Examples ===")
    test_cases = [
        "She recieved a letter from her freind yesterday.",
        "The goverment decided to impliment new policys.",
        "He beleived that knowlege is power.",
        "The quick brown fox jumps over the lazy dog.",  # No errors
    ]
    for sent in test_cases:
        corrected, changes = corrector.correct_sentence(sent)
        print(f"\nInput:     {sent}")
        print(f"Corrected: {corrected}")
        print(f"Changes:   {changes if changes else 'none'}")
