"""
grammar_corrector.py

Grammar correction using a trigram language model (LM).

Approach:
  1. Train a trigram LM on correct (target) sentences from training data.
     Uses Laplace (add-1) smoothing.
  2. For each sentence, detect likely grammar errors using simple
     heuristic detectors (subject-verb agreement, tense, articles).
  3. For each flagged position, generate candidate substitutions
     from a small set of common corrections.
  4. Select the candidate sentence with the highest trigram LM
     log-probability.

This is the baseline classical system described in your proposal.
The trigram LM is the core component you implement yourself
(not imported from a library).
"""

import re
import math
import collections
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.spell_corrector import tokenize, detokenize

# Special tokens
BOS = "<s>"   # beginning of sentence
EOS = "</s>"  # end of sentence
UNK = "<unk>" # unknown word


class TrigramLM:
    """
    Trigram language model with Laplace (add-k) smoothing.

    P(w3 | w1, w2) = (count(w1,w2,w3) + k) / (count(w1,w2) + k * |V|)
    """

    def __init__(self, smoothing_k: float = 0.1):
        """
        Args:
            smoothing_k: Laplace smoothing constant.
                         0.1 (add-0.1) works better than full add-1 for small corpora.
        """
        self.k = smoothing_k
        self.unigrams  = collections.Counter()
        self.bigrams   = collections.Counter()
        self.trigrams  = collections.Counter()
        self.vocab     = set()
        self.vocab_size = 0
        self.trained   = False

    def train(self, sentences: list):
        """
        Train the LM on a list of sentences (strings).
        """
        for sentence in sentences:
            tokens = self._prepare(sentence)
            for i, tok in enumerate(tokens):
                self.unigrams[tok] += 1
                self.vocab.add(tok)
                if i >= 1:
                    self.bigrams[(tokens[i-1], tok)] += 1
                if i >= 2:
                    self.trigrams[(tokens[i-2], tokens[i-1], tok)] += 1

        self.vocab.add(UNK)
        self.vocab_size = len(self.vocab)
        self.trained = True

        print(f"[TrigramLM] Trained on {len(sentences)} sentences.")
        print(f"[TrigramLM] Vocab size: {self.vocab_size}")
        print(f"[TrigramLM] Trigrams:   {len(self.trigrams):,}")

    def log_prob(self, sentence: str) -> float:
        """
        Compute log-probability of a sentence under the trigram LM.
        Lower (more negative) = less likely. Higher = more likely/fluent.
        """
        if not self.trained:
            raise RuntimeError("Call train() before log_prob()")

        tokens = self._prepare(sentence)
        log_p = 0.0

        for i in range(2, len(tokens)):
            w1, w2, w3 = tokens[i-2], tokens[i-1], tokens[i]
            w1 = w1 if w1 in self.vocab else UNK
            w2 = w2 if w2 in self.vocab else UNK
            w3 = w3 if w3 in self.vocab else UNK

            numerator   = self.trigrams.get((w1, w2, w3), 0) + self.k
            denominator = self.bigrams.get((w1, w2), 0) + self.k * self.vocab_size

            log_p += math.log(numerator / denominator)

        return log_p

    def perplexity(self, sentence: str) -> float:
        """Compute perplexity of a sentence (lower = more fluent)."""
        tokens = self._prepare(sentence)
        N = max(len(tokens) - 2, 1)
        lp = self.log_prob(sentence)
        return math.exp(-lp / N)

    def _prepare(self, sentence: str) -> list:
        """Add BOS/EOS markers and lowercase tokens."""
        tokens = tokenize(sentence)
        words = [t.lower() for t in tokens if re.match(r"[a-zA-Z']+", t)]
        return [BOS, BOS] + words + [EOS]


class GrammarCorrector:
    """
    Grammar correction using the trigram LM.

    Detection heuristics flag suspicious positions.
    Candidate generation proposes substitutions.
    The LM selects the most fluent candidate.
    """

    # Common corrections for known error patterns
    SVA_FIXES = {
        # Third-person singular present
        "don't": "doesn't",   "doesn't": "don't",
        "wasn't": "weren't",  "weren't": "wasn't",
        "is": "are",          "are": "is",
        "was": "were",        "were": "was",
        "has": "have",        "have": "has",
        "does": "do",         "do": "does",
        "goes": "go",         "go": "goes",
    }

    TENSE_FIXES = {
        "go": "went",   "went": "go",
        "buy": "bought", "bought": "buy",
        "see": "saw",   "saw": "seen",
        "leave": "left", "left": "leave",
        "come": "came", "came": "come",
        "take": "took", "took": "take",
        "make": "made", "made": "make",
        "give": "gave", "gave": "give",
        "begin": "began", "began": "begun",
        "go": "gone",
    }

    VERB_FORM_FIXES = {
        "writed": "wrote",
        "goed": "went",
        "speaked": "spoke",
        "taked": "took",
        "maked": "made",
        "gived": "gave",
        "haved": "had",
        "beed": "been",
        "seen": "saw",
        "went": "gone",
        "did": "done",
        "saw": "seen",
        "took": "taken",
        "wrote": "written",
    }

    ARTICLE_FIXES = {
        "a": "an",
        "an": "a",
        "the": "",
        "": "the",
    }

    def __init__(self, lm: TrigramLM, max_candidates_per_pos: int = 4):
        """
        Args:
            lm: A trained TrigramLM instance.
            max_candidates_per_pos: How many substitutions to try per flagged
                                    position. Keep this small for speed.
        """
        self.lm = lm
        self.max_candidates_per_pos = max_candidates_per_pos

    def correct_sentence(self, sentence: str) -> tuple:
        """
        Attempt to correct grammar errors in a sentence.

        Strategy:
          1. Detect flagged positions (suspicious tokens)
          2. For each flagged position, generate candidate substitutions
          3. Score all resulting candidate sentences with the trigram LM
          4. Return the highest-scoring candidate (or original if no improvement)

        Returns:
            corrected_sentence (str)
            corrections_made (list): list of (original, corrected) pairs
        """
        tokens = tokenize(sentence)
        word_indices = [i for i, t in enumerate(tokens) if re.match(r"[a-zA-Z']+", t)]

        if not word_indices:
            return sentence, []

        # Find positions to flag for possible correction
        flagged = self._detect_flags(tokens, word_indices)

        if not flagged:
            return sentence, []

        # Generate candidate sentences
        best_sentence = sentence
        best_score = self.lm.log_prob(sentence)
        best_change = None

        for idx, candidates_for_pos in flagged:
            for candidate_word in candidates_for_pos[:self.max_candidates_per_pos]:
                candidate_tokens = list(tokens)
                candidate_tokens[idx] = candidate_word
                candidate_sent = detokenize(candidate_tokens)
                score = self.lm.log_prob(candidate_sent)

                if score > best_score:
                    best_score = score
                    best_sentence = candidate_sent
                    best_change = (tokens[idx], candidate_word)

        corrections_made = [best_change] if best_change else []
        return best_sentence, corrections_made

    def correct_batch(self, sentences: list) -> list:
        """Correct a list of sentences."""
        return [self.correct_sentence(s)[0] for s in sentences]

    # ── Error detection heuristics ────────────────────────────────────────────

    def _detect_flags(self, tokens: list, word_indices: list) -> list:
        """
        Return list of (token_index, [candidate_substitutions]) for
        positions that might contain grammar errors.
        """
        flagged = []
        words = [tokens[i].lower() for i in word_indices]

        for pos, idx in enumerate(word_indices):
            tok = tokens[idx].lower()
            candidates = []

            # Subject-verb agreement fixes
            if tok in self.SVA_FIXES:
                candidates.append(self.SVA_FIXES[tok])

            # Tense fixes
            if tok in self.TENSE_FIXES:
                candidates.append(self.TENSE_FIXES[tok])

            # Irregular verb form fixes
            if tok in self.VERB_FORM_FIXES:
                candidates.append(self.VERB_FORM_FIXES[tok])

            # Word-choice confusion: there/their/they're
            if tok in {"there", "their", "they're"}:
                candidates.extend([w for w in ["there", "their", "they're"] if w != tok])

            # Article confusion: a/an
            if tok in {"a", "an"}:
                # Suggest the other article
                candidates.append("an" if tok == "a" else "a")

            # Noun number: add/remove 's'
            if tok.endswith("s") and len(tok) > 3:
                candidates.append(tok[:-1])  # try singular
            elif len(tok) > 2:
                candidates.append(tok + "s")  # try plural

            if candidates:
                flagged.append((idx, candidates))

        return flagged


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data.data_loader import DataLoader

    loader = DataLoader(use_real_data=False)
    _, train_tgt = loader.load_train()

    lm = TrigramLM(smoothing_k=0.1)
    lm.train(train_tgt)

    corrector = GrammarCorrector(lm)

    print("\n=== Grammar Correction Examples ===")
    test_cases = [
        "The students was excited about the new project.",
        "He don't know the answer to the question.",
        "Yesterday, she go to the market and buy vegetables.",
        "She is an honest person and a good leader.",  # Correct
    ]
    for sent in test_cases:
        corrected, changes = corrector.correct_sentence(sent)
        print(f"\nInput:     {sent}")
        print(f"Corrected: {corrected}")
        print(f"Changes:   {changes if changes else 'none'}")
