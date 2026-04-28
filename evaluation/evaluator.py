"""
evaluator.py

Evaluation metrics for GEC (Grammatical Error Correction):

  - Precision:          proportion of proposed corrections that are correct
  - Recall:             proportion of actual errors that were fixed
  - F0.5 Score:         weighted harmonic mean (precision weighted 2x over recall)
  - Sentence Accuracy:  proportion of sentences where full output == gold

All metrics are computed at the token level using word-level diffs.

Usage:
    evaluator = Evaluator()
    results = evaluator.evaluate(sources, system_outputs, gold_targets)
    evaluator.print_results(results)
"""

import re
import difflib
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.spell_corrector import tokenize


class Evaluator:

    def evaluate(self, sources: list, system_outputs: list, gold_targets: list) -> dict:
        """
        Compute all evaluation metrics.

        Args:
            sources:        Original (erroneous) sentences
            system_outputs: System-corrected sentences
            gold_targets:   Gold standard (human-corrected) sentences

        Returns:
            dict with keys: precision, recall, f05, sentence_accuracy,
                            tp, fp, fn, n_sentences, n_with_errors
        """
        assert len(sources) == len(system_outputs) == len(gold_targets), \
            "sources, system_outputs, and gold_targets must have the same length"

        tp_total = 0  # True positives:  correct corrections made
        fp_total = 0  # False positives: incorrect corrections made
        fn_total = 0  # False negatives: errors missed

        sentence_correct = 0

        for src, out, gold in zip(sources, system_outputs, gold_targets):
            tp, fp, fn = self._token_level_counts(src, out, gold)
            tp_total += tp
            fp_total += fp
            fn_total += fn

            if self._normalize(out) == self._normalize(gold):
                sentence_correct += 1

        precision = self._safe_div(tp_total, tp_total + fp_total)
        recall    = self._safe_div(tp_total, tp_total + fn_total)
        f05       = self._f_beta(precision, recall, beta=0.5)
        sent_acc  = sentence_correct / len(sources) if sources else 0.0

        return {
            "precision":         round(precision, 4),
            "recall":            round(recall, 4),
            "f0.5":              round(f05, 4),
            "sentence_accuracy": round(sent_acc, 4),
            "tp":                tp_total,
            "fp":                fp_total,
            "fn":                fn_total,
            "n_sentences":       len(sources),
            "n_with_errors":     sum(1 for s, g in zip(sources, gold_targets)
                                     if self._normalize(s) != self._normalize(g)),
        }

    def evaluate_all_systems(
        self,
        sources: list,
        gold_targets: list,
        systems: dict,         # {system_name: list_of_outputs}
    ) -> dict:
        """
        Evaluate multiple systems at once.

        Args:
            systems: dict mapping system name → list of output sentences

        Returns:
            dict mapping system name → metrics dict
        """
        results = {}
        for name, outputs in systems.items():
            results[name] = self.evaluate(sources, outputs, gold_targets)
        return results

    def print_results(self, results: dict, title: str = "Evaluation Results"):
        """Print a formatted results table for one or multiple systems."""
        print(f"\n{'='*65}")
        print(f"  {title}")
        print(f"{'='*65}")

        # Single system dict
        if "precision" in results:
            self._print_single(results)
        # Multi-system dict
        else:
            header = f"{'System':<35} {'P':>6} {'R':>6} {'F0.5':>6} {'SentAcc':>8}"
            print(header)
            print("-" * 65)
            for name, metrics in sorted(results.items(),
                                        key=lambda x: x[1]["f0.5"], reverse=True):
                print(f"{name:<35} "
                      f"{metrics['precision']:>6.4f} "
                      f"{metrics['recall']:>6.4f} "
                      f"{metrics['f0.5']:>6.4f} "
                      f"{metrics['sentence_accuracy']:>8.4f}")
            print("=" * 65)

    def error_analysis(
        self,
        sources: list,
        system_outputs: list,
        gold_targets: list,
        n_examples: int = 5,
    ):
        """
        Print examples of errors the system makes.
        Useful for the Discussion / Error Analysis section of your paper.
        """
        false_positives = []   # System changed something it shouldn't have
        false_negatives = []   # System missed a real error
        correct_fixes   = []   # System correctly fixed an error

        for src, out, gold in zip(sources, system_outputs, gold_targets):
            src_norm  = self._normalize(src)
            out_norm  = self._normalize(out)
            gold_norm = self._normalize(gold)

            has_real_error = src_norm != gold_norm
            system_changed = src_norm != out_norm
            correct_output = out_norm == gold_norm

            if system_changed and not has_real_error:
                false_positives.append({"source": src, "output": out, "gold": gold})
            elif has_real_error and not system_changed:
                false_negatives.append({"source": src, "output": out, "gold": gold})
            elif has_real_error and correct_output:
                correct_fixes.append({"source": src, "output": out, "gold": gold})

        print(f"\n{'='*65}")
        print("  Error Analysis")
        print(f"{'='*65}")

        print(f"\n✅ Correct Fixes ({len(correct_fixes)} total) — showing up to {n_examples}:")
        for ex in correct_fixes[:n_examples]:
            print(f"  SRC:  {ex['source']}")
            print(f"  OUT:  {ex['output']}")
            print(f"  GOLD: {ex['gold']}")
            print()

        print(f"\n❌ False Positives — system changed correct text ({len(false_positives)} total):")
        for ex in false_positives[:n_examples]:
            print(f"  SRC:  {ex['source']}")
            print(f"  OUT:  {ex['output']}")
            print(f"  GOLD: {ex['gold']}")
            print()

        print(f"\n⚠️  False Negatives — system missed real errors ({len(false_negatives)} total):")
        for ex in false_negatives[:n_examples]:
            print(f"  SRC:  {ex['source']}")
            print(f"  OUT:  {ex['output']}")
            print(f"  GOLD: {ex['gold']}")
            print()

    # ── Core metric computation ───────────────────────────────────────────────

    def _token_level_counts(self, source: str, output: str, gold: str):
        """
        Compute TP, FP, FN by comparing token-level diffs.

        Logic:
          - Edits needed: diff(source_tokens, gold_tokens)   → actual errors
          - Edits made:   diff(source_tokens, output_tokens) → system corrections

          TP = edits made that match edits needed
          FP = edits made that were NOT needed (or wrong)
          FN = edits needed that were NOT made
        """
        src_toks  = self._normalize(source).split()
        out_toks  = self._normalize(output).split()
        gold_toks = self._normalize(gold).split()

        edits_needed = self._get_edits(src_toks, gold_toks)
        edits_made   = self._get_edits(src_toks, out_toks)

        # Use sets of (position, change) tuples for matching
        needed_set = set(edits_needed)
        made_set   = set(edits_made)

        tp = len(made_set & needed_set)
        fp = len(made_set - needed_set)
        fn = len(needed_set - made_set)

        return tp, fp, fn

    def _get_edits(self, src_tokens: list, tgt_tokens: list) -> list:
        """
        Return a list of (position, original_token, new_token) tuples
        representing the token-level edits from src to tgt.
        """
        edits = []
        matcher = difflib.SequenceMatcher(None, src_tokens, tgt_tokens)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            src_span = " ".join(src_tokens[i1:i2])
            tgt_span = " ".join(tgt_tokens[j1:j2])
            edits.append((i1, src_span, tgt_span))
        return edits

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase and collapse whitespace for comparison."""
        return " ".join(text.lower().split())

    @staticmethod
    def _safe_div(num: float, den: float) -> float:
        return num / den if den > 0 else 0.0

    @staticmethod
    def _f_beta(precision: float, recall: float, beta: float) -> float:
        """F-beta score. beta=0.5 weights precision 2x over recall."""
        if precision + recall == 0:
            return 0.0
        b2 = beta ** 2
        return (1 + b2) * precision * recall / (b2 * precision + recall)

    def _print_single(self, metrics: dict):
        print(f"  Sentences evaluated:  {metrics['n_sentences']}")
        print(f"  Sentences with errors:{metrics['n_with_errors']}")
        print(f"  True  Positives (TP): {metrics['tp']}")
        print(f"  False Positives (FP): {metrics['fp']}")
        print(f"  False Negatives (FN): {metrics['fn']}")
        print(f"  Precision:            {metrics['precision']:.4f}")
        print(f"  Recall:               {metrics['recall']:.4f}")
        print(f"  F0.5 Score:           {metrics['f0.5']:.4f}")
        print(f"  Sentence Accuracy:    {metrics['sentence_accuracy']:.4f}")
        print("=" * 65)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    evaluator = Evaluator()

    sources = [
        "She recieved a letter.",
        "The students was excited.",
        "The quick brown fox.",
    ]
    outputs = [
        "She received a letter.",   # correct fix
        "The students was excited.", # missed error
        "The quick brown fox.",      # correct (no change)
    ]
    golds = [
        "She received a letter.",
        "The students were excited.",
        "The quick brown fox.",
    ]

    results = evaluator.evaluate(sources, outputs, golds)
    evaluator.print_results(results, "Quick Test Results")
