"""
main.py  —  NLP Final Project: Auto Spell and Grammar Check
Authors:    Yash Kumar & Selma Nahas
Course:     CSCI-UA.0469-001, Spring 2026

Runs the full pipeline:
  1. Load data (sample or real CoNLL-2014)
  2. Train all components
  3. Run all systems on dev and test sets
  4. Print evaluation table and error analysis

Usage:
    # With sample data (works out of the box):
    python main.py

    # With real CoNLL-2014 data:
    python main.py --real-data \
        --train data/conll14/train.m2 \
        --dev   data/conll14/dev.m2 \
        --test  data/conll14/test.m2

    # Dev set only (during development):
    python main.py --split dev
"""

import sys
import os
import argparse

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from data.data_loader import DataLoader
from modules.spell_corrector import SpellCorrector
from modules.grammar_corrector import TrigramLM, GrammarCorrector
from baselines.baselines import NoChangeBaseline, DictionaryBaseline, MostFrequentBaseline
from evaluation.evaluator import Evaluator


# ── Full pipeline ─────────────────────────────────────────────────────────────

class SpellGrammarPipeline:
    """
    The full two-stage pipeline:
      Stage 1: SpellCorrector   (Levenshtein + unigram frequency)
      Stage 2: GrammarCorrector (Trigram LM candidate selection)
    """
    name = "Full Pipeline (Spell + Grammar)"

    def __init__(self, spell_corrector: SpellCorrector, grammar_corrector: GrammarCorrector):
        self.spell  = spell_corrector
        self.grammar = grammar_corrector

    def correct_sentence(self, sentence: str) -> tuple:
        after_spell, spell_changes = self.spell.correct_sentence(sentence)
        after_grammar, grammar_changes = self.grammar.correct_sentence(after_spell)
        return after_grammar, spell_changes + grammar_changes

    def correct_batch(self, sentences: list) -> list:
        return [self.correct_sentence(s)[0] for s in sentences]


class SpellOnlyPipeline:
    """Spell correction only — useful for ablation study."""
    name = "Spell Correction Only"

    def __init__(self, spell_corrector: SpellCorrector):
        self.spell = spell_corrector

    def correct_batch(self, sentences: list) -> list:
        return [self.spell.correct_sentence(s)[0] for s in sentences]


class GrammarOnlyPipeline:
    """Grammar correction only — useful for ablation study."""
    name = "Grammar Correction Only"

    def __init__(self, grammar_corrector: GrammarCorrector):
        self.grammar = grammar_corrector

    def correct_batch(self, sentences: list) -> list:
        return [self.grammar.correct_sentence(s)[0] for s in sentences]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NLP Spell & Grammar Correction Pipeline")
    parser.add_argument("--real-data", action="store_true",
                        help="Use real CoNLL-2014 .m2 files instead of sample data")
    parser.add_argument("--m2-dir", default=None, help="Directory containing A1.m2 A2.m2 A3.m2 A10.m2")
    parser.add_argument("--split", choices=["dev", "test", "both"], default="both",
                        help="Which split to evaluate on (default: both)")
    parser.add_argument("--edit-distance", type=int, default=2,
                        help="Max Levenshtein edit distance for spell correction (default: 2)")
    parser.add_argument("--smoothing", type=float, default=0.1,
                        help="Trigram LM Laplace smoothing constant (default: 0.1)")
    parser.add_argument("--error-analysis", action="store_true",
                        help="Print error analysis examples")
    args = parser.parse_args()

    print("\n" + "="*65)
    print("  NLP Final Project: Auto Spell and Grammar Check")
    print("  Yash Kumar & Selma Nahas")
    print("="*65)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    loader = DataLoader(use_real_data=True)

    train_src, train_tgt = loader.load_train()
    dev_src,   dev_tgt   = loader.load_dev()
    test_src,  test_tgt  = loader.load_test()

    print(f"  Train: {len(train_src)} sentences")
    print(f"  Dev:   {len(dev_src)} sentences")
    print(f"  Test:  {len(test_src)} sentences")
    data_mode = "Real CoNLL-2014" 
    print(f"  Mode:  {data_mode}")

    # ── 2. Train components ───────────────────────────────────────────────────
    print(f"\n[2/4] Training components...")

    # Spell corrector trained on CORRECT target sentences
    spell = SpellCorrector(max_edit_distance=args.edit_distance)
    spell.train(train_tgt)

    # Trigram LM trained on CORRECT target sentences
    lm = TrigramLM(smoothing_k=args.smoothing)
    lm.train(train_tgt)

    grammar = GrammarCorrector(lm)

    # Baselines
    no_change    = NoChangeBaseline()
    dict_baseline = DictionaryBaseline()
    dict_baseline.train(train_tgt)
    most_freq    = MostFrequentBaseline()
    most_freq.train(train_tgt)

    # Our systems
    pipeline      = SpellGrammarPipeline(spell, grammar)
    spell_only    = SpellOnlyPipeline(spell)
    grammar_only  = GrammarOnlyPipeline(grammar)

    # ── 3. Run systems and evaluate ───────────────────────────────────────────
    evaluator = Evaluator()

    def run_evaluation(sources, targets, split_name):
        print(f"\n[3/4] Running systems on {split_name} set...")

        systems = {
            "No-Change Baseline":          no_change.correct_batch(sources),
            "Dict Spell Checker Baseline": dict_baseline.correct_batch(sources),
            "Most-Frequent Baseline":      most_freq.correct_batch(sources),
            "Spell Only (ours)":           spell_only.correct_batch(sources),
            "Grammar Only (ours)":         grammar_only.correct_batch(sources),
            "Full Pipeline (ours)":        pipeline.correct_batch(sources),
        }

        print(f"[4/4] Evaluating...")
        all_results = evaluator.evaluate_all_systems(sources, targets, systems)
        evaluator.print_results(all_results, f"{split_name} Set Results")

        if args.error_analysis:
            evaluator.error_analysis(
                sources,
                systems["Full Pipeline (ours)"],
                targets,
                n_examples=3,
            )

        return all_results, systems

    if args.split in ("dev", "both"):
        dev_results, dev_systems = run_evaluation(dev_src, dev_tgt, "Dev")

    if args.split in ("test", "both"):
        print("\n⚠️  Running on TEST set — only do this for final evaluation!")
        test_results, test_systems = run_evaluation(test_src, test_tgt, "Test")

    # ── 4. Show sample outputs ────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  Sample Pipeline Outputs (Dev Set)")
    print("="*65)
    show_sources = dev_src[:5]
    show_golds   = dev_tgt[:5]
    show_outputs = pipeline.correct_batch(show_sources)

    for i, (src, out, gold) in enumerate(zip(show_sources, show_outputs, show_golds), 1):
        match = "✅" if out.lower().strip() == gold.lower().strip() else "❌"
        print(f"\n[{i}] {match}")
        print(f"  SRC:    {src}")
        print(f"  OUTPUT: {out}")
        print(f"  GOLD:   {gold}")

    print("\nDone. Check results above for your paper's Results section.\n")


if __name__ == "__main__":
    main()
