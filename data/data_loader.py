"""
data_loader.py - updated for real CoNLL-2014 .m2 files
"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_M2_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "m2_files")
ANNOTATOR_FILES = ["A1.m2", "A2.m2", "A3.m2", "A4.m2","A5.m2","A6.m2","A7.m2","A8.m2","A9.m2","A10.m2"]

class DataLoader:
    def __init__(self, use_real_data=True, m2_dir=None, train_ratio=0.80, dev_ratio=0.10, random_seed=42):
        self.use_real_data = use_real_data
        self.m2_dir = m2_dir or DEFAULT_M2_DIR
        self.train_ratio = train_ratio
        self.dev_ratio = dev_ratio
        self.random_seed = random_seed
        self._splits = None

    def load_train(self): return self._get_split("train")
    def load_dev(self):   return self._get_split("dev")
    def load_test(self):  return self._get_split("test")

    def print_stats(self):
        for split in ["train", "dev", "test"]:
            src, tgt = self._get_split(split)
            n_err = sum(1 for s, t in zip(src, tgt) if s != t)
            print(f"  {split.capitalize():6}: {len(src):5} sentences ({n_err} with errors, {100*n_err//len(src) if src else 0}%)")

    def _get_split(self, split):
        if self._splits is None:
            self._build_splits()
        s, t = self._splits[split]
        return list(s), list(t)

    def _build_splits(self):
        if not self.use_real_data:
            from data.sample_dataset import get_split
            self._splits = {"train": get_split("train"), "dev": get_split("dev"), "test": get_split("test")}
            return

        all_src, all_tgt, loaded = [], [], []
        for fname in ANNOTATOR_FILES:
            fpath = os.path.join(self.m2_dir, fname)
            if not os.path.exists(fpath):
                continue
            s, t = self._parse_m2(fpath)
            all_src.extend(s); all_tgt.extend(t); loaded.append(fname)

        if not all_src:
            raise FileNotFoundError(f"No .m2 files found in: {self.m2_dir}\nCopy A1.m2, A2.m2, A3.m2, A10.m2 there.")

        print(f"[DataLoader] Loaded {len(all_src)} sentences from: {loaded}")
        indices = list(range(len(all_src)))
        rng = random.Random(self.random_seed)
        rng.shuffle(indices)
        n = len(indices)
        train_end = int(self.train_ratio * n)
        dev_end   = int((self.train_ratio + self.dev_ratio) * n)

        def pick(idx_list):
            return ([all_src[i] for i in idx_list], [all_tgt[i] for i in idx_list])

        self._splits = {"train": pick(indices[:train_end]), "dev": pick(indices[train_end:dev_end]), "test": pick(indices[dev_end:])}

    def _parse_m2(self, filepath):
        sources, targets = [], []
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()
        for block in raw.strip().split("\n\n"):
            lines = block.strip().split("\n")
            if not lines or not lines[0].startswith("S "): continue
            source_tokens = lines[0][2:].strip().split()
            edits = []
            for line in lines[1:]:
                if not line.startswith("A "): continue
                parts = line[2:].split("|||")
                if len(parts) < 3: continue
                span = parts[0].strip().split()
                if len(span) != 2: continue
                try: start, end = int(span[0]), int(span[1])
                except ValueError: continue
                error_type, correction = parts[1].strip(), parts[2].strip()
                if error_type == "noop" or correction == "-NONE-": continue
                edits.append((start, end, correction))
            result = list(source_tokens)
            for start, end, correction in sorted(edits, key=lambda e: e[0], reverse=True):
                if correction == "": del result[start:end]
                else: result[start:end] = correction.split()
            sources.append(" ".join(source_tokens))
            targets.append(" ".join(result))
        return sources, targets

if __name__ == "__main__":
    loader = DataLoader(use_real_data=True)
    try:
        loader.print_stats()
    except FileNotFoundError as e:
        print(e)
