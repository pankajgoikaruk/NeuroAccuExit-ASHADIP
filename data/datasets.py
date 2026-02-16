# data/datasets.py

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class LogMelDataset(Dataset):
    def __init__(self, segments_csv, features_root, split="train", return_meta: bool = False):
        df = pd.read_csv(segments_csv)

        if "split" not in df.columns:
            raise ValueError(f"'split' column not found in {segments_csv}")
        if "feat_relpath" not in df.columns:
            raise ValueError(
                f"'feat_relpath' column not found in {segments_csv} (did you run extract_features?)"
            )
        if "label" not in df.columns:
            raise ValueError(f"'label' column not found in {segments_csv}")

        self.df = df[df["split"] == split].reset_index(drop=True)
        self.features_root = Path(features_root)
        self.return_meta = bool(return_meta)

        # stable label mapping across splits (use full df, not split df)
        labels = sorted(df["label"].astype(str).unique().tolist())
        self.label2id = {l: i for i, l in enumerate(labels)}

        # normalize paths (optional but helps)
        if "wav_relpath" in self.df.columns:
            self.df["wav_relpath"] = self.df["wav_relpath"].astype(str).str.replace("\\", "/", regex=False)
        self.df["feat_relpath"] = self.df["feat_relpath"].astype(str).str.replace("\\", "/", regex=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        feat_rel = str(row["feat_relpath"]).replace("\\", "/")
        path = self.features_root / Path(feat_rel)

        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")

        S = np.load(path)  # (n_mels, T)
        x = torch.from_numpy(S).float().unsqueeze(0)  # (1, M, T)
        y = torch.tensor(self.label2id[str(row["label"])], dtype=torch.long)

        if not self.return_meta:
            return x, y

        meta = {
            # used for clip-metrics / grouping
            "wav_relpath": str(row["wav_relpath"]).replace("\\", "/") if "wav_relpath" in row else "",
            "start": float(row["start"]) if "start" in row else 0.0,
            "duration": float(row["duration"]) if "duration" in row else 0.0,
            "feat_relpath": feat_rel,
            "split": str(row["split"]) if "split" in row else "",
        }
        return x, y, meta


def _seed_worker(worker_id: int):
    """Ensure each DataLoader worker has a deterministic seed derived from the main seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loaders(
    segments_csv,
    features_root,
    batch_size=64,
    num_workers=4,
    seed=None,
    return_meta: bool = False,   # <-- NEW (default False to avoid breaking other scripts)
):
    ds_tr = LogMelDataset(segments_csv, features_root, "train", return_meta=return_meta)
    ds_va = LogMelDataset(segments_csv, features_root, "val", return_meta=return_meta)
    ds_te = LogMelDataset(segments_csv, features_root, "test", return_meta=return_meta)

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=_seed_worker if seed is not None else None,
        generator=generator,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=_seed_worker if seed is not None else None,
        generator=generator,
    )
    dl_te = DataLoader(
        ds_te,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=_seed_worker if seed is not None else None,
        generator=generator,
    )

    return dl_tr, dl_va, dl_te, ds_tr.label2id
