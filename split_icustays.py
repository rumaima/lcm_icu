#!/usr/bin/env python3
"""
Split icustays.csv into train/test (and optionally val) CSVs according to the
stays listed in train.jsonl / val.jsonl / test.jsonl.

Each JSONL line is expected to carry at least `stay_id`; `subject_id` and
`hadm_id` are used for consistency checking when present.

Usage
-----
    python split_icustays.py \
        --icustays icustays.csv \
        --splits-dir /path/to/splits \
        --out-dir /path/to/output \
        --val-mode merge

--val-mode:
    merge     val stays go into train_icustays.csv          (default)
    separate  val stays go into their own val_icustays.csv
    drop      val stays are excluded entirely
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

KEYS = ["subject_id", "hadm_id", "stay_id"]


def load_jsonl(path: Path) -> pd.DataFrame:
    """Read a split file and return a frame of whichever key columns it has."""
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    rows = []
    with path.open("r") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path.name}: malformed JSON on line {i}: {e}") from e
            rows.append({k: rec.get(k) for k in KEYS})

    if not rows:
        raise ValueError(f"{path.name} is empty")

    df = pd.DataFrame(rows)
    if df["stay_id"].isna().any():
        raise ValueError(f"{path.name}: some records are missing `stay_id`")

    df["stay_id"] = df["stay_id"].astype("int64")
    for k in ("subject_id", "hadm_id"):
        if df[k].notna().all():
            df[k] = df[k].astype("int64")

    n_raw = len(df)
    df = df.drop_duplicates(subset="stay_id")
    if len(df) < n_raw:
        print(f"  [{path.name}] dropped {n_raw - len(df)} duplicate stay_id rows")

    return df


def check_key_consistency(split_df: pd.DataFrame, icu: pd.DataFrame, name: str) -> None:
    """Verify subject_id/hadm_id in the split agree with icustays for each stay_id."""
    cols = [c for c in ("subject_id", "hadm_id") if split_df[c].notna().all()]
    if not cols:
        return

    merged = split_df.merge(
        icu[KEYS], on="stay_id", how="inner", suffixes=("_split", "_icu")
    )
    for c in cols:
        mismatch = merged[merged[f"{c}_split"] != merged[f"{c}_icu"]]
        if not mismatch.empty:
            print(
                f"  [WARN] {name}: {len(mismatch)} stays where {c} disagrees with "
                f"icustays.csv (e.g. stay_id={mismatch['stay_id'].iloc[0]})"
            )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--icustays", type=Path, default=Path("/path/to/mimic-iv-1.0/icu/icustays.csv"))
    p.add_argument("--splits-dir", type=Path, default=Path("/path/to/dataset/splits/"),
                   help="directory holding train.jsonl / val.jsonl / test.jsonl")
    p.add_argument("--out-dir", type=Path, default=Path("."))
    p.add_argument("--val-mode", choices=["merge", "separate", "drop"], default="separate",
                   help="what to do with val stays (default: merge into train)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {args.icustays}")
    icu = pd.read_csv(args.icustays)
    missing = [k for k in KEYS if k not in icu.columns]
    if missing:
        print(f"ERROR: icustays.csv is missing columns: {missing}", file=sys.stderr)
        return 1

    icu["stay_id"] = icu["stay_id"].astype("int64")
    if icu["stay_id"].duplicated().any():
        n = int(icu["stay_id"].duplicated().sum())
        print(f"  [WARN] icustays.csv has {n} duplicate stay_id rows; keeping first")
        icu = icu.drop_duplicates(subset="stay_id", keep="first")
    print(f"  {len(icu):,} ICU stays")

    print("\nReading split files")
    splits = {name: load_jsonl(args.splits_dir / f"{name}.jsonl")
              for name in ("train", "val", "test")}
    for name, df in splits.items():
        print(f"  {name}.jsonl: {len(df):,} stays")
        check_key_consistency(df, icu, name)

    # Leakage check across splits.
    names = list(splits)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = set(splits[a]["stay_id"]) & set(splits[b]["stay_id"])
            if overlap:
                print(f"  [WARN] {len(overlap)} stay_id(s) appear in both "
                      f"{a} and {b} — possible leakage")

    # Assemble output groups.
    if args.val_mode == "merge":
        groups = {
            "train": pd.concat([splits["train"], splits["val"]])
                       .drop_duplicates(subset="stay_id"),
            "test": splits["test"],
        }
    elif args.val_mode == "separate":
        groups = {"train": splits["train"], "val": splits["val"], "test": splits["test"]}
    else:
        groups = {"train": splits["train"], "test": splits["test"]}

    print(f"\nWriting outputs (val-mode={args.val_mode})")
    written = 0
    for name, split_df in groups.items():
        wanted = set(split_df["stay_id"])
        out = icu[icu["stay_id"].isin(wanted)].copy()

        not_found = wanted - set(out["stay_id"])
        if not_found:
            print(f"  [WARN] {name}: {len(not_found)} stay_id(s) from the split "
                  f"file are absent from icustays.csv "
                  f"(e.g. {sorted(not_found)[:5]})")

        out = out.sort_values("stay_id").reset_index(drop=True)
        out_path = args.out_dir / f"{name}_icustays.csv"
        out.to_csv(out_path, index=False)
        written += len(out)
        print(f"  {out_path}: {len(out):,} rows, "
              f"{out['subject_id'].nunique():,} unique subjects")

    unassigned = len(icu) - written
    if unassigned:
        print(f"\n{unassigned:,} of {len(icu):,} stays in icustays.csv "
              f"are not in any split.")

    return 0


if __name__ == "__main__":
    sys.exit(main())