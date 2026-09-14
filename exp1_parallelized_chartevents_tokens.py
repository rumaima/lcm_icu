"""
Experiment 1 (raw version): exact token counts per ICU stay from icu/chartevents.

Advantages over the extracted 48h files:
  - exact full-stay counts, no extrapolation
  - exact crossing time (the day each stay exceeds each budget)
  - counts at any cutoff (24h, 48h, 72h, ...) from one pass

Caveat: chartevents alone excludes labs, notes, and prescriptions, so these
counts are a LOWER BOUND on real record length. Say so in the paper, or add
labevents with the same pattern.

Inputs (.csv and .csv.gz both work):
  --chartevents   icu/chartevents.csv
  --d-items       icu/d_items.csv       (itemid -> label)
  --icustays      icu/icustays.csv      (intime, los)
  --stays         optional txt file, one stay_id per line (cohort filter);
                  if omitted and --jsonl given, cohort = stay_ids in the JSONL;
                  if both omitted, ALL stays in icustays (slow, full MIMIC-IV)
  --jsonl         optional extraction JSONL (one or more); adds static summary
                  and radiology report tokens per stay, and mortality labels

Outputs (in --out):
  token_counts.csv   per stay: tokens at 24h/48h/72h/full, crossing days, los
  budget_table.csv   % of stays over 4k/8k/16k/32k at each cutoff
  growth.txt         median crossing day per budget
  hist_tokens.png    distributions at 24h / 48h / full
  tokens_vs_los.png  full-stay tokens vs LOS

Example:
  python exp1_chartevents_tokens.py \
      --chartevents /path/mimic-iv-1.0/icu/chartevents.csv \
      --d-items     /path/mimic-iv-1.0/icu/d_items.csv \
      --icustays    /path/mimic-iv-1.0/icu/icustays.csv \
      --jsonl       splits/train.jsonl \
      --n-samples 100 --out exp1_out
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_BUDGETS = [4000, 8000, 16000, 32000]
DEFAULT_CUTOFFS_H = [24, 48, 72]  # plus full stay


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    io = p.add_argument_group("inputs / outputs")
    io.add_argument("--chartevents", type=Path, required=True,
                    help="path to icu/chartevents.csv[.gz]")
    io.add_argument("--d-items", type=Path, required=True,
                    help="path to icu/d_items.csv[.gz] (itemid -> label)")
    io.add_argument("--icustays", type=Path, required=True,
                    help="path to icu/icustays.csv[.gz]")
    io.add_argument("--stays", type=Path, default=None,
                    help="optional txt file with one stay_id per line "
                         "(takes precedence over --jsonl for cohort selection)")
    io.add_argument("--jsonl", type=Path, nargs="+", default=None,
                    help="optional extraction JSONL file(s); supplies the "
                         "cohort, static summary/report text, and labels")
    io.add_argument("--out", type=Path, default=Path("exp1_out"),
                    help="output directory (default: exp1_out)")

    mdl = p.add_argument_group("tokenizer")
    mdl.add_argument("--tokenizer", default="Qwen/Qwen2.5-VL-7B-Instruct",
                     help="HF tokenizer name or local path "
                          "(default: Qwen/Qwen2.5-VL-7B-Instruct)")

    smp = p.add_argument_group("sampling")
    smp.add_argument("--n-samples", type=int, default=-1,
                     help="subsample cohort to this many stays; "
                          "use 0 or a negative value for all (default: -1)")
    smp.add_argument("--seed", type=int, default=42,
                     help="RNG seed for subsampling (default: 42)")
    smp.add_argument("--label-key", default="in_hospital_mortality_48hr",
                     help="key under 'labels' in the JSONL used for stratified "
                          "subsampling (default: in_hospital_mortality_48hr)")

    ana = p.add_argument_group("analysis")
    ana.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS,
                     help="context budgets in tokens "
                          f"(default: {' '.join(map(str, DEFAULT_BUDGETS))})")
    ana.add_argument("--cutoffs-h", type=float, nargs="+",
                     default=DEFAULT_CUTOFFS_H,
                     help="time cutoffs in hours, in addition to the full stay "
                          f"(default: {' '.join(map(str, DEFAULT_CUTOFFS_H))})")
    ana.add_argument("--chunksize", type=int, default=5_000_000,
                     help="rows per chunk when streaming chartevents "
                          "(default: 5000000)")
    ana.add_argument(
        "--workers", type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        help="worker processes (default: SLURM_CPUS_PER_TASK, otherwise 1)",
    )

    args = p.parse_args(argv)
    if args.n_samples is not None and args.n_samples <= 0:
        args.n_samples = None
    if args.workers < 1:
        p.error("--workers must be at least 1")
    return args


def budget_tag(b):
    """4000 -> '4k', 1500 -> '1500'. Used in output column names."""
    return f"{b // 1000}k" if b % 1000 == 0 else str(b)


def load_cohort(args, icu):
    labels = {}  # stay_id -> label, used for stratified subsampling
    if args.stays:
        wanted = set(int(x) for x in Path(args.stays).read_text().split())
    elif args.jsonl:
        wanted = set()
        for path in args.jsonl:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        s = json.loads(line)
                        wanted.add(s["stay_id"])
                        labels[s["stay_id"]] = s.get("labels", {}).get(
                            args.label_key)
    else:
        wanted = set(icu["stay_id"])
        print("WARNING: no cohort filter given, processing ALL stays")

    if args.n_samples is not None and len(wanted) > args.n_samples:
        rng = np.random.default_rng(args.seed)
        if labels and all(v is not None for v in labels.values()):
            # stratified: keep the cohort mortality rate in the subsample
            pos = sorted(s for s in wanted if labels.get(s) == 1)
            neg = sorted(s for s in wanted if labels.get(s) == 0)
            n_pos = round(args.n_samples * len(pos) / len(wanted))
            keep = (list(rng.choice(pos, size=min(n_pos, len(pos)),
                                    replace=False)) +
                    list(rng.choice(neg, size=args.n_samples - min(n_pos, len(pos)),
                                    replace=False)))
            wanted = set(int(s) for s in keep)
            print(f"Subsampled to {len(wanted)} stays "
                  f"({sum(labels[s] for s in wanted)} positive), "
                  f"seed={args.seed}")
        else:
            wanted = set(int(s) for s in
                         rng.choice(sorted(wanted), size=args.n_samples,
                                    replace=False))
            print(f"Subsampled to {len(wanted)} stays (unstratified), "
                  f"seed={args.seed}")
    return wanted


def stream_chartevents(path, wanted, chunksize):
    """Stream the big file, keep only cohort rows and needed columns."""
    keep_cols = ["stay_id", "charttime", "itemid", "value", "valuenum"]
    parts = []
    n_seen = 0
    n_kept = 0
    for chunk in pd.read_csv(path, usecols=keep_cols, chunksize=chunksize,
                             dtype={"stay_id": "int64", "itemid": "int64",
                                    "value": "string"},
                             low_memory=False):
        n_seen += len(chunk)
        part = chunk[chunk["stay_id"].isin(wanted)]
        if len(part):
            parts.append(part)
            n_kept += len(part)
        print(f"  scanned {n_seen/1e6:.0f}M rows, "
              f"kept {n_kept}", flush=True)
    if not parts:
        return pd.DataFrame(columns=keep_cols)
    ev = pd.concat(parts, ignore_index=True)
    ev["charttime"] = pd.to_datetime(ev["charttime"])
    return ev


def serialize_lines(g, labels):
    """One text line per charttime: '[12.5h] Heart Rate: 88; SBP: 120'.
    Returns (hours_array, lines_list), sorted by time.
    Replace with your EHRSession format before final numbers.
    """
    lines, hours = [], []
    for t, grp in g.groupby("hours", sort=True):
        obs = []
        for _, r in grp.iterrows():
            name = labels.get(r["itemid"], f"item{r['itemid']}")
            val = r["valuenum"] if pd.notna(r["valuenum"]) else r["value"]
            if pd.isna(val):
                continue
            obs.append(f"{name}: {val}")
        if obs:
            hours.append(t)
            lines.append(f"[{t:.2f}h] " + "; ".join(obs))
    return np.array(hours), lines


# Set once per child process by init_worker().
_WORKER_TOK = None
_WORKER_LABELS = None
_WORKER_CUTOFFS_H = None
_WORKER_BUDGETS = None


def init_worker(tokenizer_name, labels, cutoffs_h, budgets):
    """Load one tokenizer per worker instead of once per ICU stay."""
    global _WORKER_TOK, _WORKER_LABELS
    global _WORKER_CUTOFFS_H, _WORKER_BUDGETS

    # We parallelize across processes, so internal tokenizer thread pools would
    # oversubscribe the CPUs allocated by Slurm.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    _WORKER_TOK = AutoTokenizer.from_pretrained(tokenizer_name,
                                                trust_remote_code=True)
    _WORKER_LABELS = labels
    _WORKER_CUTOFFS_H = cutoffs_h
    _WORKER_BUDGETS = budgets


def process_stay(task):
    """Serialize and tokenize one stay; executed in a worker process."""
    sid, g, st = task
    hours, lines = serialize_lines(g, _WORKER_LABELS)
    if not lines:
        return None

    # A single batched call has the same per-line counts with much less Python
    # overhead than invoking the tokenizer separately for every line.
    encoded = _WORKER_TOK(lines, add_special_tokens=False,
                          return_length=True, verbose=False)
    lengths = encoded.get("length")
    if lengths is None:
        lengths = [len(ids) for ids in encoded["input_ids"]]
    cum = np.cumsum(np.asarray(lengths, dtype=np.int64) + 1)  # newline

    base = st.get("tok_static", 0) or 0
    row = {
        "stay_id": sid,
        "tok_static": st.get("tok_static", np.nan),
        "tok_reports": st.get("tok_reports", np.nan),
        "label": st.get("label", np.nan),
        "n_events_full": int(len(g)),
        "tok_full": int(base + cum[-1]),
    }
    for h in _WORKER_CUTOFFS_H:
        idx = np.searchsorted(hours, h, side="right") - 1
        row[f"tok_{h:g}h"] = int(base + (cum[idx] if idx >= 0 else 0))
    for b in _WORKER_BUDGETS:
        j = np.searchsorted(base + cum, b, side="left")
        row[f"cross_day_{budget_tag(b)}"] = (
            hours[j] / 24.0 if j < len(cum) else np.nan
        )
    return row


def main(argv=None):
    args = parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    def ntok(text):
        return len(tok(text, add_special_tokens=False).input_ids) if text else 0

    icu = pd.read_csv(args.icustays,
                      usecols=["subject_id", "hadm_id", "stay_id",
                               "intime", "los"],
                      parse_dates=["intime"])
    labels_map = pd.read_csv(args.d_items, usecols=["itemid", "label"]) \
                   .set_index("itemid")["label"].to_dict()

    wanted = load_cohort(args, icu)
    print(f"Cohort: {len(wanted)} stays")

    # optional static text + labels from the extraction jsonl
    static = {}
    if args.jsonl:
        for path in args.jsonl:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    s = json.loads(line)
                    if s["stay_id"] not in wanted:
                        continue
                    static[s["stay_id"]] = {
                        "tok_static": ntok(s.get("patient_summary_text", "")),
                        "tok_reports": ntok(s.get("radiology_report_text", "")),
                        "label": s.get("labels", {}).get(args.label_key),
                    }

    print("Streaming chartevents (one pass over the full file)...")
    ev = stream_chartevents(args.chartevents, wanted, args.chunksize)

    ev = ev.merge(icu[["stay_id", "intime"]], on="stay_id", how="left")
    ev["hours"] = (ev["charttime"] - ev["intime"]).dt.total_seconds() / 3600.0
    ev = ev[ev["hours"] >= 0]

    rows = []
    stay_groups = ev.groupby("stay_id", sort=False)
    n_stays = ev["stay_id"].nunique()
    workers = min(args.workers, max(1, n_stays))
    print(f"Serializing/tokenizing {n_stays} stays with {workers} workers")
    tasks = ((sid, g, static.get(sid, {})) for sid, g in stay_groups)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(args.tokenizer, labels_map, args.cutoffs_h, args.budgets),
    ) as pool:
        for i, row in enumerate(pool.map(process_stay, tasks, chunksize=4), 1):
            if row is not None:
                rows.append(row)
            if i % 100 == 0 or i == n_stays:
                print(f"  serialized {i}/{n_stays} stays", flush=True)

    df = pd.DataFrame(rows).merge(icu.drop(columns=["intime"]),
                                  on="stay_id", how="left")
    df.to_csv(out / "token_counts.csv", index=False)

    # budget table
    conditions = {f"{h:g}h": f"tok_{h:g}h" for h in args.cutoffs_h}
    conditions["full"] = "tok_full"
    table = []
    for name, col in conditions.items():
        v = df[col].dropna()
        row = {"condition": name, "n": len(v),
               "median": int(v.median()), "p90": int(v.quantile(0.9))}
        for b in args.budgets:
            row[f"pct_over_{budget_tag(b)}"] = round(100 * (v > b).mean(), 1)
        table.append(row)
    budget = pd.DataFrame(table)
    budget.to_csv(out / "budget_table.csv", index=False)
    print("\n", budget.to_string(index=False))

    # growth: exact crossing days
    with open(out / "growth.txt", "w") as f:
        for b in args.budgets:
            c = df[f"cross_day_{budget_tag(b)}"].dropna()
            frac = 100 * len(c) / len(df)
            med = c.median() if len(c) else float("nan")
            f.write(f"{b} tokens: {frac:.1f}% of stays ever cross it; "
                    f"median crossing at day {med:.1f}\n")
    print(open(out / "growth.txt").read())

    # plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    fig, ax = plt.subplots(figsize=(6, 4))
    hi = max(df["tok_full"].max(), max(args.budgets) * 1.25)
    bins = np.logspace(np.log10(200), np.log10(hi), 40)
    hist_conditions = [n for n in conditions if n != "full"][:2] + ["full"]
    for name in hist_conditions:
        ax.hist(df[conditions[name]].dropna(), bins=bins, alpha=0.5, label=name)
    for b in args.budgets:
        ax.axvline(b, color="gray", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(f"Tokens ({args.tokenizer} tokenizer, chartevents only)")
    ax.set_ylabel("Stays")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "hist_tokens.png", dpi=200)

    fig, ax = plt.subplots(figsize=(6, 4))
    ok = df[["los", "tok_full"]].dropna()
    rho, _ = spearmanr(ok["los"], ok["tok_full"])
    ax.scatter(ok["los"], ok["tok_full"], s=6, alpha=0.4)
    for b in args.budgets:
        ax.axhline(b, color="gray", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("ICU length of stay (days)")
    ax.set_ylabel("Full-stay tokens")
    ax.set_title(f"Spearman rho = {rho:.2f}")
    fig.tight_layout()
    fig.savefig(out / "tokens_vs_los.png", dpi=200)

    print(f"\nDone. Outputs in {out}/")


if __name__ == "__main__":
    main()
