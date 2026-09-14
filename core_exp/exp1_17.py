"""
Experiment 1 (raw version): exact token counts per ICU stay from icu/chartevents,
restricted to the 17 standard clinical variables used by the MIMIC benchmark /
MedFuse pipelines.

Variables kept (everything else in chartevents is dropped):
  Capillary refill rate, Diastolic blood pressure, Fraction inspired oxygen,
  Glascow coma scale eye opening, Glascow coma scale motor response,
  Glascow coma scale total, Glascow coma scale verbal response, Glucose,
  Heart Rate, Height, Mean blood pressure, Oxygen saturation,
  Respiratory rate, Systolic blood pressure, Temperature, Weight, pH

Advantages over the extracted 48h files:
  - exact full-stay counts, no extrapolation
  - exact crossing time (the day each stay exceeds each budget)
  - counts at any cutoff (24h, 48h, 72h, ...) from one pass

Caveats to state in the paper:
  - chartevents alone excludes labs, notes, and prescriptions
  - the 17-variable restriction drops most of the remaining chartevents rows
  So these counts are a strict LOWER BOUND on real record length. That is the
  point for the motivating experiment: even the minimal serialization overflows
  the budgets.

Inputs (.csv and .csv.gz both work):
  --chartevents   icu/chartevents.csv
  --d-items       icu/d_items.csv       (itemid -> label, used to verify the map)
  --icustays      icu/icustays.csv      (intime, los)
  --stays         optional txt file, one stay_id per line (cohort filter);
                  if omitted and --jsonl given, cohort = stay_ids in the JSONL;
                  if both omitted, ALL stays in icustays (slow, full MIMIC-IV)
  --jsonl         optional extraction JSONL (one or more); adds static summary
                  and radiology report tokens per stay, and mortality labels

Outputs (in --out):
  token_counts.csv       per stay: tokens at 24h/48h/72h/full, crossing days, los
  budget_table.csv       % of stays over 4k/8k/16k/32k at each cutoff
  variable_coverage.csv  per variable: events, stays covered, events per stay
  itemid_map.csv         resolved itemid -> variable -> d_items label
  growth.txt             median crossing day per budget
  hist_tokens.png        distributions at 24h / 48h / full
  tokens_vs_los.png      full-stay tokens vs LOS

Example:
  python exp1_chartevents_tokens_core17.py \
      --chartevents /path/mimic-iv-1.0/icu/chartevents.csv \
      --d-items     /path/mimic-iv-1.0/icu/d_items.csv \
      --icustays    /path/mimic-iv-1.0/icu/icustays.csv \
      --jsonl       splits/train.jsonl \
      --n-samples 100 --out exp1_core_out
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_BUDGETS = [4000, 8000, 16000, 32000]
DEFAULT_CUTOFFS_H = [24, 48, 72]  # plus full stay

# ---------------------------------------------------------------------------
# The 17 clinical variables -> MIMIC-IV (MetaVision) itemids.
# Derived from the MIMIC benchmark itemid_to_variable_map. Verify against your
# own d_items with the itemid_map.csv the script writes; MIMIC-III CareVue
# itemids (e.g. 211, 618, 198) do not exist in MIMIC-IV and are omitted.
# ---------------------------------------------------------------------------
VARIABLE_ITEMIDS = {
    "Capillary refill rate": [223951, 224308],
    "Diastolic blood pressure": [220051, 220180, 224643, 225310, 227242],
    "Fraction inspired oxygen": [223835, 226754, 227009, 227010],
    "Glascow coma scale eye opening": [220739],
    "Glascow coma scale motor response": [223901],
    "Glascow coma scale total": [226755],
    "Glascow coma scale verbal response": [223900],
    "Glucose": [220621, 225664, 226537, 228388],
    "Heart Rate": [220045],
    "Height": [226707, 226730],
    "Mean blood pressure": [220052, 220181, 224322, 225312],
    "Oxygen saturation": [220227, 220277],
    "Respiratory rate": [220210, 224688, 224689, 224690],
    "Systolic blood pressure": [220050, 220179, 224167, 225309, 227243],
    "Temperature": [223761, 223762],
    "Weight": [224639, 226512, 226531],
    "pH": [220274, 220734, 223830],
}
VARIABLE_ORDER = list(VARIABLE_ITEMIDS)

# GCS items carry the clinically meaningful information in the text field
# ("Spontaneously", "Obeys Commands"), so prefer value over valuenum for these.
PREFER_TEXT_ITEMIDS = {220739, 223900, 223901, 223951, 224308}

# Unit harmonisation so one variable does not mix scales across itemids.
UNIT_CONVERT = {
    223761: lambda v: (v - 32.0) * 5.0 / 9.0,   # Temperature F -> C
    226531: lambda v: v * 0.45359237,           # Admission weight lbs -> kg
    226707: lambda v: v * 2.54,                 # Height inches -> cm
}


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

    ser = p.add_argument_group("serialization")
    ser.add_argument("--round-decimals", type=int, default=1,
                     help="round numeric values before serializing (default: 1)")
    ser.add_argument("--no-dedupe-per-timestamp", dest="dedupe_per_timestamp",
                     action="store_false",
                     help="keep the FIRST value per (stay, time, variable) "
                          "instead of the last")
    ser.set_defaults(dedupe_per_timestamp=True)

    args = p.parse_args(argv)
    if args.n_samples is not None and args.n_samples <= 0:
        args.n_samples = None
    return args


def budget_tag(b):
    """4000 -> '4k', 1500 -> '1500'. Used in output column names."""
    return f"{b // 1000}k" if b % 1000 == 0 else str(b)


def build_itemid_map(d_items_path):
    """Resolve the configured itemids against d_items and report the result."""
    d = pd.read_csv(d_items_path, usecols=["itemid", "label"])
    label_by_itemid = d.set_index("itemid")["label"].to_dict()

    rows, itemid_to_var, missing = [], {}, []
    for var, ids in VARIABLE_ITEMIDS.items():
        for iid in ids:
            if iid not in label_by_itemid:
                missing.append((var, iid))
                continue
            itemid_to_var[iid] = var
            rows.append({"variable": var, "itemid": iid,
                         "d_items_label": label_by_itemid[iid]})

    map_df = pd.DataFrame(rows).sort_values(["variable", "itemid"])
    empty = [v for v in VARIABLE_ORDER
             if v not in set(map_df["variable"])] if len(map_df) else VARIABLE_ORDER

    print(f"Resolved {len(itemid_to_var)} itemids across "
          f"{map_df['variable'].nunique() if len(map_df) else 0}/17 variables")
    if missing:
        print("  itemids not present in d_items (ignored):")
        for var, iid in missing:
            print(f"    {iid}  ({var})")
    if empty:
        print("  WARNING: no itemid resolved for:", ", ".join(empty))
    return itemid_to_var, map_df


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


def stream_chartevents(path, wanted, keep_itemids, chunksize):
    """Stream the big file, keep only cohort rows for the 17 variables."""
    keep_cols = ["stay_id", "charttime", "itemid", "value", "valuenum"]
    parts = []
    n_seen = n_kept = 0
    for chunk in pd.read_csv(path, usecols=keep_cols, chunksize=chunksize,
                             dtype={"stay_id": "int64", "itemid": "int64",
                                    "value": "string"},
                             low_memory=False):
        n_seen += len(chunk)
        part = chunk[chunk["itemid"].isin(keep_itemids) &
                     chunk["stay_id"].isin(wanted)]
        if len(part):
            parts.append(part)
            n_kept += len(part)
        print(f"  scanned {n_seen/1e6:.0f}M rows, kept {n_kept}")
    if not parts:
        raise SystemExit("No rows matched the cohort and the 17-variable map.")
    ev = pd.concat(parts, ignore_index=True)
    ev["charttime"] = pd.to_datetime(ev["charttime"])
    return ev


def serialize_lines(g, round_decimals=1, dedupe_per_timestamp=True):
    """One text line per charttime: '[12.5h] Heart Rate: 88; Glucose: 120'.
    Expects a 'variable' column already mapped from itemid.
    Returns (hours_array, lines_list), sorted by time.
    Replace with your EHRSession format before final numbers.
    """
    lines, hours = [], []
    for t, grp in g.groupby("hours", sort=True):
        obs = {}
        for _, r in grp.iterrows():
            var = r["variable"]
            iid = r["itemid"]
            if iid in PREFER_TEXT_ITEMIDS and pd.notna(r["value"]):
                val = r["value"]
            else:
                val = r["valuenum"] if pd.notna(r["valuenum"]) else r["value"]
            if pd.isna(val):
                continue
            if isinstance(val, (int, float, np.integer, np.floating)):
                conv = UNIT_CONVERT.get(iid)
                if conv is not None:
                    val = conv(float(val))
                val = round(float(val), round_decimals)
                if float(val).is_integer():
                    val = int(val)
            if dedupe_per_timestamp or var not in obs:
                obs[var] = val   # last value wins within this timestamp
        if obs:
            ordered = [f"{v}: {obs[v]}" for v in VARIABLE_ORDER if v in obs]
            hours.append(t)
            lines.append(f"[{t:.2f}h] " + "; ".join(ordered))
    return np.array(hours), lines


def main(argv=None):
    args = parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    def ntok(text):
        return len(tok(text, add_special_tokens=False).input_ids) if text else 0

    icu = pd.read_csv(args.icustays,
                      usecols=["subject_id", "hadm_id", "stay_id",
                               "intime", "los"],
                      parse_dates=["intime"])

    itemid_to_var, map_df = build_itemid_map(args.d_items)
    map_df.to_csv(out / "itemid_map.csv", index=False)

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
    ev = stream_chartevents(args.chartevents, wanted, set(itemid_to_var),
                            args.chunksize)

    ev["variable"] = ev["itemid"].map(itemid_to_var)
    ev = ev[ev["variable"].notna()]

    ev = ev.merge(icu[["stay_id", "intime"]], on="stay_id", how="left")
    ev["hours"] = (ev["charttime"] - ev["intime"]).dt.total_seconds() / 3600.0
    ev = ev[ev["hours"] >= 0]

    # which of the 17 variables actually show up, and how often
    cov = (ev.groupby("variable")
             .agg(n_events=("stay_id", "size"),
                  n_stays=("stay_id", "nunique"))
             .reindex(VARIABLE_ORDER)
             .fillna(0).astype(int))
    cov["pct_stays"] = (100 * cov["n_stays"] / ev["stay_id"].nunique()).round(1)
    cov["events_per_stay"] = (cov["n_events"] /
                              cov["n_stays"].replace(0, np.nan)).round(1)
    cov.to_csv(out / "variable_coverage.csv")
    print("\nVariable coverage:\n", cov.to_string())

    rows = []
    stay_groups = ev.groupby("stay_id")
    n_stays = ev["stay_id"].nunique()
    for i, (sid, g) in enumerate(stay_groups):
        hours, lines = serialize_lines(g, args.round_decimals,
                                       args.dedupe_per_timestamp)
        if len(lines) == 0:
            continue
        # tokenize per line, cumulative sum -> tokens at any cutoff
        per_line = np.array([ntok(l) + 1 for l in lines])  # +1 for newline
        cum = np.cumsum(per_line)

        st = static.get(sid, {})
        base = st.get("tok_static", 0) or 0

        row = {"stay_id": sid,
               "tok_static": st.get("tok_static", np.nan),
               "tok_reports": st.get("tok_reports", np.nan),
               "label": st.get("label", np.nan),
               "n_events_full": int(len(g)),
               "n_timesteps_full": int(len(lines)),
               "tok_full": int(base + cum[-1])}
        for h in args.cutoffs_h:
            idx = np.searchsorted(hours, h, side="right") - 1
            row[f"tok_{h:g}h"] = int(base + (cum[idx] if idx >= 0 else 0))
        # exact crossing day per budget
        for b in args.budgets:
            j = np.searchsorted(base + cum, b, side="left")
            row[f"cross_day_{budget_tag(b)}"] = \
                (hours[j] / 24.0) if j < len(cum) else np.nan
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  serialized {i + 1}/{n_stays} stays")

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
    ax.set_xlabel(f"Tokens ({args.tokenizer} tokenizer, "
                  f"{len(VARIABLE_ORDER)} chartevents variables)")
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