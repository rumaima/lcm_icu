"""
Experiment 2: truncation ladder (CORE SET variant).

Fixed cohort, identical patients at every budget. For each stay, serialize
chartevents up to the prediction cutoff (48h) using ONLY the 17 core clinical
variables, then build inputs containing the MOST RECENT 2k/4k/8k/16k/32k
event tokens. Each larger context strictly contains each smaller one. Predict
in-hospital mortality at every budget with Qwen2.5-VL-7B and compare AUROC
across budgets with a paired bootstrap.

Difference from the full-variable version: chartevents rows are filtered to
VARIABLE_ITEMIDS below, and each row is named by its canonical variable name
(from the mapping) rather than by its d_items label. Several itemids collapse
onto one variable name, e.g. all five diastolic BP itemids print as
"Diastolic blood pressure". Within a given timestamp, if two itemids for the
same variable both fire, the later-charted one wins (one value per variable
per timestamp).

Scoring: no sampling at all. One forward pass per input; the mortality
probability is softmax over the logits of the "Yes" / "No" tokens at the
answer position. Fully deterministic, no seed variance to worry about.

Cohort: identical to Experiment 1 (same JSONL, same SEED, same N_SAMPLES,
same stratified subsampling), so results are directly comparable.

Outputs (in OUT_DIR):
  predictions.csv         one row per stay x budget: p_death, label, tokens used
  auroc_by_budget.csv     AUROC + paired-bootstrap 95% CI per budget,
                          plus delta vs the smallest budget with CI
  ladder.png              the main figure: AUROC vs budget with CIs
  coverage.txt            how many stays actually reach each budget
  variable_coverage.csv   per-variable event counts and stay coverage
  token_stats.csv         per-stay core-set token totals (for the paper table)

Run on a GPU node (A100 40/80GB is fine for 7B at 32k with bf16):
  python exp2_truncation_ladder_core.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CORE VARIABLE SET (17 variables, MIMIC-IV itemids)
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

# Canonical print order within a timestamp (stable across stays and budgets)
VARIABLE_ORDER = list(VARIABLE_ITEMIDS)
ITEMID_TO_VAR = {iid: name
                 for name, ids in VARIABLE_ITEMIDS.items() for iid in ids}
CORE_ITEMIDS = set(ITEMID_TO_VAR)

# Variables whose free-text `value` is more informative than `valuenum`
# (categorical scales). Everything else prefers valuenum.
USE_TEXT_VALUE_FOR = {
    "Capillary refill rate",
    "Glascow coma scale eye opening",
    "Glascow coma scale motor response",
    "Glascow coma scale verbal response",
}

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CHARTEVENTS = "/path/to/mimic-iv-1.0/icu/chartevents.csv"
D_ITEMS     = "/path/to/mimic-iv-1.0/icu/d_items.csv"  # unused now: names come from VARIABLE_ITEMIDS
ICUSTAYS    = "/path/to/mimic-iv-1.0/icu/icustays.csv"
STAYS_FILE  = None            # e.g. "cohort_stay_ids.txt", or None
JSONL_FILES = ["/path/to/dataset/splits/train.jsonl"] # e.g. ["train.jsonl"], or None to skip
EVENTS_CACHE = "exp1_out/cohort_events.parquet"  # reused if it exists
                              # NOTE: the cache stores ALL itemids for the
                              # cohort (shared with Exp1). Core filtering
                              # happens after loading, so the two experiments
                              # can share one cache.

MODEL_NAME  = "Qwen/Qwen2.5-VL-7B-Instruct"
OUT_DIR     = "exp2_core_out"

N_SAMPLES   = 100
SEED        = 42

PREDICTION_HOURS = 48.0       # only data before this feeds the model
BUDGETS = [2000, 4000, 8000, 16000, 32000]   # event-token budgets
                              # Core-set serialization is roughly 5x more
                              # compact than the full variable set, so the
                              # upper rungs will have thin coverage. Check
                              # coverage.txt before reading the top of the
                              # ladder; consider [500, 1000, 2000, 4000, 8000].
REQUIRE_TOP_BUDGET = False    # True: keep only stays with >= max budget of
                              # event tokens (cleanest ladder, fewer stays).
                              # False: keep all; stays shorter than a budget
                              # reuse their full context (curve flattens).
N_BOOTSTRAP = 1000

ROUND_NUMERIC = None          # None = print valuenum as stored (matches Exp1).
                              # Set to e.g. 2 to round numerics and shave
                              # tokens, but then don't mix with Exp1 counts.

SYSTEM_PROMPT = (
    "You are a critical care physician. Based on the ICU record provided, "
    "assess the risk of in-hospital mortality for this patient."
)
QUESTION = (
    "\n\nBased on the record above, will this patient die during this "
    "hospital admission? Answer with exactly one word, Yes or No.\nAnswer:"
)

BUDGET_COL = lambda b: f"{b//1000}k"

# ---------------------------------------------------------------------------
# Cohort and events (mirrors Experiment 1)
# ---------------------------------------------------------------------------

def load_jsonl_stays():
    stays = {}
    for p in JSONL_FILES or []:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    s = json.loads(line)
                    stays[s["stay_id"]] = s
    return stays


def load_cohort(jsonl_stays):
    if STAYS_FILE:
        return set(int(x) for x in Path(STAYS_FILE).read_text().split())
    wanted = set(jsonl_stays)
    labels = {sid: s.get("labels", {}).get("in_hospital_mortality_48hr")
              for sid, s in jsonl_stays.items()}
    if N_SAMPLES is not None and len(wanted) > N_SAMPLES:
        rng = np.random.default_rng(SEED)
        pos = sorted(s for s in wanted if labels.get(s) == 1)
        neg = sorted(s for s in wanted if labels.get(s) == 0)
        n_pos = round(N_SAMPLES * len(pos) / len(wanted))
        n_pos = min(n_pos, len(pos))
        keep = (list(rng.choice(pos, size=n_pos, replace=False)) +
                list(rng.choice(neg, size=N_SAMPLES - n_pos, replace=False)))
        wanted = set(int(s) for s in keep)
        print(f"Cohort: {len(wanted)} stays "
              f"({sum(labels[s] for s in wanted)} positive), seed={SEED}")
    return wanted


def get_events(wanted):
    """Cohort chartevents, all itemids (core filtering happens downstream)."""
    cache = Path(EVENTS_CACHE)
    if cache.exists():
        print(f"Loading cached events from {cache}")
        ev = pd.read_parquet(cache)
        ev = ev[ev["stay_id"].isin(wanted)]
        if set(ev["stay_id"]) >= wanted:
            return ev
        print("Cache is missing some cohort stays, re-streaming.")
    keep_cols = ["stay_id", "charttime", "itemid", "value", "valuenum"]
    parts, n_seen = [], 0
    for chunk in pd.read_csv(CHARTEVENTS, usecols=keep_cols,
                             chunksize=5_000_000,
                             dtype={"stay_id": "int64", "itemid": "int64",
                                    "value": "string"}, low_memory=False):
        n_seen += len(chunk)
        part = chunk[chunk["stay_id"].isin(wanted)]
        if len(part):
            parts.append(part)
        print(f"  scanned {n_seen/1e6:.0f}M rows")
    ev = pd.concat(parts, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(cache)
    return ev


def filter_core(ev):
    """Keep only the 17 core variables and attach canonical names."""
    n_before = len(ev)
    ev = ev[ev["itemid"].isin(CORE_ITEMIDS)].copy()
    ev["variable"] = ev["itemid"].map(ITEMID_TO_VAR)
    pct = 100.0 * len(ev) / max(n_before, 1)
    print(f"Core filter: {len(ev):,}/{n_before:,} rows kept ({pct:.1f}%), "
          f"{ev['variable'].nunique()}/{len(VARIABLE_ITEMIDS)} variables present")
    return ev


def format_value(var, value, valuenum):
    """Pick the value to print for one row, or None to skip it."""
    if var in USE_TEXT_VALUE_FOR:
        v = value if pd.notna(value) else valuenum
    else:
        v = valuenum if pd.notna(valuenum) else value
    if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        if ROUND_NUMERIC is not None:
            f = round(f, ROUND_NUMERIC)
        return int(f) if f.is_integer() else f
    v = str(v).strip()
    return v or None


def serialize_lines(g):
    """One line per timestamp: '[t h] Var: val; Var: val; ...'

    `g` must be sorted by charttime so that, when two itemids map to the same
    variable at the same timestamp, the later-charted value wins.
    """
    lines = []
    for t, grp in g.groupby("hours", sort=True):
        vals = {}
        for r in grp.itertuples(index=False):
            var = ITEMID_TO_VAR.get(r.itemid)
            if var is None:
                continue
            v = format_value(var, r.value, r.valuenum)
            if v is None:
                continue
            vals[var] = v          # last write wins within the timestamp
        if vals:
            obs = [f"{k}: {vals[k]}" for k in VARIABLE_ORDER if k in vals]
            lines.append(f"[{t:.2f}h] " + "; ".join(obs))
    return lines


# ---------------------------------------------------------------------------
# Model scoring
# ---------------------------------------------------------------------------

def load_model():
    import torch
    from transformers import AutoProcessor, AutoTokenizer
    from transformers import Qwen2_5_VLForConditionalGeneration
    print(f"Loading {MODEL_NAME} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    return model, processor, tokenizer, yes_id, no_id


def score_mortality(model, processor, yes_id, no_id, user_text):
    """P(death) from one forward pass: softmax over Yes/No logits."""
    import torch
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    pair = torch.stack([logits[yes_id], logits[no_id]])
    p_yes = torch.softmax(pair.float(), dim=0)[0].item()
    return p_yes, inputs["input_ids"].shape[1]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def auroc(y, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p)


def paired_bootstrap(y, preds_by_budget, n_boot, seed):
    """Same resample indices for every budget -> comparable CIs."""
    rng = np.random.default_rng(seed)
    n = len(y)
    budgets = list(preds_by_budget)
    boot = {b: [] for b in budgets}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == n:   # need both classes
            continue
        for b in budgets:
            boot[b].append(auroc(yb, preds_by_budget[b][idx]))
    return {b: np.array(v) for b, v in boot.items()}


# ---------------------------------------------------------------------------

def main():
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    jsonl_stays = load_jsonl_stays()
    wanted = load_cohort(jsonl_stays)

    icu = pd.read_csv(ICUSTAYS, usecols=["stay_id", "intime"],
                      parse_dates=["intime"])

    ev = get_events(wanted)
    ev = filter_core(ev)
    ev["charttime"] = pd.to_datetime(ev["charttime"])
    ev = ev.merge(icu, on="stay_id", how="left")
    ev["hours"] = (ev["charttime"] - ev["intime"]).dt.total_seconds() / 3600.0
    ev = ev[(ev["hours"] >= 0) & (ev["hours"] <= PREDICTION_HOURS)]
    ev = ev.sort_values(["stay_id", "charttime", "itemid"])

    # Per-variable coverage over the windowed cohort
    cov = (ev.groupby("variable")
             .agg(n_events=("stay_id", "size"),
                  n_stays=("stay_id", "nunique"))
             .reindex(VARIABLE_ORDER)
             .fillna(0).astype(int))
    cov["pct_stays"] = (100.0 * cov["n_stays"] / max(len(wanted), 1)).round(1)
    cov.to_csv(out / "variable_coverage.csv")
    print("\nPer-variable coverage:\n", cov.to_string())

    from transformers import AutoTokenizer
    tokenizer_only = AutoTokenizer.from_pretrained(MODEL_NAME)

    def ntok(t):
        return len(tokenizer_only(t, add_special_tokens=False).input_ids)

    # Serialize each stay once; store per-line token counts for suffix cuts
    print("\nSerializing and tokenizing stays (core set) ...")
    stay_data = {}
    for sid, g in ev.groupby("stay_id"):
        lines = serialize_lines(g)
        if not lines:
            continue
        per_line = np.array([ntok(l) + 1 for l in lines])
        s = jsonl_stays.get(sid, {})
        stay_data[sid] = {
            "lines": lines,
            "per_line": per_line,
            "total": int(per_line.sum()),
            "summary": s.get("patient_summary_text", ""),
            "label": s.get("labels", {}).get("in_hospital_mortality_48hr"),
        }
    stay_data = {sid: d for sid, d in stay_data.items()
                 if d["label"] is not None}

    # Token stats for the paper table
    tok = pd.DataFrame([{"stay_id": sid,
                         "n_lines": len(d["lines"]),
                         "core_event_tokens": d["total"]}
                        for sid, d in stay_data.items()])
    tok.to_csv(out / "token_stats.csv", index=False)
    if len(tok):
        q = tok["core_event_tokens"].quantile([0.25, 0.5, 0.75])
        print(f"Core event tokens per stay: median {q[0.5]:.0f} "
              f"(IQR {q[0.25]:.0f}-{q[0.75]:.0f}), max "
              f"{tok['core_event_tokens'].max():.0f}")

    # Coverage report
    with open(out / "coverage.txt", "w") as f:
        f.write(f"Core set: {len(VARIABLE_ITEMIDS)} variables, "
                f"{len(CORE_ITEMIDS)} itemids\n")
        f.write(f"Stays with >=1 core event in [0, {PREDICTION_HOURS}h]: "
                f"{len(stay_data)}\n\n")
        for b in BUDGETS:
            n_ok = sum(d["total"] >= b for d in stay_data.values())
            f.write(f"{b} event tokens: {n_ok}/{len(stay_data)} stays "
                    f"have full coverage\n")
    print("\n" + open(out / "coverage.txt").read())

    top = max(BUDGETS)
    n_top = sum(d["total"] >= top for d in stay_data.values())
    if stay_data and n_top < 0.5 * len(stay_data):
        print(f"WARNING: only {n_top}/{len(stay_data)} stays reach the top "
              f"budget ({top}). With the core set most stays saturate below "
              f"this, so the upper rungs mostly repeat the same context. "
              f"Consider lowering BUDGETS.")

    if REQUIRE_TOP_BUDGET:
        stay_data = {sid: d for sid, d in stay_data.items()
                     if d["total"] >= top}
        print(f"REQUIRE_TOP_BUDGET: {len(stay_data)} stays kept")

    def build_context(d, budget):
        """Suffix of event lines fitting in `budget` tokens (most recent
        kept; longer budgets strictly contain shorter ones)."""
        cum_rev = np.cumsum(d["per_line"][::-1])
        k = int(np.searchsorted(cum_rev, budget, side="right"))
        kept = d["lines"][len(d["lines"]) - k:]
        return "\n".join(kept), int(cum_rev[k - 1]) if k else 0

    # Inference
    model, processor, tokenizer, yes_id, no_id = load_model()

    rows = []
    sids = sorted(stay_data)
    for i, sid in enumerate(sids):
        d = stay_data[sid]
        for b in BUDGETS:
            events_text, used = build_context(d, b)
            user_text = (
                "PATIENT SUMMARY:\n" + d["summary"] +
                "\n\nICU EVENTS (chronological, most recent last):\n" +
                events_text + QUESTION)
            p, n_in = score_mortality(model, processor, yes_id, no_id,
                                      user_text)
            rows.append({"stay_id": sid, "budget": b, "p_death": p,
                         "label": d["label"], "event_tokens_used": used,
                         "input_tokens": n_in,
                         "saturated": int(used >= d["total"])})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(sids)} stays scored")
            pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False)

    preds = pd.DataFrame(rows)
    preds.to_csv(out / "predictions.csv", index=False)

    # Metrics with paired bootstrap
    wide = preds.pivot(index="stay_id", columns="budget", values="p_death")
    lab = preds.groupby("stay_id")["label"].first().loc[wide.index]
    y = lab.to_numpy()
    preds_by_budget = {b: wide[b].to_numpy() for b in BUDGETS}

    boot = paired_bootstrap(y, preds_by_budget, N_BOOTSTRAP, SEED)
    ref = min(BUDGETS)
    sat = preds.groupby("budget")["saturated"].mean()
    table = []
    for b in BUDGETS:
        point = auroc(y, preds_by_budget[b])
        lo, hi = np.percentile(boot[b], [2.5, 97.5])
        delta = boot[b] - boot[ref]
        dlo, dhi = np.percentile(delta, [2.5, 97.5])
        table.append({"budget": b, "auroc": round(point, 4),
                      "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                      f"delta_vs_{BUDGET_COL(ref)}": round(delta.mean(), 4),
                      "delta_ci_lo": round(dlo, 4),
                      "delta_ci_hi": round(dhi, 4),
                      "frac_saturated": round(float(sat[b]), 3)})
    res = pd.DataFrame(table)
    res.to_csv(out / "auroc_by_budget.csv", index=False)
    print("\n", res.to_string(index=False))

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = np.array(BUDGETS)
    yv = res["auroc"].to_numpy()
    ax.errorbar(x, yv,
                yerr=[yv - res["ci_lo"], res["ci_hi"] - yv],
                marker="o", capsize=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([BUDGET_COL(b) for b in BUDGETS])
    ax.set_xlabel("Event-token budget")
    ax.set_ylabel("AUROC")
    ax.set_title(f"In-hospital mortality, core set ({len(VARIABLE_ITEMIDS)} "
                 f"vars), n={len(y)} stays")
    fig.tight_layout()
    fig.savefig(out / "ladder.png", dpi=200)
    print(f"\nDone. Outputs in {out}/")


if __name__ == "__main__":
    main()