"""
Experiment 3: padding control. Length vs content. (CORE SET variant)

For each stay, the clinical content is FROZEN at the 2k-token version from
the truncation ladder. The input is then inflated to 4k / 8k / 16k total
event tokens with clinically plausible but irrelevant filler: serialized
chartevent lines drawn from OTHER stays in the cohort. Two placements:

  after  : [patient record] [filler] [question]
  before : [filler] [patient record] [question]

The same filler text is used for both placements at a given budget, and
larger fillers strictly contain smaller ones, so across all 7 conditions
(baseline + 3 budgets x 2 placements) the decision-relevant content is
identical and only length and position vary.

The filler block is introduced with a header marking it as documentation
not specific to this patient. This is deliberate: it makes the filler
identifiable-in-principle as irrelevant. If it were unlabeled, other
patients' vitals could be mistaken for the patient's own, and a performance
drop could be blamed on genuine ambiguity, which is a content confound.
With the label, an ideal reader is unaffected, and any drop is attributable
to length and position alone.

CORE SET: chartevents rows are filtered to VARIABLE_ITEMIDS below, and each
row is named by its canonical variable name rather than its d_items label.
Serialization is byte-identical to the Exp2 core variant, so the frozen
BASE_BUDGET content here is exactly the 2k rung of that ladder. The filler
now consists of core-set lines too, which makes it distributionally
indistinguishable from the patient's own record apart from the header - the
strongest form of the length-not-content control.

Cohort: identical to Experiments 1 and 2 (same JSONL, SEED, N_SAMPLES,
stratified subsampling, same events cache).

Outputs (in OUT_DIR):
  predictions.csv         stay x condition: p_death, label, tokens
  auroc_by_condition.csv  AUROC + paired-bootstrap CI, delta vs baseline
  positional_effect.txt   before-minus-after AUROC per budget
  padding.png             AUROC vs total budget, one line per placement
  variable_coverage.csv   per-variable event counts and stay coverage
  base_content.txt        how much of each stay the frozen 2k content covers
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

VARIABLE_ORDER = list(VARIABLE_ITEMIDS)
ITEMID_TO_VAR = {iid: name
                 for name, ids in VARIABLE_ITEMIDS.items() for iid in ids}
CORE_ITEMIDS = set(ITEMID_TO_VAR)

# Variables whose free-text `value` is more informative than `valuenum`
USE_TEXT_VALUE_FOR = {
    "Capillary refill rate",
    "Glascow coma scale eye opening",
    "Glascow coma scale motor response",
    "Glascow coma scale verbal response",
}

# ---------------------------------------------------------------------------
# CONFIG (keep identical to Experiment 2 where shared)
# ---------------------------------------------------------------------------
CHARTEVENTS = "/path/to/mimic-iv-1.0/icu/chartevents.csv"
D_ITEMS     = "/path/to/mimic-iv-1.0/icu/d_items.csv"  # unused now: names come from VARIABLE_ITEMIDS
ICUSTAYS    = "/path/to/mimic-iv-1.0/icu/icustays.csv"
STAYS_FILE  = None            # e.g. "cohort_stay_ids.txt", or None
JSONL_FILES = ["/path/to/dataset/splits/train.jsonl"] # e.g. ["train.jsonl"], or None to skip
EVENTS_CACHE = "exp1_out/cohort_events.parquet"  # reused if it exists
                              # NOTE: stores ALL itemids for the cohort
                              # (shared with Exp1/Exp2); core filtering
                              # happens after loading.

MODEL_NAME  = "Qwen/Qwen2.5-VL-7B-Instruct"
OUT_DIR     = "exp3_core_out"

N_SAMPLES   = 100
SEED        = 42

PREDICTION_HOURS = 48.0
BASE_BUDGET = 2000                    # frozen clinical content
PAD_BUDGETS = [4000, 8000, 16000]     # total event tokens after padding
N_BOOTSTRAP = 1000

ROUND_NUMERIC = None          # None = print valuenum as stored (matches Exp1/
                              # Exp2 core). Must match the Exp2 setting or the
                              # frozen 2k content will not be the same text.

SYSTEM_PROMPT = (
    "You are a critical care physician. Based on the ICU record provided, "
    "assess the risk of in-hospital mortality for this patient."
)
QUESTION = (
    "\n\nBased on the record above, will this patient die during this "
    "hospital admission? Answer with exactly one word, Yes or No.\nAnswer:"
)
FILLER_HEADER = (
    "ADDITIONAL UNIT DOCUMENTATION (routine records from other patients "
    "on the unit, not specific to this patient):\n"
)

# ---------------------------------------------------------------------------
# Cohort and events (identical logic to Experiment 2)
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
        n_pos = min(round(N_SAMPLES * len(pos) / len(wanted)), len(pos))
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
    if v is None or pd.isna(v):
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
# Model scoring (identical to Experiment 2)
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
    return torch.softmax(pair.float(), dim=0)[0].item(), \
        inputs["input_ids"].shape[1]


def auroc(y, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p)


def paired_bootstrap(y, preds_by_cond, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    boot = {c: [] for c in preds_by_cond}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        for c, p in preds_by_cond.items():
            boot[c].append(auroc(yb, p[idx]))
    return {c: np.array(v) for c, v in boot.items()}


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

    print("\nSerializing and tokenizing stays (core set) ...")
    stay_data = {}
    for sid, g in ev.groupby("stay_id"):
        lines = serialize_lines(g)
        if not lines:
            continue
        per_line = np.array([ntok(l) + 1 for l in lines])
        s = jsonl_stays.get(sid, {})
        stay_data[sid] = {
            "lines": lines, "per_line": per_line,
            "total": int(per_line.sum()),
            "summary": s.get("patient_summary_text", ""),
            "label": s.get("labels", {}).get("in_hospital_mortality_48hr"),
        }
    stay_data = {sid: d for sid, d in stay_data.items()
                 if d["label"] is not None}
    print(f"{len(stay_data)} stays with core events and labels")

    # Frozen 2k clinical content: same suffix construction as Experiment 2
    for sid, d in stay_data.items():
        cum_rev = np.cumsum(d["per_line"][::-1])
        k = int(np.searchsorted(cum_rev, BASE_BUDGET, side="right"))
        d["base_lines"] = d["lines"][len(d["lines"]) - k:]
        d["base_tokens"] = int(cum_rev[k - 1]) if k else 0
        d["base_is_whole_stay"] = int(k == len(d["lines"]))

    # How much of the record the frozen content actually covers. With the core
    # set many stays fit entirely inside BASE_BUDGET, in which case "baseline"
    # is the full record rather than a truncation of it.
    n_whole = sum(d["base_is_whole_stay"] for d in stay_data.values())
    n_empty = sum(d["base_tokens"] == 0 for d in stay_data.values())
    with open(out / "base_content.txt", "w") as f:
        f.write(f"BASE_BUDGET = {BASE_BUDGET} core event tokens\n")
        f.write(f"stays whose whole core record fits in BASE_BUDGET: "
                f"{n_whole}/{len(stay_data)}\n")
        f.write(f"stays with empty base content (first line alone exceeds "
                f"the budget): {n_empty}\n")
        bt = np.array([d["base_tokens"] for d in stay_data.values()])
        if len(bt):
            f.write(f"base tokens: median {np.median(bt):.0f}, "
                    f"min {bt.min()}, max {bt.max()}\n")
    print(open(out / "base_content.txt").read())
    if n_empty:
        print(f"WARNING: {n_empty} stays have no base content at all.")

    # Filler pool: (line, tokens, source_stay) from all cohort stays
    pool = []
    for sid, d in stay_data.items():
        for line, t in zip(d["lines"], d["per_line"]):
            pool.append((line, int(t), sid))
    pool_tokens = sum(t for _, t, _ in pool)
    print(f"Filler pool: {len(pool)} lines, {pool_tokens} tokens")
    if pool_tokens < max(PAD_BUDGETS) * 1.2:
        print(f"WARNING: pool holds {pool_tokens} tokens; after excluding a "
              f"stay's own lines it may not reach {max(PAD_BUDGETS)}. "
              f"Raise N_SAMPLES or lower PAD_BUDGETS.")

    header_tokens = ntok(FILLER_HEADER)

    def build_filler(sid, n_tokens):
        """Deterministic filler for this stay, excluding its own lines.
        Longer fillers strictly extend shorter ones (prefix property)."""
        if n_tokens <= 0:
            return "", 0
        rng = np.random.default_rng(SEED + sid)
        order = rng.permutation(len(pool))
        lines, used = [], header_tokens
        for j in order:
            line, t, src = pool[j]
            if src == sid:
                continue
            if used + t > n_tokens:
                break
            lines.append(line)
            used += t
        return FILLER_HEADER + "\n".join(lines), used

    # Conditions
    conditions = [("baseline_2k", None, None)]
    for b in PAD_BUDGETS:
        conditions.append((f"{b//1000}k_after", b, "after"))
        conditions.append((f"{b//1000}k_before", b, "before"))

    model, processor, tokenizer, yes_id, no_id = load_model()

    rows = []
    sids = sorted(stay_data)
    for i, sid in enumerate(sids):
        d = stay_data[sid]
        record = ("PATIENT RECORD (this patient, chronological, most recent "
                  "last):\n" + "\n".join(d["base_lines"]))
        # cache fillers per stay so before/after at the same budget share text
        fillers = {b: build_filler(sid, b - d["base_tokens"])
                   for b in PAD_BUDGETS}
        for name, b, placement in conditions:
            if placement is None:
                body, fill_tok = record, 0
            elif placement == "after":
                body, fill_tok = record + "\n\n" + fillers[b][0], fillers[b][1]
            else:
                body, fill_tok = fillers[b][0] + "\n\n" + record, fillers[b][1]
            user_text = ("PATIENT SUMMARY:\n" + d["summary"] + "\n\n" +
                         body + QUESTION)
            p, n_in = score_mortality(model, processor, yes_id, no_id,
                                      user_text)
            rows.append({"stay_id": sid, "condition": name,
                         "p_death": p, "label": d["label"],
                         "base_tokens": d["base_tokens"],
                         "filler_tokens": fill_tok,
                         "base_is_whole_stay": d["base_is_whole_stay"],
                         "input_tokens": n_in})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(sids)} stays scored")
            pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False)

    preds = pd.DataFrame(rows)
    preds.to_csv(out / "predictions.csv", index=False)

    # Metrics
    wide = preds.pivot(index="stay_id", columns="condition",
                       values="p_death")
    y = preds.groupby("stay_id")["label"].first().loc[wide.index].to_numpy()
    cond_names = [c[0] for c in conditions]
    preds_by_cond = {c: wide[c].to_numpy() for c in cond_names}

    boot = paired_bootstrap(y, preds_by_cond, N_BOOTSTRAP, SEED)
    table = []
    for c in cond_names:
        point = auroc(y, preds_by_cond[c])
        lo, hi = np.percentile(boot[c], [2.5, 97.5])
        delta = boot[c] - boot["baseline_2k"]
        dlo, dhi = np.percentile(delta, [2.5, 97.5])
        table.append({"condition": c, "auroc": round(point, 4),
                      "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                      "delta_vs_baseline": round(delta.mean(), 4),
                      "delta_ci_lo": round(dlo, 4),
                      "delta_ci_hi": round(dhi, 4)})
    res = pd.DataFrame(table)
    res.to_csv(out / "auroc_by_condition.csv", index=False)
    print("\n", res.to_string(index=False))

    # positional effect: before minus after at each budget
    with open(out / "positional_effect.txt", "w") as f:
        for b in PAD_BUDGETS:
            d_pos = boot[f"{b//1000}k_before"] - boot[f"{b//1000}k_after"]
            f.write(f"{b//1000}k: before-after AUROC = {d_pos.mean():+.4f} "
                    f"(95% CI {np.percentile(d_pos, 2.5):+.4f}, "
                    f"{np.percentile(d_pos, 97.5):+.4f})\n")
    print(open(out / "positional_effect.txt").read())

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = np.array(PAD_BUDGETS)
    base = res.loc[res.condition == "baseline_2k", "auroc"].item()
    ax.axhline(base, color="black", ls="--", lw=1,
               label=f"baseline 2k ({base:.3f})")
    for placement, marker in [("after", "o"), ("before", "s")]:
        names = [f"{b//1000}k_{placement}" for b in PAD_BUDGETS]
        sub = res.set_index("condition").loc[names]
        yv = sub["auroc"].to_numpy()
        ax.errorbar(x, yv,
                    yerr=[yv - sub["ci_lo"], sub["ci_hi"] - yv],
                    marker=marker, capsize=3, label=f"filler {placement}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b//1000}k" for b in PAD_BUDGETS])
    ax.set_xlabel("Total event tokens (content fixed at 2k)")
    ax.set_ylabel("AUROC")
    ax.set_title(f"Padding control, core set ({len(VARIABLE_ITEMIDS)} vars), "
                 f"n={len(y)} stays")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "padding.png", dpi=200)
    print(f"\nDone. Outputs in {out}/")


if __name__ == "__main__":
    main()