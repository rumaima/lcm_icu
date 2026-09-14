"""
Experiment 2: truncation ladder.

Fixed cohort, identical patients at every budget. For each stay, serialize
chartevents up to the prediction cutoff (48h), then build inputs containing
the MOST RECENT 2k/4k/8k/16k/32k event tokens. Each larger context strictly
contains each smaller one. Predict in-hospital mortality at every budget with
Qwen2.5-VL-7B and compare AUROC across budgets with a paired bootstrap.

Scoring: no sampling at all. One forward pass per input; the mortality
probability is softmax over the logits of the "Yes" / "No" tokens at the
answer position. Fully deterministic, no seed variance to worry about.

Cohort: identical to Experiment 1 (same JSONL, same SEED, same N_SAMPLES,
same stratified subsampling), so results are directly comparable.

Outputs (in OUT_DIR):
  predictions.csv       one row per stay x budget: p_death, label, tokens used
  auroc_by_budget.csv   AUROC + paired-bootstrap 95% CI per budget,
                        plus delta vs the smallest budget with CI
  ladder.png            the main figure: AUROC vs budget with CIs
  coverage.txt          how many stays actually reach each budget

Run on a GPU node (A100 40/80GB is fine for 7B at 32k with bf16):
  python exp2_truncation_ladder.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CHARTEVENTS = "/path/to/mimic-iv-1.0/icu/chartevents.csv"
D_ITEMS     = "/path/to/mimic-iv-1.0/icu/d_items.csv"
ICUSTAYS    = "/path/to/mimic-iv-1.0/icu/icustays.csv"
STAYS_FILE  = None            # e.g. "cohort_stay_ids.txt", or None
JSONL_FILES = ["/path/to/dataset/splits/train.jsonl"] # e.g. ["train.jsonl"], or None to skip
EVENTS_CACHE = "exp1_out/cohort_events.parquet"  # reused if it exists

MODEL_NAME  = "Qwen/Qwen2.5-VL-7B-Instruct"
OUT_DIR     = "exp2_out"

N_SAMPLES   = 100
SEED        = 42

PREDICTION_HOURS = 48.0       # only data before this feeds the model
BUDGETS = [2000, 4000, 8000, 16000, 32000]   # event-token budgets
REQUIRE_TOP_BUDGET = False    # True: keep only stays with >= max budget of
                              # event tokens (cleanest ladder, fewer stays).
                              # False: keep all; stays shorter than a budget
                              # reuse their full context (curve flattens).
N_BOOTSTRAP = 1000

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


def serialize_lines(g, labels_map):
    lines, hours = [], []
    for t, grp in g.groupby("hours", sort=True):
        obs = []
        for _, r in grp.iterrows():
            name = labels_map.get(r["itemid"], f"item{r['itemid']}")
            val = r["valuenum"] if pd.notna(r["valuenum"]) else r["value"]
            if pd.isna(val):
                continue
            obs.append(f"{name}: {val}")
        if obs:
            hours.append(t)
            lines.append(f"[{t:.2f}h] " + "; ".join(obs))
    return hours, lines


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
    labels_map = pd.read_csv(D_ITEMS, usecols=["itemid", "label"]) \
                   .set_index("itemid")["label"].to_dict()

    ev = get_events(wanted)
    ev["charttime"] = pd.to_datetime(ev["charttime"])
    ev = ev.merge(icu, on="stay_id", how="left")
    ev["hours"] = (ev["charttime"] - ev["intime"]).dt.total_seconds() / 3600.0
    ev = ev[(ev["hours"] >= 0) & (ev["hours"] <= PREDICTION_HOURS)]

    from transformers import AutoTokenizer
    tokenizer_only = AutoTokenizer.from_pretrained(MODEL_NAME)

    def ntok(t):
        return len(tokenizer_only(t, add_special_tokens=False).input_ids)

    # Serialize each stay once; store per-line token counts for suffix cuts
    print("Serializing and tokenizing stays ...")
    stay_data = {}
    for sid, g in ev.groupby("stay_id"):
        hours, lines = serialize_lines(g, labels_map)
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

    # Coverage report
    with open(out / "coverage.txt", "w") as f:
        for b in BUDGETS:
            n_ok = sum(d["total"] >= b for d in stay_data.values())
            f.write(f"{b} event tokens: {n_ok}/{len(stay_data)} stays "
                    f"have full coverage\n")
    print(open(out / "coverage.txt").read())

    if REQUIRE_TOP_BUDGET:
        top = max(BUDGETS)
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
                         "input_tokens": n_in})
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
                      "delta_ci_hi": round(dhi, 4)})
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
    ax.set_title(f"In-hospital mortality, n={len(y)} stays "
                 f"(identical across budgets)")
    fig.tight_layout()
    fig.savefig(out / "ladder.png", dpi=200)
    print(f"\nDone. Outputs in {out}/")


if __name__ == "__main__":
    main()