"""
Experiment 3: padding control. Length vs content.

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

Cohort: identical to Experiments 1 and 2 (same JSONL, SEED, N_SAMPLES,
stratified subsampling, same events cache).

Outputs (in OUT_DIR):
  predictions.csv        stay x condition: p_death, label, tokens
  auroc_by_condition.csv AUROC + paired-bootstrap CI, delta vs baseline
  padding.png            AUROC vs total budget, one line per placement
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG (keep identical to Experiment 2 where shared)
# ---------------------------------------------------------------------------
CHARTEVENTS = "/path/to/mimic-iv-1.0/icu/chartevents.csv"
D_ITEMS     = "/path/to/mimic-iv-1.0/icu/d_items.csv"
ICUSTAYS    = "/path/to/mimic-iv-1.0/icu/icustays.csv"
STAYS_FILE  = None            # e.g. "cohort_stay_ids.txt", or None
JSONL_FILES = ["/path/to/dataset/splits/train.jsonl"] # e.g. ["train.jsonl"], or None to skip
EVENTS_CACHE = "exp1_out/cohort_events.parquet"  # reused if it exists


MODEL_NAME  = "Qwen/Qwen2.5-VL-7B-Instruct"
OUT_DIR     = "exp3_out"

N_SAMPLES   = 100
SEED        = 42

PREDICTION_HOURS = 48.0
BASE_BUDGET = 2000                    # frozen clinical content
PAD_BUDGETS = [4000, 8000, 16000]     # total event tokens after padding
N_BOOTSTRAP = 1000

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
    lines = []
    for t, grp in g.groupby("hours", sort=True):
        obs = []
        for _, r in grp.iterrows():
            name = labels_map.get(r["itemid"], f"item{r['itemid']}")
            val = r["valuenum"] if pd.notna(r["valuenum"]) else r["value"]
            if pd.isna(val):
                continue
            obs.append(f"{name}: {val}")
        if obs:
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

    print("Serializing and tokenizing stays ...")
    stay_data = {}
    for sid, g in ev.groupby("stay_id"):
        lines = serialize_lines(g, labels_map)
        if not lines:
            continue
        per_line = np.array([ntok(l) + 1 for l in lines])
        s = jsonl_stays.get(sid, {})
        stay_data[sid] = {
            "lines": lines, "per_line": per_line,
            "summary": s.get("patient_summary_text", ""),
            "label": s.get("labels", {}).get("in_hospital_mortality_48hr"),
        }
    stay_data = {sid: d for sid, d in stay_data.items()
                 if d["label"] is not None}
    print(f"{len(stay_data)} stays with events and labels")

    # Frozen 2k clinical content: same suffix construction as Experiment 2
    for sid, d in stay_data.items():
        cum_rev = np.cumsum(d["per_line"][::-1])
        k = int(np.searchsorted(cum_rev, BASE_BUDGET, side="right"))
        d["base_lines"] = d["lines"][len(d["lines"]) - k:]
        d["base_tokens"] = int(cum_rev[k - 1]) if k else 0

    # Filler pool: (line, tokens, source_stay) from all cohort stays
    pool = []
    for sid, d in stay_data.items():
        for line, t in zip(d["lines"], d["per_line"]):
            pool.append((line, int(t), sid))
    print(f"Filler pool: {len(pool)} lines")

    header_tokens = ntok(FILLER_HEADER)

    def build_filler(sid, n_tokens):
        """Deterministic filler for this stay, excluding its own lines.
        Longer fillers strictly extend shorter ones (prefix property)."""
        if n_tokens <= 0:
            return ""
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
        return FILLER_HEADER + "\n".join(lines)

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
                body = record
            elif placement == "after":
                body = record + "\n\n" + fillers[b]
            else:
                body = fillers[b] + "\n\n" + record
            user_text = ("PATIENT SUMMARY:\n" + d["summary"] + "\n\n" +
                         body + QUESTION)
            p, n_in = score_mortality(model, processor, yes_id, no_id,
                                      user_text)
            rows.append({"stay_id": sid, "condition": name,
                         "p_death": p, "label": d["label"],
                         "base_tokens": d["base_tokens"],
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
    ax.set_title(f"Padding control, n={len(y)} stays")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "padding.png", dpi=200)
    print(f"\nDone. Outputs in {out}/")


if __name__ == "__main__":
    main()