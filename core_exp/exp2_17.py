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

Cohort: identical to Experiment 1 (same JSONL, same --seed, same --n-samples,
same stratified subsampling), so results are directly comparable.

Outputs (in --out):
  predictions.csv         one row per stay x budget: p_death, label, tokens used
  auroc_by_budget.csv     AUROC + paired-bootstrap 95% CI per budget,
                          plus delta vs the smallest budget with CI
  ladder.png              the main figure: AUROC vs budget with CIs
  coverage.txt            how many stays actually reach each budget
  variable_coverage.csv   per-variable event counts and stay coverage
  token_stats.csv         per-stay core-set token totals (for the paper table)

Core-set serialization is roughly 5x more compact than the full variable set,
so the default upper rungs will have thin coverage. Check coverage.txt before
reading the top of the ladder; consider
  --budgets 500 1000 2000 4000 8000

Run on a GPU node (A100 40/80GB is fine for 7B at 32k with bf16):
  python exp2_17.py \
      --chartevents /path/mimic-iv-1.0/icu/chartevents.csv \
      --icustays    /path/mimic-iv-1.0/icu/icustays.csv \
      --jsonl       splits/train.jsonl \
      --events-cache exp1_out/cohort_events.parquet \
      --out exp2_core_out --workers 8 --device cuda --batch-size 2

Parallelism uses CPU worker processes for per-stay serialization/tokenization,
then batched inference through one model instance on the selected device. This
avoids loading one 7B model per worker and exhausting GPU memory.
"""

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_BUDGETS = [2000, 4000, 8000, 16000, 32000]

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

SYSTEM_PROMPT = (
    "You are a critical care physician. Based on the ICU record provided, "
    "assess the risk of in-hospital mortality for this patient."
)
QUESTION = (
    "\n\nBased on the record above, will this patient die during this "
    "hospital admission? Answer with exactly one word, Yes or No.\nAnswer:"
)

_WORKER_TOKENIZER = None
_WORKER_ROUND_NUMERIC = None


def _init_prepare_worker(model_name, round_numeric):
    """Load one tokenizer in each CPU preprocessing worker."""
    global _WORKER_TOKENIZER, _WORKER_ROUND_NUMERIC
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    _WORKER_ROUND_NUMERIC = round_numeric


def _prepare_stay(task):
    """Serialize and tokenize one stay in a CPU worker."""
    sid, g, summary, label = task
    lines = serialize_lines(g, _WORKER_ROUND_NUMERIC)
    if not lines or label is None:
        return None
    encoded = _WORKER_TOKENIZER(lines, add_special_tokens=False).input_ids
    per_line = np.fromiter((len(ids) + 1 for ids in encoded), dtype=np.int64)
    return int(sid), {
        "lines": lines,
        "per_line": per_line,
        "total": int(per_line.sum()),
        "summary": summary,
        "label": label,
    }


def budget_tag(b):
    """4000 -> '4k', 500 -> '500'. Used in column names and plot ticks."""
    return f"{b // 1000}k" if b % 1000 == 0 else str(b)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    io = p.add_argument_group("inputs / outputs")
    io.add_argument("--chartevents", type=Path, required=True,
                    help="path to icu/chartevents.csv[.gz]")
    io.add_argument("--icustays", type=Path, required=True,
                    help="path to icu/icustays.csv[.gz]")
    io.add_argument("--d-items", type=Path, default=None,
                    help="unused in the core variant (names come from "
                         "VARIABLE_ITEMIDS); accepted for CLI parity with the "
                         "full-variable scripts")
    io.add_argument("--stays", type=Path, default=None,
                    help="optional txt file with one stay_id per line "
                         "(takes precedence over --jsonl for cohort selection)")
    io.add_argument("--jsonl", type=Path, nargs="+", default=None,
                    help="extraction JSONL file(s): cohort, patient summaries, "
                         "and labels")
    io.add_argument("--events-cache", type=Path,
                    default=Path("exp1_out/cohort_events.parquet"),
                    help="parquet cache of cohort chartevents. Stores ALL "
                         "itemids, so it is shared with Exp1/Exp3; core "
                         "filtering happens after loading "
                         "(default: exp1_out/cohort_events.parquet)")
    io.add_argument("--out", type=Path, default=Path("exp2_core_out"),
                    help="output directory (default: exp2_core_out)")

    mdl = p.add_argument_group("model")
    mdl.add_argument("--model", dest="model_name",
                     default="Qwen/Qwen2.5-VL-7B-Instruct",
                     help="HF model name or local path "
                          "(default: Qwen/Qwen2.5-VL-7B-Instruct)")
    mdl.add_argument("--device", default="cuda",
                     help="inference device, e.g. cuda, cuda:0, or cpu "
                          "(default: cuda)")
    mdl.add_argument("--batch-size", type=int, default=1,
                     help="number of stay/budget prompts per GPU forward pass "
                          "(default: 1)")

    smp = p.add_argument_group("cohort")
    smp.add_argument("--n-samples", type=int, default=-1,
                     help="subsample cohort to this many stays; "
                          "use 0 or a negative value for all (default: -1)")
    smp.add_argument("--seed", type=int, default=42,
                     help="RNG seed for subsampling and bootstrap (default: 42)")
    smp.add_argument("--label-key", default="in_hospital_mortality_48hr",
                     help="key under 'labels' in the JSONL "
                          "(default: in_hospital_mortality_48hr)")
    smp.add_argument("--chunksize", type=int, default=5_000_000,
                     help="rows per chunk when streaming chartevents "
                          "(default: 5000000)")
    smp.add_argument("--workers", type=int, default=1,
                     help="CPU processes for stay serialization/tokenization "
                          "(default: 1)")

    exp = p.add_argument_group("experiment")
    exp.add_argument("--prediction-hours", type=float, default=48.0,
                     help="only data before this hour feeds the model "
                          "(default: 48)")
    exp.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS,
                     help="event-token budgets "
                          f"(default: {' '.join(map(str, DEFAULT_BUDGETS))})")
    exp.add_argument("--require-top-budget", action="store_true",
                     help="keep only stays with >= the largest budget of event "
                          "tokens (cleanest ladder, fewer stays). Default: keep "
                          "all; short stays reuse their full context and the "
                          "curve flattens")
    exp.add_argument("--n-bootstrap", type=int, default=1000,
                     help="paired bootstrap resamples (default: 1000)")
    exp.add_argument("--round-numeric", type=int, default=None,
                     help="round valuenum to this many decimals before "
                          "printing. Default: print as stored, which matches "
                          "Exp1 and the Exp3 core variant. Changing this "
                          "changes the serialized text, so keep it consistent "
                          "across experiments")

    args = p.parse_args(argv)
    if args.n_samples is not None and args.n_samples <= 0:
        args.n_samples = None
    if args.workers < 1:
        p.error("--workers must be at least 1")
    if args.batch_size < 1:
        p.error("--batch-size must be at least 1")
    args.budgets = sorted(set(args.budgets))
    return args


# ---------------------------------------------------------------------------
# Cohort and events (mirrors Experiment 1)
# ---------------------------------------------------------------------------

def load_jsonl_stays(args):
    stays = {}
    for path in args.jsonl or []:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    s = json.loads(line)
                    stays[s["stay_id"]] = s
    return stays


def load_cohort(args, jsonl_stays):
    if args.stays:
        return set(int(x) for x in Path(args.stays).read_text().split())
    wanted = set(jsonl_stays)
    labels = {sid: s.get("labels", {}).get(args.label_key)
              for sid, s in jsonl_stays.items()}
    if args.n_samples is not None and len(wanted) > args.n_samples:
        rng = np.random.default_rng(args.seed)
        pos = sorted(s for s in wanted if labels.get(s) == 1)
        neg = sorted(s for s in wanted if labels.get(s) == 0)
        n_pos = round(args.n_samples * len(pos) / len(wanted))
        n_pos = min(n_pos, len(pos))
        keep = (list(rng.choice(pos, size=n_pos, replace=False)) +
                list(rng.choice(neg, size=args.n_samples - n_pos,
                                replace=False)))
        wanted = set(int(s) for s in keep)
        print(f"Cohort: {len(wanted)} stays "
              f"({sum(labels[s] for s in wanted)} positive), seed={args.seed}")
    return wanted


def get_events(args, wanted):
    """Cohort chartevents, all itemids (core filtering happens downstream)."""
    cache = Path(args.events_cache)
    if cache.exists():
        print(f"Loading cached events from {cache}")
        ev = pd.read_parquet(cache)
        ev = ev[ev["stay_id"].isin(wanted)]
        if set(ev["stay_id"]) >= wanted:
            return ev
        print("Cache is missing some cohort stays, re-streaming.")
    keep_cols = ["stay_id", "charttime", "itemid", "value", "valuenum"]
    parts, n_seen = [], 0
    for chunk in pd.read_csv(args.chartevents, usecols=keep_cols,
                             chunksize=args.chunksize,
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


def format_value(var, value, valuenum, round_numeric=None):
    """Pick the value to print for one row, or None to skip it."""
    if var in USE_TEXT_VALUE_FOR:
        v = value if pd.notna(value) else valuenum
    else:
        v = valuenum if pd.notna(valuenum) else value
    if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        if round_numeric is not None:
            f = round(f, round_numeric)
        return int(f) if f.is_integer() else f
    v = str(v).strip()
    return v or None


def serialize_lines(g, round_numeric=None):
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
            v = format_value(var, r.value, r.valuenum, round_numeric)
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

def load_model(model_name, device):
    import torch
    from transformers import AutoProcessor, AutoTokenizer
    from transformers import Qwen2_5_VLForConditionalGeneration
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"--device {device!r} requested, but CUDA is unavailable")
    print(f"Loading {model_name} ...")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype, device_map={"": device})
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    return model, processor, tokenizer, yes_id, no_id


def score_mortality_batch(model, processor, yes_id, no_id, user_texts):
    """P(death) for a batch from softmax over Yes/No logits."""
    import torch
    texts = []
    for user_text in user_texts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        texts.append(processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True))
    inputs = processor(text=texts, padding=True,
                       return_tensors="pt").to(model.device)
    with torch.inference_mode():
        logits = model(**inputs).logits[:, -1, :]
    pair = torch.stack([logits[:, yes_id], logits[:, no_id]], dim=1)
    probs = torch.softmax(pair.float(), dim=1)[:, 0].cpu().tolist()
    lengths = inputs["attention_mask"].sum(dim=1).cpu().tolist()
    return probs, [int(n) for n in lengths]


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

def main(argv=None):
    args = parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    jsonl_stays = load_jsonl_stays(args)
    wanted = load_cohort(args, jsonl_stays)

    icu = pd.read_csv(args.icustays, usecols=["stay_id", "intime"],
                      parse_dates=["intime"])

    ev = get_events(args, wanted)
    ev = filter_core(ev)
    ev["charttime"] = pd.to_datetime(ev["charttime"])
    ev = ev.merge(icu, on="stay_id", how="left")
    ev["hours"] = (ev["charttime"] - ev["intime"]).dt.total_seconds() / 3600.0
    ev = ev[(ev["hours"] >= 0) & (ev["hours"] <= args.prediction_hours)]
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

    # Serialize each stay once; store per-line token counts for suffix cuts
    n_groups = ev["stay_id"].nunique()
    print(f"\nSerializing and tokenizing {n_groups} stays with "
          f"{args.workers} CPU worker(s) ...")

    def preparation_tasks():
        for sid, g in ev.groupby("stay_id"):
            s = jsonl_stays.get(int(sid), {})
            yield (int(sid), g,
                   s.get("patient_summary_text", ""),
                   s.get("labels", {}).get(args.label_key))

    worker_args = (args.model_name, args.round_numeric)
    if args.workers == 1:
        _init_prepare_worker(*worker_args)
        prepared = map(_prepare_stay, preparation_tasks())
        stay_data = dict(item for item in prepared if item is not None)
    else:
        # Spawn is safe on SLURM and ensures workers do not inherit any future
        # CUDA state. The pool closes before the GPU model is loaded.
        with ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=get_context("spawn"),
                initializer=_init_prepare_worker,
                initargs=worker_args) as executor:
            prepared = executor.map(_prepare_stay, preparation_tasks(),
                                    chunksize=8)
            # executor.map preserves input order for reproducible output.
            stay_data = dict(item for item in prepared if item is not None)

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
        f.write(f"Stays with >=1 core event in [0, {args.prediction_hours}h]: "
                f"{len(stay_data)}\n\n")
        for b in args.budgets:
            n_ok = sum(d["total"] >= b for d in stay_data.values())
            f.write(f"{b} event tokens: {n_ok}/{len(stay_data)} stays "
                    f"have full coverage\n")
    print("\n" + open(out / "coverage.txt").read())

    top = max(args.budgets)
    n_top = sum(d["total"] >= top for d in stay_data.values())
    if stay_data and n_top < 0.5 * len(stay_data):
        print(f"WARNING: only {n_top}/{len(stay_data)} stays reach the top "
              f"budget ({top}). With the core set most stays saturate below "
              f"this, so the upper rungs mostly repeat the same context. "
              f"Consider lowering --budgets.")

    if args.require_top_budget:
        stay_data = {sid: d for sid, d in stay_data.items()
                     if d["total"] >= top}
        print(f"--require-top-budget: {len(stay_data)} stays kept")

    def build_context(d, budget):
        """Suffix of event lines fitting in `budget` tokens (most recent
        kept; longer budgets strictly contain shorter ones)."""
        cum_rev = np.cumsum(d["per_line"][::-1])
        k = int(np.searchsorted(cum_rev, budget, side="right"))
        kept = d["lines"][len(d["lines"]) - k:]
        return "\n".join(kept), int(cum_rev[k - 1]) if k else 0

    # Inference
    model, processor, tokenizer, yes_id, no_id = load_model(
        args.model_name, args.device)

    rows = []
    sids = sorted(stay_data)
    def iter_inference_items():
        # Generate prompts lazily: materializing every 32k-token prompt at once
        # can consume substantial host RAM on a full cohort.
        for sid in sids:
            d = stay_data[sid]
            for b in args.budgets:
                events_text, used = build_context(d, b)
                user_text = (
                    "PATIENT SUMMARY:\n" + d["summary"] +
                    "\n\nICU EVENTS (chronological, most recent last):\n" +
                    events_text + QUESTION)
                yield {
                    "stay_id": sid, "budget": b, "user_text": user_text,
                    "label": d["label"], "event_tokens_used": used,
                    "saturated": int(used >= d["total"]),
                }

    n_items = len(sids) * len(args.budgets)
    print(f"Scoring {n_items} stay/budget inputs on {args.device} "
          f"with batch size {args.batch_size} ...")
    item_iter = iter_inference_items()
    done = 0
    while True:
        batch = list(itertools.islice(item_iter, args.batch_size))
        if not batch:
            break
        probs, lengths = score_mortality_batch(
            model, processor, yes_id, no_id,
            [item["user_text"] for item in batch])
        for item, prob, n_in in zip(batch, probs, lengths):
            rows.append({
                "stay_id": item["stay_id"],
                "budget": item["budget"],
                "p_death": prob,
                "label": item["label"],
                "event_tokens_used": item["event_tokens_used"],
                "input_tokens": n_in,
                "saturated": item["saturated"],
            })
        done += len(batch)
        if done % (10 * args.batch_size) == 0 or done == n_items:
            print(f"  {done}/{n_items} inputs scored")
            pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False)

    preds = pd.DataFrame(rows)
    preds.to_csv(out / "predictions.csv", index=False)

    # Metrics with paired bootstrap
    wide = preds.pivot(index="stay_id", columns="budget", values="p_death")
    lab = preds.groupby("stay_id")["label"].first().loc[wide.index]
    y = lab.to_numpy()
    preds_by_budget = {b: wide[b].to_numpy() for b in args.budgets}

    boot = paired_bootstrap(y, preds_by_budget, args.n_bootstrap, args.seed)
    ref = min(args.budgets)
    sat = preds.groupby("budget")["saturated"].mean()
    table = []
    for b in args.budgets:
        point = auroc(y, preds_by_budget[b])
        lo, hi = np.percentile(boot[b], [2.5, 97.5])
        delta = boot[b] - boot[ref]
        dlo, dhi = np.percentile(delta, [2.5, 97.5])
        table.append({"budget": b, "auroc": round(point, 4),
                      "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                      f"delta_vs_{budget_tag(ref)}": round(delta.mean(), 4),
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
    x = np.array(args.budgets)
    yv = res["auroc"].to_numpy()
    ax.errorbar(x, yv,
                yerr=[yv - res["ci_lo"], res["ci_hi"] - yv],
                marker="o", capsize=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([budget_tag(b) for b in args.budgets])
    ax.set_xlabel("Event-token budget")
    ax.set_ylabel("AUROC")
    ax.set_title(f"In-hospital mortality, core set ({len(VARIABLE_ITEMIDS)} "
                 f"vars), n={len(y)} stays")
    fig.tight_layout()
    fig.savefig(out / "ladder.png", dpi=200)
    print(f"\nDone. Outputs in {out}/")


if __name__ == "__main__":
    main()
