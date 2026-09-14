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

Cohort: identical to Experiment 1 (same JSONL, same --seed, same --n-samples,
same stratified subsampling), so results are directly comparable.

Outputs (in --out):
  predictions.csv       one row per stay x budget: p_death, label, tokens used
  auroc_by_budget.csv   AUROC + paired-bootstrap 95% CI per budget,
                        plus delta vs the smallest budget with CI
  ladder.png            the main figure: AUROC vs budget with CIs
  coverage.txt          how many stays actually reach each budget

Run on a GPU node (A100 40/80GB is fine for 7B at 32k with bf16):
  python exp2_truncation_ladder.py \
      --chartevents /path/mimic-iv-1.0/icu/chartevents.csv \
      --d-items     /path/mimic-iv-1.0/icu/d_items.csv \
      --icustays    /path/mimic-iv-1.0/icu/icustays.csv \
      --jsonl       splits/train.jsonl \
      --events-cache exp1_out/cohort_events.parquet \
      --out exp2_out
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_BUDGETS = [2000, 4000, 8000, 16000, 32000]

SYSTEM_PROMPT = (
    "You are a critical care physician. Based on the ICU record provided, "
    "assess the risk of in-hospital mortality for this patient."
)
QUESTION = (
    "\n\nBased on the record above, will this patient die during this "
    "hospital admission? Answer with exactly one word, Yes or No.\nAnswer:"
)


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
    io.add_argument("--d-items", type=Path, required=True,
                    help="path to icu/d_items.csv[.gz] (itemid -> label)")
    io.add_argument("--icustays", type=Path, required=True,
                    help="path to icu/icustays.csv[.gz]")
    io.add_argument("--stays", type=Path, default=None,
                    help="optional txt file with one stay_id per line "
                         "(takes precedence over --jsonl for cohort selection)")
    io.add_argument("--jsonl", type=Path, nargs="+", default=None,
                    help="extraction JSONL file(s): cohort, patient summaries, "
                         "and labels")
    io.add_argument("--events-cache", type=Path,
                    default=Path("exp1_out/cohort_events.parquet"),
                    help="parquet cache of cohort chartevents, shared with "
                         "Exp1/Exp3 (default: exp1_out/cohort_events.parquet)")
    io.add_argument("--out", type=Path, default=Path("exp2_out"),
                    help="output directory (default: exp2_out)")

    mdl = p.add_argument_group("model")
    mdl.add_argument("--model", dest="model_name",
                     default="Qwen/Qwen2.5-VL-7B-Instruct",
                     help="HF model name or local path "
                          "(default: Qwen/Qwen2.5-VL-7B-Instruct)")
    mdl.add_argument("--device", default="cuda",
                     help="inference device (default: cuda)")
    mdl.add_argument("--batch-size", type=int, default=1,
                     help="GPU inference batch size; start with 1 for 32k "
                          "contexts on a 40GB GPU (default: 1)")

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
    smp.add_argument(
        "--workers", type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        help="CPU preprocessing workers (default: SLURM_CPUS_PER_TASK, "
             "otherwise 1)",
    )

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
    cache = Path(args.events_cache)
    if cache.exists():
        print(f"Loading cached events from {cache}")
        ev = pd.read_parquet(cache)
        ev = ev[ev["stay_id"].isin(wanted)]
        if set(ev["stay_id"]) >= wanted:
            return ev
        print("Cache is missing some cohort stays, re-streaming.")
    keep_cols = ["stay_id", "charttime", "itemid", "value", "valuenum"]
    parts, n_seen, n_kept = [], 0, 0
    for chunk in pd.read_csv(args.chartevents, usecols=keep_cols,
                             chunksize=args.chunksize,
                             dtype={"stay_id": "int64", "itemid": "int64",
                                    "value": "string"}, low_memory=False):
        n_seen += len(chunk)
        part = chunk[chunk["stay_id"].isin(wanted)]
        if len(part):
            parts.append(part)
            n_kept += len(part)
        print(f"  scanned {n_seen/1e6:.0f}M rows, kept {n_kept}",
              flush=True)
    if not parts:
        raise RuntimeError("No chartevents rows matched the selected cohort")
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


_WORKER_TOKENIZER = None
_WORKER_LABELS_MAP = None
_WORKER_JSONL_STAYS = None
_WORKER_LABEL_KEY = None


def init_preprocessing_worker(model_name, labels_map, jsonl_stays, label_key):
    """Initialize CPU-only state once in each preprocessing process."""
    global _WORKER_TOKENIZER, _WORKER_LABELS_MAP
    global _WORKER_JSONL_STAYS, _WORKER_LABEL_KEY
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    _WORKER_LABELS_MAP = labels_map
    _WORKER_JSONL_STAYS = jsonl_stays
    _WORKER_LABEL_KEY = label_key


def preprocess_stay(task):
    """Serialize and batch-tokenize one stay in a worker process."""
    sid, g = task
    _, lines = serialize_lines(g, _WORKER_LABELS_MAP)
    if not lines:
        return sid, None

    encoded = _WORKER_TOKENIZER(lines, add_special_tokens=False,
                                return_length=True, verbose=False)
    lengths = encoded.get("length")
    if lengths is None:
        lengths = [len(ids) for ids in encoded["input_ids"]]
    per_line = np.asarray(lengths, dtype=np.int64) + 1  # newline
    s = _WORKER_JSONL_STAYS.get(sid, {})
    return sid, {
        "lines": lines,
        "per_line": per_line,
        "total": int(per_line.sum()),
        "summary": s.get("patient_summary_text", ""),
        "label": s.get("labels", {}).get(_WORKER_LABEL_KEY),
    }


# ---------------------------------------------------------------------------
# Model scoring
# ---------------------------------------------------------------------------

def load_model(model_name, device):
    import torch
    from transformers import AutoProcessor, AutoTokenizer
    from transformers import Qwen2_5_VLForConditionalGeneration
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Run on a GPU node with a CUDA-enabled PyTorch installation."
        )
    print(f"Loading {model_name} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        device_map={"": device},
        attn_implementation="sdpa",
    )
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
    """P(death) for a batch from Yes/No logits at the answer position."""
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
        # Only materialize the final-position logits. At 32k tokens, returning
        # logits for every position would otherwise consume many extra GB.
        logits = model(**inputs, logits_to_keep=1).logits[:, -1, :]
    pair = torch.stack([logits[:, yes_id], logits[:, no_id]], dim=1)
    p_yes = torch.softmax(pair.float(), dim=1)[:, 0].cpu().tolist()
    lengths = inputs["attention_mask"].sum(dim=1).cpu().tolist()
    return p_yes, lengths


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
    labels_map = pd.read_csv(args.d_items, usecols=["itemid", "label"]) \
                   .set_index("itemid")["label"].to_dict()

    ev = get_events(args, wanted)
    ev["charttime"] = pd.to_datetime(ev["charttime"])
    ev = ev.merge(icu, on="stay_id", how="left")
    ev["hours"] = (ev["charttime"] - ev["intime"]).dt.total_seconds() / 3600.0
    ev = ev[(ev["hours"] >= 0) & (ev["hours"] <= args.prediction_hours)]

    # Serialize each stay once; store per-line token counts for suffix cuts
    n_stays = ev["stay_id"].nunique()
    workers = min(args.workers, max(1, n_stays))
    print(f"Serializing/tokenizing {n_stays} stays with {workers} CPU workers")
    stay_data = {}
    tasks = ev.groupby("stay_id", sort=False)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_preprocessing_worker,
        initargs=(args.model_name, labels_map, jsonl_stays, args.label_key),
    ) as pool:
        for i, (sid, data) in enumerate(
                pool.map(preprocess_stay, tasks, chunksize=4), 1):
            if data is not None:
                stay_data[sid] = data
            if i % 100 == 0 or i == n_stays:
                print(f"  preprocessed {i}/{n_stays} stays", flush=True)
    stay_data = {sid: d for sid, d in stay_data.items()
                 if d["label"] is not None}

    # Coverage report
    with open(out / "coverage.txt", "w") as f:
        for b in args.budgets:
            n_ok = sum(d["total"] >= b for d in stay_data.values())
            f.write(f"{b} event tokens: {n_ok}/{len(stay_data)} stays "
                    f"have full coverage\n")
    print(open(out / "coverage.txt").read())

    if args.require_top_budget:
        top = max(args.budgets)
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

    sids = sorted(stay_data)

    # Resume: skip (stay_id, budget) pairs already present in predictions.csv
    prev = None
    done = set()
    pred_path = out / "predictions.csv"
    if pred_path.exists():
        prev = pd.read_csv(pred_path).dropna(subset=["p_death"])
        done = set(zip(prev["stay_id"].astype(int), prev["budget"].astype(int)))
        print(f"Resuming: {len(done)} inputs already scored", flush=True)

    # Grouping by budget reduces padding waste when --batch-size > 1.
    def iter_requests():
        for b in args.budgets:
            for sid in sids:
                if (int(sid), int(b)) in done:
                    continue
                d = stay_data[sid]
                events_text, used = build_context(d, b)
                user_text = (
                    "PATIENT SUMMARY:\n" + d["summary"] +
                    "\n\nICU EVENTS (chronological, most recent last):\n" +
                    events_text + QUESTION)
                yield sid, b, d["label"], used, user_text

    rows = prev.to_dict("records") if prev is not None else []

    total_requests = len(sids) * len(args.budgets)
    print(f"Scoring {total_requests} inputs on {args.device} "
          f"with batch size {args.batch_size}")
    request_iter = iter_requests()
    completed = len(done)
    while True:
        batch = list(islice(request_iter, args.batch_size))
        if not batch:
            break
        probabilities, input_lengths = score_mortality_batch(
            model, processor, yes_id, no_id, [x[4] for x in batch])
        for (sid, b, label, used, _), p, n_in in zip(
                batch, probabilities, input_lengths):
            rows.append({"stay_id": sid, "budget": b, "p_death": p,
                         "label": label, "event_tokens_used": used,
                         "input_tokens": int(n_in)})
        completed += len(batch)
        # if completed % 50 == 0 or completed == total_requests:
        if completed % 50 < args.batch_size or completed == total_requests:
            print(f"  {completed}/{total_requests} inputs scored", flush=True)
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
                      "delta_ci_hi": round(dhi, 4)})
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
    ax.set_title(f"In-hospital mortality, n={len(y)} stays "
                 f"(identical across budgets)")
    fig.tight_layout()
    fig.savefig(out / "ladder.png", dpi=200)
    print(f"\nDone. Outputs in {out}/")


if __name__ == "__main__":
    main()
