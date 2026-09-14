"""
Experiment 1b: which clinical variables consume the context?

Reads the cached cohort events from Experiment 1 (no re-streaming of the
30 GB chartevents file) and breaks token usage down by itemid label.

For each variable it reports:
  - total tokens across the cohort, and share of the whole context
  - number of recorded events (charting frequency)
  - tokens per event (how verbose a single measurement is)
  - per-stay token counts, so you can show a distribution not just a total

Outputs (in OUT_DIR):
  variable_tokens.csv        one row per variable, cohort-level totals
  variable_per_stay.csv      stay x variable token counts (long format)
  top_variables_bar.png      top-N variables by total tokens
  top_variables_box.png      per-stay distribution for the top-N variables
  cumulative_share.png       how few variables account for most of the context
  category_share.png         tokens grouped by chartevents category

Run: python exp1b_variable_breakdown.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG (match Experiment 1)
# ---------------------------------------------------------------------------
EVENTS_CACHE = "exp1_out/cohort_events.parquet"   # written by exp1/exp2
CHARTEVENTS = "/path/to/mimic-iv-1.0/icu/chartevents.csv"
D_ITEMS     = "/path/to/mimic-iv-1.0/icu/d_items.csv"
ICUSTAYS    = "/path/to/mimic-iv-1.0/icu/icustays.csv"
TOKENIZER    = "Qwen/Qwen2.5-VL-7B-Instruct"
OUT_DIR      = "exp1b_out"
           
PREDICTION_HOURS = None   # None = full stay; set 48.0 to match Exp 2/3 inputs
TOP_N = 25                # variables shown in the figures


def main():
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    cache = Path(EVENTS_CACHE)
    if not cache.exists():
        raise SystemExit(
            f"{cache} not found. Run exp1/exp2 first so the cohort events "
            f"are cached, or point EVENTS_CACHE at the right file.")
    ev = pd.read_parquet(cache)
    print(f"{len(ev):,} events, {ev.stay_id.nunique()} stays")

    d = pd.read_csv(D_ITEMS, usecols=["itemid", "label", "category", "param_type"])
    ev = ev.merge(d, on="itemid", how="left")
    ev["label"] = ev["label"].fillna("unknown_item")
    ev["category"] = ev["category"].fillna("unknown")

    if PREDICTION_HOURS is not None:
        icu = pd.read_csv(ICUSTAYS, usecols=["stay_id", "intime"], parse_dates=["intime"])
        ev["charttime"] = pd.to_datetime(ev["charttime"])
        ev = ev.merge(icu, on="stay_id", how="left")
        hrs = (ev["charttime"] - ev["intime"]).dt.total_seconds() / 3600
        ev = ev[(hrs >= 0) & (hrs <= PREDICTION_HOURS)]
        print(f"Filtered to first {PREDICTION_HOURS}h: {len(ev):,} events")

    # ------------------------------------------------------------------
    # Token cost of one rendered observation: "Heart Rate: 88; "
    # Tokenize each unique (label, value) string once, then map back.
    # ------------------------------------------------------------------
    ev["val"] = ev["valuenum"].where(ev["valuenum"].notna(), ev["value"])
    ev = ev[ev["val"].notna()]
    ev["snippet"] = ev["label"].astype(str) + ": " + ev["val"].astype(str) + "; "

    uniq = ev["snippet"].drop_duplicates()
    print(f"Tokenizing {len(uniq):,} unique observation strings ...")
    enc = tok(list(uniq), add_special_tokens=False).input_ids
    cost = dict(zip(uniq, (len(e) for e in enc)))
    ev["tokens"] = ev["snippet"].map(cost)

    total_tokens = ev["tokens"].sum()
    n_stays = ev["stay_id"].nunique()

    # ------------------------------------------------------------------
    # Cohort-level table
    # ------------------------------------------------------------------
    g = ev.groupby("label").agg(
        tokens=("tokens", "sum"),
        events=("tokens", "size"),
        stays=("stay_id", "nunique"),
        category=("category", "first"),
    ).reset_index()
    g["tokens_per_event"] = (g["tokens"] / g["events"]).round(2)
    g["pct_of_context"] = (100 * g["tokens"] / total_tokens).round(2)
    g["tokens_per_stay"] = (g["tokens"] / n_stays).round(1)
    g["pct_stays_present"] = (100 * g["stays"] / n_stays).round(1)
    g = g.sort_values("tokens", ascending=False)
    g.to_csv(out / "variable_tokens.csv", index=False)

    print("\nTop 15 variables by token usage:")
    print(g.head(15)[["label", "tokens", "pct_of_context", "events",
                      "tokens_per_event", "pct_stays_present"]]
          .to_string(index=False))

    # concentration
    cum = g["tokens"].cumsum() / total_tokens
    n50 = int((cum < 0.5).sum() + 1)
    n80 = int((cum < 0.8).sum() + 1)
    n90 = int((cum < 0.9).sum() + 1)
    summary = (f"{len(g)} distinct variables, {total_tokens:,} tokens total\n"
               f"{n50} variables account for 50% of all context tokens\n"
               f"{n80} variables account for 80%\n"
               f"{n90} variables account for 90%\n")
    (out / "concentration.txt").write_text(summary)
    print("\n" + summary)

    # per-stay long table
    per_stay = ev.groupby(["stay_id", "label"])["tokens"].sum().reset_index()
    per_stay.to_csv(out / "variable_per_stay.csv", index=False)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = g.head(TOP_N).iloc[::-1]

    fig, ax = plt.subplots(figsize=(7, 0.32 * TOP_N + 1.2))
    ax.barh(top["label"], top["pct_of_context"], color="#5b8ff9")
    ax.set_xlabel("% of all context tokens")
    ax.set_title(f"Top {TOP_N} variables by token usage "
                 f"({n_stays} stays, {total_tokens/1e6:.1f}M tokens)")
    for y, (v, t) in enumerate(zip(top["pct_of_context"], top["tokens"])):
        ax.text(v, y, f" {v:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "top_variables_bar.png", dpi=200)

    # per-stay distribution (shows spread, not just cohort totals)
    labels = list(g.head(TOP_N)["label"])[::-1]
    data = [per_stay.loc[per_stay["label"] == l, "tokens"].to_numpy()
            for l in labels]
    fig, ax = plt.subplots(figsize=(7, 0.32 * TOP_N + 1.2))
    ax.boxplot(data, vert=False, labels=labels, showfliers=False,
               widths=0.6, patch_artist=True,
               boxprops=dict(facecolor="#9fc5ff", edgecolor="#33507a"),
               medianprops=dict(color="#1a2b45"))
    ax.set_xscale("log")
    ax.set_xlabel("Tokens per stay (log scale)")
    ax.set_title(f"Per-stay token usage by variable (top {TOP_N})")
    fig.tight_layout()
    fig.savefig(out / "top_variables_box.png", dpi=200)

    # cumulative share
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(g) + 1), 100 * cum.to_numpy())
    for n, lab in [(n50, "50%"), (n80, "80%"), (n90, "90%")]:
        ax.axvline(n, color="gray", ls="--", lw=0.8)
        ax.text(n, 5, f" {n} vars = {lab}", rotation=90, fontsize=8,
                va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("Number of variables (ranked by token usage)")
    ax.set_ylabel("Cumulative % of context tokens")
    ax.set_title("Context is dominated by a small number of variables")
    fig.tight_layout()
    fig.savefig(out / "cumulative_share.png", dpi=200)

    # category composition
    cat = ev.groupby("category")["tokens"].sum().sort_values(ascending=False)
    cat_pct = 100 * cat / total_tokens
    keep = cat_pct.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(keep.index, keep.to_numpy(), color="#7bc47f")
    ax.set_xlabel("% of all context tokens")
    ax.set_title("Token usage by chartevents category")
    fig.tight_layout()
    fig.savefig(out / "category_share.png", dpi=200)

    print(f"Done. Outputs in {out}/")


if __name__ == "__main__":
    main()