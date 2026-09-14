# Long Context Is Not Enough

Code accompanying the anonymous manuscript **“Long Context Is Not Enough: An Empirical Case for Dynamic Context Management and Reasoning in ICU Clinical Decision-Support Agents.”**

This repository studies how the volume and composition of ICU records affect long-context language-model reasoning. Using MIMIC-IV `chartevents`, it:

* Measures how serialized clinical context grows with ICU length of stay.
* Characterizes how tokens are distributed across clinical variables.
* Evaluates zero-shot in-hospital mortality prediction across context budgets.
* Uses a padding control to separate the effects of input length from added clinical content.

The experiments compare a wide context containing all available `chartevents` variables with a clinician-curated core set of 17 variables.

## Repository structure

| File                         | Purpose                                                                         |
| ---------------------------- | ------------------------------------------------------------------------------- |
| `exp1_chartevents_tokens.py` | Measure per-stay token counts, budget crossings, and growth with length of stay |
| `token_distribution.py`      | Analyze token concentration across clinical variables and categories            |
| `exp2_context_truncation.py` | Evaluate mortality prediction over a 2K–32K truncation ladder                   |
| `exp3_padding_control.py`    | Hold clinical content fixed while varying input length and position             |
| `exp*_parallelized_*.py`     | Parallelized versions of the main experiments                                   |
| `New Folder With Items/`     | Core 17-variable versions of Experiments 1–3                                    |
| `split_icustays.py`          | Create ICU-stay CSV files from JSONL train/validation/test splits               |
| `patients_demographics.py`   | Summarize cohort demographics from `patients.csv`                               |
| `chartevents_analysis.ipynb` | Small exploratory notebook for `chartevents`                                    |

Some non-CLI scripts expose configuration constants near the top of the file. The `exp1_*.py`, `exp2_*.py`, and `exp3_*.py` entry points provide command-line arguments and are recommended for reproducible runs.

## Data requirements

The code expects credentialed access to [MIMIC-IV](https://physionet.org/content/mimiciv/). Data are not included in this repository.

Required ICU tables:

* `icu/chartevents.csv` or `chartevents.csv.gz`
* `icu/d_items.csv` or `d_items.csv.gz`
* `icu/icustays.csv` or `icustays.csv.gz`

`patients_demographics.py` additionally uses `hosp/patients.csv`.

An optional extraction JSONL can define the cohort and supply static text and mortality labels. Each line should contain `stay_id` and may include `subject_id`, `hadm_id`, `patient_summary_text`, `radiology_report_text`, and:

```json
{
  "labels": {
    "in_hospital_mortality_48hr": 0
  }
}
```

Alternatively, use a text file containing one `stay_id` per line with `--stays`.

## Installation

Python 3.10 or newer is recommended. Install the main dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn matplotlib pyarrow torch transformers
```

A CUDA-capable GPU is strongly recommended for Experiments 2 and 3. Model weights and tokenizer files for `Qwen/Qwen2.5-VL-7B-Instruct` are downloaded through Hugging Face unless a local model path is supplied.

## Usage

Replace the example paths below with the locations of your credentialed MIMIC-IV files and cohort split.

### 1. Measure context growth

```bash
python exp1_chartevents_tokens.py \
  --chartevents /path/to/mimic-iv/icu/chartevents.csv.gz \
  --d-items /path/to/mimic-iv/icu/d_items.csv.gz \
  --icustays /path/to/mimic-iv/icu/icustays.csv.gz \
  --jsonl /path/to/splits/train.jsonl \
  --out exp1_out
```

Key outputs include `token_counts.csv`, `budget_table.csv`, `growth.txt`, `hist_tokens.png`, and `tokens_vs_los.png`. The generated `cohort_events.parquet` cache can be reused by later experiments.

### 2. Run the context-budget sweep

```bash
python exp2_context_truncation.py \
  --chartevents /path/to/mimic-iv/icu/chartevents.csv.gz \
  --d-items /path/to/mimic-iv/icu/d_items.csv.gz \
  --icustays /path/to/mimic-iv/icu/icustays.csv.gz \
  --jsonl /path/to/splits/train.jsonl \
  --events-cache exp1_out/cohort_events.parquet \
  --budgets 2000 4000 8000 16000 32000 \
  --out exp2_out
```

This produces per-stay predictions, AUROC estimates with paired-bootstrap confidence intervals, coverage statistics, and the context-ladder figure.

### 3. Run the padding control

```bash
python exp3_padding_control.py \
  --chartevents /path/to/mimic-iv/icu/chartevents.csv.gz \
  --d-items /path/to/mimic-iv/icu/d_items.csv.gz \
  --icustays /path/to/mimic-iv/icu/icustays.csv.gz \
  --jsonl /path/to/splits/train.jsonl \
  --events-cache exp1_out/cohort_events.parquet \
  --base-budget 2000 \
  --pad-budgets 4000 8000 16000 \
  --out exp3_out
```

The control freezes decision-relevant clinical content at 2K tokens and adds labeled, non-informative events before or after the patient record. Outputs include `predictions.csv`, `auroc_by_condition.csv`, `positional_effect.txt`, and `padding.png`.

Run any CLI script with `--help` to display its complete arguments:

```bash
python exp2_context_truncation.py --help
```

## Reproducibility notes

* Use the same JSONL split, `--seed`, `--n-samples`, and events cache across experiments.
* By default, the prediction task uses the label key `in_hospital_mortality_48hr`.
* Experiment 2 uses the most recent event tokens at each budget, so each larger context contains the smaller one.
* `chartevents` excludes sources such as clinical notes, prescriptions, and portions of laboratory data. Its token counts should therefore be interpreted as a lower bound on the full longitudinal record.
* MIMIC-IV remains subject to its PhysioNet credentialing and data-use requirements. Do not commit source data, derived patient-level records, credentials, model tokens, or generated caches containing protected data.

## Main finding

ICU context grows strongly with length of stay and is dispersed across many variables. In the manuscript experiments, increasing the raw context budget does not monotonically improve zero-shot mortality prediction; performance peaks at an intermediate budget and can decline at longer contexts.

These results motivate active selection, retrieval, summarization, and context-folding mechanisms for longitudinal ICU agents.
