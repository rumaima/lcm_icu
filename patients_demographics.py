#!/usr/bin/env python3
"""
Compute cohort demographics from the MIMIC-IV patients.csv table.

Cohort: ALL patients in patients.csv (no age or admission filter).
Note this includes patients who never had an inpatient admission
(e.g. ED-only or outpatient encounters).

Metrics:
    - Patients per year
    - Mean age
    - Median age
    - Number of female patients
    - Number of male patients

Age is taken from `anchor_age`, the patient's age in their `anchor_year`
(their first year of data). Ages above 89 are de-identified to 91.

Usage:
    python patients_demographics.py patients.csv
    python patients_demographics.py patients.csv --csv-out demographics.csv
"""

import argparse
import sys

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["subject_id", "gender", "anchor_age", "anchor_year_group"]
CENSOR_AGE = 91  # MIMIC de-identifies ages > 89 to 91


def load_patients(path):
    df = pd.read_csv(path, usecols=lambda c: c in REQUIRED_COLUMNS)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {path} is missing required column(s): {', '.join(missing)}")

    n_raw = len(df)
    df = df.drop_duplicates(subset="subject_id")
    if len(df) < n_raw:
        print(f"NOTE: dropped {n_raw - len(df):,} duplicate subject_id rows", file=sys.stderr)

    return df


def year_span(df):
    """Derive the number of calendar years covered from anchor_year_group.

    anchor_year_group holds shifted three-year buckets such as '2008 - 2010'.
    The span runs from the first year of the earliest bucket to the last year
    of the latest bucket, inclusive.
    """
    groups = df["anchor_year_group"].dropna().unique()
    if len(groups) == 0:
        return None, None, None

    years = []
    for g in groups:
        years.extend(int(tok) for tok in str(g).replace("-", " ").split() if tok.isdigit())

    if not years:
        return None, None, None

    start, end = min(years), max(years)
    return start, end, end - start + 1


def summarise(df):
    ages = df["anchor_age"].dropna()
    gender = df["gender"].str.strip().str.upper()

    start, end, n_years = year_span(df)
    n_patients = len(df)

    return {
        "n_patients": n_patients,
        "period_start": start,
        "period_end": end,
        "n_years": n_years,
        "patients_per_year": n_patients / n_years if n_years else np.nan,
        "mean_age": ages.mean(),
        "median_age": ages.median(),
        "age_q25": ages.quantile(0.25),
        "age_q75": ages.quantile(0.75),
        "n_female": int((gender == "F").sum()),
        "n_male": int((gender == "M").sum()),
        "n_gender_other_or_missing": int((~gender.isin(["F", "M"])).sum()),
        "n_age_censored": int((df["anchor_age"] >= CENSOR_AGE).sum()),
    }


def report(s):
    n = s["n_patients"]
    pct = lambda x: f"{100 * x / n:.1f}%" if n else "n/a"

    lines = [
        "MIMIC-IV patient demographics (all patients in patients.csv)",
        "=" * 60,
        f"Total patients          {n:>12,}",
        f"Data period             {s['period_start']} - {s['period_end']} "
        f"({s['n_years']} years)",
        f"Patients per year       {s['patients_per_year']:>12,.0f}",
        "",
        f"Mean age                {s['mean_age']:>12.1f} years",
        f"Median age              {s['median_age']:>12.1f} years  "
        f"(IQR {s['age_q25']:.0f}-{s['age_q75']:.0f})",
        "",
        f"Female                  {s['n_female']:>12,}  ({pct(s['n_female'])})",
        f"Male                    {s['n_male']:>12,}  ({pct(s['n_male'])})",
    ]

    if s["n_gender_other_or_missing"]:
        lines.append(
            f"Other / missing gender  {s['n_gender_other_or_missing']:>12,}  "
            f"({pct(s['n_gender_other_or_missing'])})"
        )

    lines += [
        "",
        "Caveats",
        "-" * 60,
        f"- {s['n_age_censored']:,} patients ({pct(s['n_age_censored'])}) have anchor_age "
        f">= {CENSOR_AGE}, the",
        "  de-identified value for ages over 89. This pulls the mean down.",
        "- anchor_age is age at anchor_year (first year of data), not age at",
        "  any particular admission. Admission-time ages are higher.",
        "- Patients per year is total patients divided by the span in years, not",
        "  a true annual census. anchor_year_group is keyed to first contact.",
        "- This cohort includes patients with no inpatient admission.",
    ]

    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("patients_csv", help="path to MIMIC-IV patients.csv")
    ap.add_argument("--csv-out", help="also write the metrics to this CSV file")
    args = ap.parse_args()

    df = load_patients(args.patients_csv)
    stats = summarise(df)
    report(stats)

    if args.csv_out:
        pd.DataFrame([stats]).to_csv(args.csv_out, index=False)
        print(f"\nWrote metrics to {args.csv_out}")


if __name__ == "__main__":
    main()
