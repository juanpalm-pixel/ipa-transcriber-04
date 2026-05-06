"""Produce a model comparison summary similar to the attached table.

Usage:
  python ipa/summarize_model_results.py --input-dir ipa/output --out-file ipa/output/model_summary.csv

The script looks for CSV files in the input directory. Each CSV should contain
columns with the reference and hypothesis text. Common names attempted are:
- 'target', 'reference', 'ref'
- 'prediction', 'hypothesis', 'hyp'

For each file the script computes per-row:
- character error rate (CER)
- word error rate (WER)
- normalized Levenshtein (lev / max_len)
- exact match flag
- length ratio (len(hyp)/len(ref))

Then it aggregates means and counts into a summary table and writes a CSV.
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import sys


def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein distance (iterative DP)."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    b_len = len(b)
    prev = list(range(b_len + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * b_len
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (0 if ca == cb else 1)
            curr[j] = min(ins, delete, replace)
        prev = curr
    return prev[b_len]


def word_distance(ref_words, hyp_words):
    # token-level DP
    m = len(ref_words)
    n = len(hyp_words)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def infer_columns(df: pd.DataFrame):
    # Try common names
    ref_cols = [c for c in df.columns if c.lower() in ("target", "reference", "ref", "ground_truth")]
    hyp_cols = [c for c in df.columns if c.lower() in ("prediction", "hypothesis", "hyp", "pred")]
    if ref_cols and hyp_cols:
        return ref_cols[0], hyp_cols[0]
    # fallback: try first two string/object columns
    string_cols = [c for c in df.columns if df[c].dtype == object]
    if len(string_cols) >= 2:
        return string_cols[0], string_cols[1]
    # try any two columns
    if len(df.columns) >= 2:
        return df.columns[0], df.columns[1]
    raise ValueError("Could not infer reference/hypothesis columns from CSV. Please provide columns with text.")


def summarize_file(path: Path):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Warning: failed to read {path}: {e}")
        return None

    if df.empty:
        return None

    try:
        ref_col, hyp_col = infer_columns(df)
    except Exception as e:
        print(f"Skipping {path.name}: {e}")
        return None

    refs = df[ref_col].fillna("").astype(str)
    hyps = df[hyp_col].fillna("").astype(str)

    n = len(df)
    cer_list = []
    wer_list = []
    norm_lev_list = []
    exact_flags = []
    length_ratios = []

    for r, h in zip(refs, hyps):
        lev = levenshtein(r, h)
        ref_len = max(len(r), 1)
        cer = (lev / ref_len)
        cer_list.append(cer)

        norm_lev = lev / ref_len
        norm_lev_list.append(norm_lev)

        r_words = r.split()
        h_words = h.split()
        wlev = word_distance(r_words, h_words)
        ref_words_len = max(len(r_words), 1)
        wer = (wlev / ref_words_len)
        wer_list.append(wer)

        exact_flags.append(1 if r.strip() == h.strip() else 0)

        length_ratios.append((len(h) / max(len(r), 1)))

    res = {
        "ipa_model": path.stem,
        "rows_total": n,
        "cer_mean": float(np.mean(cer_list)) if cer_list else 0.0,
        "wer_mean": float(np.mean(wer_list)) if wer_list else 0.0,
        "norm_lev_mean": float(np.mean(norm_lev_list)) if norm_lev_list else 0.0,
        # exact_match_rate as decimal fraction (0..1), not percentage
        "exact_match_rate": float(np.mean(exact_flags)) if exact_flags else 0.0,
        "length_ratio_mean": float(np.mean(length_ratios)) if length_ratios else 0.0,
        "rows_all": n,
        "rows_mat": int(sum(exact_flags)),
        "rows_unm": int(n - sum(exact_flags)),
    }

    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", "-i", default="ipa/output", help="Directory with model CSV files")
    p.add_argument("--out-file", "-o", default="ipa/output/model_summary.csv", help="Summary CSV output path")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)

    csv_files = sorted(list(input_dir.glob("*.csv")))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        sys.exit(1)

    rows = []
    for f in csv_files:
        print(f"Processing {f.name}")
        s = summarize_file(f)
        if s:
            rows.append(s)

    if not rows:
        print("No summaries generated")
        sys.exit(0)

    df = pd.DataFrame(rows)
    col_order = [
        "ipa_model", "rows_total", "cer_mean", "wer_mean", "norm_lev_mean",
        "exact_match_rate", "length_ratio_mean", "rows_all", "rows_mat", "rows_unm"
    ]
    # ensure all columns exist
    for c in col_order:
        if c not in df.columns:
            df[c] = 0
    df = df[col_order]

    # Round numeric columns to 6 decimal places (decimal fractions, not percentages)
    float_cols = ["cer_mean", "wer_mean", "norm_lev_mean", "exact_match_rate", "length_ratio_mean"]
    for c in float_cols:
        if c in df.columns:
            df[c] = df[c].astype(float).round(6)

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.6f")

    print("\nSummary:\n")
    print(df.to_string(index=False))
    print(f"\nWrote summary to {out_path}")


if __name__ == "__main__":
    main()
