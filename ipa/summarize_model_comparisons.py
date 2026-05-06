"""Generate model comparison summary from compare_to_target_rows.csv with per-model metrics.

Usage:
  python ipa/summarize_model_comparisons.py --input ipa/output/compare_to_target_rows.csv

Computes per-model aggregates:
- CER (character error rate)
- WER (word error rate)
- Normalized Levenshtein similarity
- Exact match rate
- Reference length mean
- Hypothesis length mean
- Length ratio (hyp / ref)
"""
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        "-i",
        default="ipa/output/compare_to_target_rows.csv",
        help="Input CSV with model comparisons"
    )
    p.add_argument(
        "--output",
        "-o",
        default="ipa/output/model_comparison_summary.csv",
        help="Output summary CSV file"
    )
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading {input_path}: {e}")
        sys.exit(1)

    print(f"Loaded {len(df)} rows from {input_path.name}")
    print(f"Columns: {df.columns.tolist()}")

    # Extract model names
    if 'ipa_model' not in df.columns:
        print("ERROR: 'ipa_model' column not found")
        sys.exit(1)

    models = df['ipa_model'].dropna().unique()
    print(f"Found {len(models)} models: {models}")

    results = []

    for model in sorted(models):
        model_df = df[df['ipa_model'] == model].copy()
        n = len(model_df)

        # Identify reference and hypothesis columns
        ref_col = None
        hyp_col = None
        
        if 'target_ref_text' in model_df.columns:
            ref_col = 'target_ref_text'
        elif 'target_transcription' in model_df.columns:
            ref_col = 'target_transcription'
        
        if 'ipa_transcription' in model_df.columns:
            hyp_col = 'ipa_transcription'

        if not ref_col or not hyp_col:
            print(f"WARNING: Could not find reference/hypothesis columns for {model}. Skipping.")
            continue

        refs = model_df[ref_col].fillna("").astype(str)
        hyps = model_df[hyp_col].fillna("").astype(str)

        # Compute metrics
        cer_values = []
        wer_values = []
        norm_lev_values = []
        exact_matches = []
        ref_lengths = []
        hyp_lengths = []

        for idx, (ref, hyp) in enumerate(zip(refs, hyps)):
            ref_clean = ref.strip()
            hyp_clean = hyp.strip()

            # Use pre-computed metrics if available
            if 'cer' in model_df.columns:
                cer_val = model_df.iloc[idx]['cer']
                if pd.notna(cer_val):
                    cer_values.append(float(cer_val))
                else:
                    lev_dist = levenshtein(ref_clean, hyp_clean)
                    cer = lev_dist / max(len(ref_clean), 1)
                    cer_values.append(cer)
            else:
                lev_dist = levenshtein(ref_clean, hyp_clean)
                cer = lev_dist / max(len(ref_clean), 1)
                cer_values.append(cer)

            if 'wer' in model_df.columns:
                wer_val = model_df.iloc[idx]['wer']
                if pd.notna(wer_val):
                    wer_values.append(float(wer_val))
                else:
                    ref_words = ref_clean.split()
                    hyp_words = hyp_clean.split()
                    wlev = word_levenshtein(ref_words, hyp_words)
                    wer = wlev / max(len(ref_words), 1)
                    wer_values.append(wer)
            else:
                ref_words = ref_clean.split()
                hyp_words = hyp_clean.split()
                wlev = word_levenshtein(ref_words, hyp_words)
                wer = wlev / max(len(ref_words), 1)
                wer_values.append(wer)

            if 'norm_lev_similarity' in model_df.columns:
                nlev_val = model_df.iloc[idx]['norm_lev_similarity']
                if pd.notna(nlev_val):
                    norm_lev_values.append(float(nlev_val))
                else:
                    lev_dist = levenshtein(ref_clean, hyp_clean)
                    nlev = lev_dist / max(len(ref_clean), 1)
                    norm_lev_values.append(nlev)
            else:
                lev_dist = levenshtein(ref_clean, hyp_clean)
                nlev = lev_dist / max(len(ref_clean), 1)
                norm_lev_values.append(nlev)

            # Exact match
            exact_matches.append(1 if ref_clean == hyp_clean else 0)

            # Lengths
            ref_lengths.append(len(ref_clean))
            hyp_lengths.append(len(hyp_clean))

        # Aggregate
        cer_mean = float(np.mean(cer_values)) if cer_values else 0.0
        wer_mean = float(np.mean(wer_values)) if wer_values else 0.0
        norm_lev_mean = float(np.mean(norm_lev_values)) if norm_lev_values else 0.0
        exact_match_rate = float(np.mean(exact_matches)) if exact_matches else 0.0
        ref_len_mean = float(np.mean(ref_lengths)) if ref_lengths else 0.0
        hyp_len_mean = float(np.mean(hyp_lengths)) if hyp_lengths else 0.0
        length_ratio = (hyp_len_mean / ref_len_mean) if ref_len_mean > 0 else 0.0

        results.append({
            'ipa_model': model,
            'rows_total': n,
            'cer_mean': cer_mean,
            'wer_mean': wer_mean,
            'norm_lev_mean': norm_lev_mean,
            'exact_match_rate': exact_match_rate,
            'ref_len_mean': ref_len_mean,
            'hyp_len_mean': hyp_len_mean,
            'length_ratio_mean': length_ratio,
        })

        print(f"  {model}: {n} rows, CER={cer_mean:.4f}, WER={wer_mean:.4f}, "
              f"ref_len={ref_len_mean:.1f}, hyp_len={hyp_len_mean:.1f}, ratio={length_ratio:.4f}")

    if not results:
        print("No results generated")
        sys.exit(1)

    result_df = pd.DataFrame(results)
    col_order = [
        'ipa_model', 'rows_total', 'cer_mean', 'wer_mean', 'norm_lev_mean',
        'exact_match_rate', 'ref_len_mean', 'hyp_len_mean', 'length_ratio_mean'
    ]
    result_df = result_df[col_order]

    # Round
    float_cols = [
        'cer_mean', 'wer_mean', 'norm_lev_mean', 'exact_match_rate',
        'ref_len_mean', 'hyp_len_mean', 'length_ratio_mean'
    ]
    for col in float_cols:
        result_df[col] = result_df[col].round(6)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(result_df.to_string(index=False))
    print(f"\nWrote summary to {output_path}")


def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein distance."""
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


def word_levenshtein(ref_words, hyp_words) -> int:
    """Compute word-level Levenshtein distance."""
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


if __name__ == "__main__":
    main()
