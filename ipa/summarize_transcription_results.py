"""Summarize transcription results from ipa/output/ipa_transcriptions.csv.

The summary is grouped by ipa_model and uses the target token extracted from
the segment filename. For a filename like:

    002_tòlúŋ_caterpillar_350_1.wav

the target is `tòlúŋ`, i.e. the text between the first two underscores.

Output columns:
- ipa_model
- rows_total
- cer_mean
- wer_mean
- norm_lev_similarity_mean
- length_ratio_mean
- #match
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein distance with iterative dynamic programming."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, delete, replace))
        prev = curr
    return prev[-1]


def normalize_text(text: object) -> str:
    if text is None:
        return ""
    if isinstance(text, float) and np.isnan(text):
        return ""
    return str(text).strip()


def extract_target_from_segment_filename(segment_filename: str) -> str:
    """Extract the text between the first two underscores in the filename."""
    filename = normalize_text(segment_filename)
    if not filename:
        return ""

    if filename.lower().endswith(".wav"):
        filename = filename[:-4]

    parts = filename.split("_", 2)
    if len(parts) < 2:
        return ""
    return parts[1].strip()


def word_distance(ref_words: list[str], hyp_words: list[str]) -> int:
    """Compute token-level Levenshtein distance."""
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
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n]


def summarize_file(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"ERROR: failed to read {path}: {exc}")
        return pd.DataFrame()

    required = ["segment_filename", "ipa_model", "ipa_transcription"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"ERROR: {path.name} is missing required columns: {missing}")
        return pd.DataFrame()

    work = df[["segment_filename", "ipa_model", "ipa_transcription"]].copy()
    work["segment_filename"] = work["segment_filename"].map(normalize_text)
    work["ipa_model"] = work["ipa_model"].map(normalize_text)
    work["ipa_transcription"] = work["ipa_transcription"].map(normalize_text)
    work["target_text"] = work["segment_filename"].map(extract_target_from_segment_filename)

    rows = []
    for ipa_model, model_df in work.groupby("ipa_model", dropna=False):
        cer_values = []
        wer_values = []
        similarity_values = []
        length_ratios = []
        match_count = 0

        for target_text, hyp_text in zip(model_df["target_text"], model_df["ipa_transcription"]):
            ref = normalize_text(target_text)
            hyp = normalize_text(hyp_text)

            ref_len = max(len(ref), 1)
            hyp_len = len(hyp)

            lev = levenshtein(ref, hyp)
            cer_values.append(lev / ref_len)

            ref_words = ref.split()
            hyp_words = hyp.split()
            wer_values.append(word_distance(ref_words, hyp_words) / max(len(ref_words), 1))

            denom = max(len(ref), len(hyp), 1)
            similarity_values.append(1.0 - (lev / denom))

            length_ratios.append(hyp_len / ref_len)

            if ref == hyp:
                match_count += 1

        rows.append(
            {
                "ipa_model": ipa_model,
                "rows_total": int(len(model_df)),
                "cer_mean": float(np.mean(cer_values)) if cer_values else 0.0,
                "wer_mean": float(np.mean(wer_values)) if wer_values else 0.0,
                "norm_lev_similarity_mean": float(np.mean(similarity_values)) if similarity_values else 0.0,
                "length_ratio_mean": float(np.mean(length_ratios)) if length_ratios else 0.0,
                "#match": int(match_count),
            }
        )

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df

    summary_df = summary_df[
        [
            "ipa_model",
            "rows_total",
            "cer_mean",
            "wer_mean",
            "norm_lev_similarity_mean",
            "length_ratio_mean",
            "#match",
        ]
    ]

    for column in ["cer_mean", "wer_mean", "norm_lev_similarity_mean", "length_ratio_mean"]:
        summary_df[column] = summary_df[column].astype(float).round(6)

    return summary_df.sort_values("ipa_model").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        "-i",
        default="ipa/output/ipa_transcriptions.csv",
        help="Transcription CSV to summarize",
    )
    parser.add_argument(
        "--out-file",
        "-o",
        default="ipa/output/transcription_summary.csv",
        help="Summary CSV output path",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    summary_df = summarize_file(input_path)
    if summary_df.empty:
        print("No summary rows were generated")
        sys.exit(1)

    output_path = Path(args.out_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False, encoding="utf-8")

    print(summary_df.to_string(index=False))
    print(f"\nWrote summary to {output_path}")


if __name__ == "__main__":
    main()