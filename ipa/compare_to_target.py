"""
Compare model outputs (01/02 CSV files) against target labels from input TXT.

Outputs:
- ipa/output/compare_to_target_rows.csv
- ipa/output/compare_to_target_summary.csv
"""

from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path
from collections import defaultdict

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

TARGET_TXT = PROJECT_ROOT / "input" / "MGM_AFA4_2nd.txt"
MODEL_CSV_FILES = [
    BASE_DIR / "output" / "ipa_transcriptions.csv",]

ROWS_OUT = BASE_DIR / "output" / "compare_to_target_rows.csv"
SUMMARY_OUT = BASE_DIR / "output" / "compare_to_target_summary.csv"


def normalize_for_match(text: str) -> str:
    """Normalize labels so target/transcribed filenames can align reliably."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.strip()
    # Mirror filename sanitization behavior: slash becomes underscore.
    text = text.replace("/", "_")
    # Collapse repeated spaces for robust matching.
    text = re.sub(r"\s+", " ", text)
    return text


def safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def levenshtein_distance(a: str, b: str) -> int:
    """Simple dynamic-programming Levenshtein distance."""
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
            sub = prev[j - 1] + (ca != cb)
            curr.append(min(ins, delete, sub))
        prev = curr
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    ref_chars = list(ref)
    hyp_chars = list(hyp)
    denom = max(1, len(ref_chars))
    return levenshtein_distance("".join(ref_chars), "".join(hyp_chars)) / denom


def wer(ref: str, hyp: str) -> float:
    ref_words = ref.split()
    hyp_words = hyp.split()
    denom = max(1, len(ref_words))
    return levenshtein_distance("\u0001".join(ref_words), "\u0001".join(hyp_words)) / denom


def normalized_similarity(ref: str, hyp: str) -> float:
    denom = max(len(ref), len(hyp), 1)
    dist = levenshtein_distance(ref, hyp)
    return 1.0 - (dist / denom)


def extract_reference_token(transcription: str) -> str:
    """Use first token before first underscore as requested."""
    raw = safe_text(transcription)
    return raw.split("_", 1)[0].strip()


def parse_target_file(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    # Skip header line; rows are tab-separated.
    for i, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue

        start = safe_text(parts[0])
        end = safe_text(parts[1])
        transcription = safe_text(parts[2])

        rows.append(
            {
                "target_line": i,
                "target_start": start,
                "target_end": end,
                "target_transcription": transcription,
                "target_ref_text": extract_reference_token(transcription),
                "target_match_key": normalize_for_match(transcription),
            }
        )

    return pd.DataFrame(rows)


def parse_model_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]

    for col in ["segment_filename", "ipa_model", "ipa_transcription", "audio_path"]:
        if col in df.columns:
            df[col] = df[col].astype(str).map(safe_text)

    def filename_to_match_key(filename: str) -> str:
        # Example: 001_Spider_349.wav -> Spider_349
        base = filename
        if base.lower().endswith(".wav"):
            base = base[:-4]
        base = re.sub(r"^\d+_", "", base)
        return normalize_for_match(base)

    df["model_source_file"] = path.name
    df["pred_match_key"] = df["segment_filename"].map(filename_to_match_key)
    return df


def build_target_lookup(target_df: pd.DataFrame) -> dict[str, list[dict]]:
    lookup: dict[str, list[dict]] = defaultdict(list)
    for rec in target_df.to_dict(orient="records"):
        lookup[rec["target_match_key"]].append(rec)
    return lookup


def compare() -> tuple[pd.DataFrame, pd.DataFrame]:
    target_df = parse_target_file(TARGET_TXT)
    target_lookup = build_target_lookup(target_df)

    all_rows = []

    for model_csv in MODEL_CSV_FILES:
        if not model_csv.exists():
            print(f"WARNING: Missing model output file: {model_csv}")
            continue

        model_df = parse_model_csv(model_csv)

        for rec in model_df.to_dict(orient="records"):
            pred_key = rec.get("pred_match_key", "")
            pred_text = safe_text(rec.get("ipa_transcription", ""))

            matched_target = None
            candidates = target_lookup.get(pred_key, [])
            if candidates:
                # Keep deterministic behavior if duplicates exist.
                matched_target = candidates[0]

            out = {
                "model_source_file": rec.get("model_source_file", ""),
                "ipa_model": safe_text(rec.get("ipa_model", "")),
                "segment_filename": safe_text(rec.get("segment_filename", "")),
                "audio_path": safe_text(rec.get("audio_path", "")),
                "pred_match_key": pred_key,
                "ipa_transcription": pred_text,
            }

            if matched_target is None:
                out.update(
                    {
                        "match_status": "unmatched",
                        "target_line": None,
                        "target_start": None,
                        "target_end": None,
                        "target_transcription": "",
                        "target_ref_text": "",
                        "target_match_key": "",
                        "cer": None,
                        "wer": None,
                        "norm_lev_similarity": None,
                        "exact_match": None,
                        "length_ratio": None,
                    }
                )
            else:
                ref = safe_text(matched_target["target_ref_text"])
                hyp = pred_text
                out.update(
                    {
                        "match_status": "matched",
                        "target_line": matched_target["target_line"],
                        "target_start": matched_target["target_start"],
                        "target_end": matched_target["target_end"],
                        "target_transcription": matched_target["target_transcription"],
                        "target_ref_text": ref,
                        "target_match_key": matched_target["target_match_key"],
                        "cer": cer(ref, hyp),
                        "wer": wer(ref, hyp),
                        "norm_lev_similarity": normalized_similarity(ref, hyp),
                        "exact_match": int(ref == hyp),
                        "length_ratio": (len(hyp) / len(ref)) if len(ref) > 0 else None,
                    }
                )

            all_rows.append(out)

    rows_df = pd.DataFrame(all_rows)

    if rows_df.empty:
        return rows_df, pd.DataFrame()

    # Per-model summary from matched rows only.
    matched_df = rows_df[rows_df["match_status"] == "matched"].copy()
    if matched_df.empty:
        summary_df = pd.DataFrame()
    else:
        summary_df = (
            matched_df.groupby("ipa_model", dropna=False)
            .agg(
                rows_total=("ipa_model", "size"),
                cer_mean=("cer", "mean"),
                wer_mean=("wer", "mean"),
                norm_lev_similarity_mean=("norm_lev_similarity", "mean"),
                exact_match_rate=("exact_match", "mean"),
                length_ratio_mean=("length_ratio", "mean"),
            )
            .reset_index()
        )

    # Add coverage details (matched/unmatched) from all rows.
    coverage = (
        rows_df.groupby("ipa_model", dropna=False)
        .agg(
            rows_all=("ipa_model", "size"),
            rows_matched=("match_status", lambda s: int((s == "matched").sum())),
            rows_unmatched=("match_status", lambda s: int((s == "unmatched").sum())),
        )
        .reset_index()
    )

    if not summary_df.empty:
        summary_df = summary_df.merge(coverage, on="ipa_model", how="left")
    else:
        summary_df = coverage

    return rows_df, summary_df


def main() -> None:
    print("=" * 80)
    print("COMPARE MODEL OUTPUTS TO TARGET TXT")
    print("=" * 80)

    print(f"Target file: {TARGET_TXT}")
    for p in MODEL_CSV_FILES:
        print(f"Model output: {p}")

    rows_df, summary_df = compare()

    if rows_df.empty:
        print("No rows to compare. Check input files.")
        return

    ROWS_OUT.parent.mkdir(parents=True, exist_ok=True)
    rows_df.to_csv(ROWS_OUT, index=False, encoding="utf-8")
    summary_df.to_csv(SUMMARY_OUT, index=False, encoding="utf-8")

    print("\nSaved outputs:")
    print(f"- {ROWS_OUT}")
    print(f"- {SUMMARY_OUT}")

    print("\nSummary by model:")
    if summary_df.empty:
        print("No matched rows were found.")
    else:
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
