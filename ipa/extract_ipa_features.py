#!/usr/bin/env python3
"""
Extract IPA phonetic component features from three sources:
- input/MGM_AFA4_2nd.txt (TSV) -> target
- ipa/output/ipa_transcriptions.csv -> 01

Outputs written to ipa/output/:
- target_ipa_features.csv
- ipa_features.csv
- phonetic_components_rows.csv
- phonetic_components_summary.csv
"""
import os
import sys
import unicodedata
from collections import defaultdict, Counter
import pandas as pd

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
INPUT_DIR = os.path.join(WORKSPACE_ROOT, "input")
IPA_OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "ipa", "output")

os.makedirs(IPA_OUTPUT_DIR, exist_ok=True)

# Vowel set (base characters, lowercase)
VOWELS = set([
    "a","e","i","o","u","y",
    "ɑ","æ","ɔ","ɪ","ʊ","ɐ","ə","ɜ","ɞ","ɵ","ɒ","ɨ","ʉ","ɯ","ɤ","ɰ","ʏ","œ","ø",
    "ɶ"
])
# normalize vowel set to NFD base characters (lower)
VOWELS = set(unicodedata.normalize("NFD", v).lower() for v in VOWELS)

def find_cols(df):
    cols = [c for c in df.columns]
    id_col = None
    tr_col = None
    for c in cols:
        cl = c.lower()
        if cl in ("id","utterance_id","utt_id","utt","row_id"):
            id_col = c
        if cl in ("transcription","transcripts","transcript","trans","text","orig_text","ipa","ipa_transcription"):
            tr_col = c
    if id_col is None:
        id_col = cols[0] if cols else None
    if tr_col is None:
        if len(cols) >= 2:
            tr_col = cols[1]
        elif cols:
            tr_col = cols[0]
        else:
            tr_col = None
    return id_col, tr_col

def read_input_file(path, sep=",", source_name=""):
    if not os.path.exists(path):
        print(f"Warning: input file missing, skipping: {path}", file=sys.stderr)
        return None
    # Special-case target TXT files that may have a header with spaces
    if source_name == "target":
        rows = []
        try:
            with open(path, "r", encoding="utf-8") as fh:
                lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
        except Exception:
            with open(path, "r", encoding="latin-1") as fh:
                lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
        if not lines:
            return None
        # skip header line
        for i, ln in enumerate(lines[1:]):
            parts = ln.split("\t")
            if len(parts) >= 3:
                trans = parts[2]
            elif len(parts) >= 2:
                trans = parts[-1]
            else:
                trans = parts[0]
            uid = f"target_{i+1}"
            rows.append({"id": uid, "TRANSCRIPTION": trans})
        return pd.DataFrame(rows)

    # Use a flexible separator that matches comma with optional surrounding whitespace
    try:
        df = pd.read_csv(path, sep=r"\s*,\s*", engine="python", dtype=str, keep_default_na=False, on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(path, sep=r"\s*,\s*", engine="python", dtype=str, encoding="utf-8", keep_default_na=False, on_bad_lines="skip")
    id_col, tr_col = find_cols(df)
    df = df.rename(columns={id_col: "id", tr_col: "TRANSCRIPTION"})
    return df[["id","TRANSCRIPTION"]].copy()

def tokenize_nfd(s):
    tokens = []
    i = 0
    s = s or ""
    L = len(s)
    while i < L:
        ch = s[i]
        cat = unicodedata.category(ch)
        if cat[0] == "Z" or cat[0] == "P":
            i += 1
            continue
        if cat[0] != "M":
            token_chars = [ch]
            j = i+1
            while j < L:
                ch2 = s[j]
                if unicodedata.category(ch2)[0] == "M":
                    token_chars.append(ch2)
                    j += 1
                else:
                    break
            tokens.append("".join(token_chars))
            i = j
        else:
            i += 1
    return tokens

def classify_token(token):
    if not token:
        return None, []
    base = token[0]
    base_norm = base.lower()
    is_vowel = base_norm in VOWELS
    diacs = [c for c in token if unicodedata.category(c).startswith("M")]
    return ("vowel" if is_vowel else "consonant"), diacs

def detect_tones(nfd_text, nfc_text, row_diacs_set):
    tones = set()
    if "\u0301" in row_diacs_set:
        tones.add("rising")
    if "\u0300" in row_diacs_set:
        tones.add("falling")
    if "˩˥" in nfc_text:
        tones.add("rising")
    if "˥˩" in nfc_text:
        tones.add("falling")
    if not tones:
        return "none"
    return ";".join(sorted(tones))

def process_df(df, source_label):
    rows = []
    for _, r in df.iterrows():
        uid = r.get("id", "")
        orig = r.get("TRANSCRIPTION", "")
        nfd = unicodedata.normalize("NFD", str(orig))
        nfc = unicodedata.normalize("NFC", str(orig))
        tokens = tokenize_nfd(nfd)
        vowels = []
        consonants = []
        diac_set = set()
        for t in tokens:
            cls, diacs = classify_token(t)
            if cls == "vowel":
                vowels.append(t)
            elif cls == "consonant":
                consonants.append(t)
            for d in diacs:
                diac_set.add(d)
        diacritics_str = " ".join(sorted(diac_set))
        detected = detect_tones(nfd, nfc, diac_set)
        rows.append({
            "source": source_label,
            "id": uid,
            "orig_text": orig,
            "vowels": " ".join(vowels),
            "consonants": " ".join(consonants),
            "diacritics": diacritics_str,
            "detected_tones": detected
        })
    return pd.DataFrame(rows)

def main():
    sources = []

    target_path = os.path.join(WORKSPACE_ROOT, "input", "MGM_AFA4_2nd.txt")
    df_target = read_input_file(target_path, sep="\t", source_name="target")
    if df_target is not None:
        sources.append(("target", df_target))

    path_01 = os.path.join(WORKSPACE_ROOT, "ipa", "output", "01-ipa_transcriptions.csv")
    df_01 = read_input_file(path_01, sep=",", source_name="01")
    if df_01 is not None:
        sources.append(("01", df_01))

    path_02 = os.path.join(WORKSPACE_ROOT, "ipa", "output", "02_ipa_transcriptions.csv")
    df_02 = read_input_file(path_02, sep=",", source_name="02")
    if df_02 is not None:
        sources.append(("02", df_02))

    if not sources:
        print("No input sources found. Exiting.", file=sys.stderr)
        return 2

    all_rows = []
    summary_counts = []

    for label, df in sources:
        df_proc = process_df(df, label)
        if label == "target":
            out_file = os.path.join(IPA_OUTPUT_DIR, "target_ipa_features.csv")
        elif label == "01":
            out_file = os.path.join(IPA_OUTPUT_DIR, "ipa_features.csv")
        df_proc.to_csv(out_file, index=False)
        print(f"Wrote {out_file}")
        all_rows.append(df_proc)
        total = len(df_proc)
        rising = df_proc["detected_tones"].str.contains("rising", na=False).sum()
        falling = df_proc["detected_tones"].str.contains("falling", na=False).sum()
        none = (df_proc["detected_tones"] == "none").sum()
        summary_counts.append({
            "source": label,
            "rows_total": total,
            "rising_count": int(rising),
            "falling_count": int(falling),
            "none_count": int(none)
        })

    combined_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=["source","id","orig_text","vowels","consonants","diacritics","detected_tones"])
    combined_path = os.path.join(IPA_OUTPUT_DIR, "phonetic_components_rows.csv")
    combined_df.to_csv(combined_path, index=False)
    print(f"Wrote {combined_path}")

    summary_df = pd.DataFrame(summary_counts)
    summary_path = os.path.join(IPA_OUTPUT_DIR, "phonetic_components_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    print(summary_path)

if __name__ == "__main__":
    sys.exit(main())
