#!/usr/bin/env python3
"""
Build IPA inventories (consonant manner x place and vowel height x backness)
for each source: target, 01, 02.

Target is read from `input/MGM_AFA4_2nd.txt` and models are read from the
IPA transcription CSVs.
"""
import os
import sys
import unicodedata
from collections import defaultdict
import pandas as pd

BASE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(BASE, ".."))
IPA_OUT = os.path.join(ROOT, "ipa", "output")
os.makedirs(IPA_OUT, exist_ok=True)

CONSONANT_PLACE_ORDER = [
    "bilabial",
    "labiodental",
    "dental",
    "alveolar",
    "postalveolar",
    "retroflex",
    "palatal",
    "velar",
    "uvular",
    "pharyngeal",
    "glottal",
    "labiovelar",
    "other",
]

CONSONANT_MANNER_ORDER = [
    "nasal",
    "stop",
    "affricate",
    "fricative",
    "approximant",
    "tap",
    "trill",
    "other",
]

VOWEL_HEIGHT_ORDER = ["close", "near-close", "close-mid", "mid", "open-mid", "near-open", "open", "other"]
VOWEL_BACKNESS_ORDER = ["front", "central", "back", "other"]

# Simple consonant classification maps
PLACE_BUCKETS = {
    "bilabial": set(list("pbmɓɸ")),
    "labiodental": set(list("fv")),
    "dental": set(list("tθð")),
    "alveolar": set(list("tdnslrɾz")),
    "postalveolar": set(list("ʃʒtʃdʒ")),
    "retroflex": set(list("ʈɖʂɳ")),
    "palatal": set(list("cɟjʝ")),
    "velar": set(list("kgŋqɢɰ")),
    "uvular": set(list("qɢʁ")),
    "pharyngeal": set(list("ħʕ")),
    "glottal": set(list("hʔ")),
    "labiovelar": set(["k͡p","g͡b","kp","k͡p","kʷ","ɡʷ"]),
}

MANNER_BUCKETS = {
    "nasal": set(list("m n ɲ ŋ ɳ ɴ".split())),
    "stop": set(list("p b t d k g ʔ ɟ c q ɢ".split())),
    "affricate": set(list("t͡ʃ t͡ɕ ts t͡s".split())),
    "fricative": set(list("f v θ ð s z ʃ ʒ ʂ ɕ χ x ɣ ħ ʕ h".split())),
    "approximant": set(list("j w ɹ ɰ ɻ l".split())),
    "tap": set(["ɾ"]),
    "trill": set(["r","ʙ"]),
}

# Vowel classification: map base vowel to (height, backness)
VOWEL_MAP = {
    "i": ("close","front"), "y": ("close","front"), "ɪ": ("near-close","front"),
    "e": ("close-mid","front"), "ɛ": ("open-mid","front"), "a": ("open","front"),
    "æ": ("near-open","front"),
    "ə": ("mid","central"), "ɨ": ("close","central"), "ɯ": ("close","back"),
    "u": ("close","back"), "ʊ": ("near-close","back"), "o": ("close-mid","back"), "ɔ": ("open-mid","back"),
    "ɒ": ("open","back"), "ɑ": ("open","back"),
}

def strip_marks(token):
    return "".join(ch for ch in token if not unicodedata.category(ch).startswith("M") )

def classify_consonant(token):
    t = strip_marks(token)
    if "͡" in t:
        parts = t.split("͡")
        core = parts[-1]
    else:
        core = t
    core = core.strip()
    for p, syms in PLACE_BUCKETS.items():
        for s in syms:
            if s and core.startswith(s):
                for m, ms in MANNER_BUCKETS.items():
                    if core in ms or any(core.startswith(x) for x in ms):
                        return m, p
                return "stop", p
    for m, ms in MANNER_BUCKETS.items():
        if core in ms:
            for p, syms in PLACE_BUCKETS.items():
                if any(core.startswith(x) for x in syms):
                    return m, p
            return m, "other"
    return "other", "other"

def classify_vowel(token):
    t = strip_marks(token)
    if not t:
        return None, None
    base = t[0]
    if base in VOWEL_MAP:
        return VOWEL_MAP[base]
    b = unicodedata.normalize("NFD", base)
    if b in VOWEL_MAP:
        return VOWEL_MAP[b]
    return None, None

def read_target_transcriptions(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        lines = [line.rstrip("\n") for line in fh if line.strip()]
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) >= 3:
            transcription = parts[2]
        elif len(parts) >= 2:
            transcription = parts[-1]
        else:
            transcription = parts[0]
        # Use the IPA-like portion before the first underscore when present.
        transcription = transcription.split("_")[0].strip()
        if transcription:
            rows.append(transcription)
    return rows

def read_model_transcriptions(path):
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path, sep=r"\s*,\s*", engine="python", dtype=str, keep_default_na=False, on_bad_lines="skip")
    column = None
    for candidate in ["ipa_transcription", "transcription", "text", "orig_text"]:
        if candidate in df.columns:
            column = candidate
            break
    if column is None:
        return []
    return [str(value).strip() for value in df[column].tolist() if str(value).strip()]

def build_inventories(source_texts):
    inventories = {}
    for src, texts in source_texts.items():
        cons_cells = defaultdict(lambda: defaultdict(set))
        vowels_cells = defaultdict(lambda: defaultdict(set))
        for text in texts:
            nfd = unicodedata.normalize('NFD', str(text))
            toks = [t for t in nfd.split() if t.strip()]
            for t in toks:
                base = strip_marks(t)
                if not base:
                    continue
                first = base[0]
                if first.lower() in VOWEL_MAP or first in 'aeiouyɑæɔɪʊɐəɜɵɒɨʉɯɤɰœøɶ':
                    height, back = classify_vowel(t)
                    if height and back:
                        vowels_cells[height][back].add(strip_marks(t))
                    else:
                        vowels_cells['other']['other'].add(strip_marks(t))
                else:
                    manner, place = classify_consonant(t)
                    cons_cells[manner][place].add(strip_marks(t))
        inventories[src] = (cons_cells, vowels_cells)
    return inventories

def ordered_keys(keys, preferred_order):
    present = [key for key in preferred_order if key in keys]
    extras = sorted(key for key in keys if key not in preferred_order)
    return present + extras

def save_matrix_cells(cells, out_prefix, kind, row_header, row_order, col_order):
    rows = ordered_keys(cells.keys(), row_order)
    cols = ordered_keys({col for row in cells.values() for col in row.keys()}, col_order)
    data = []
    for r in rows:
        rowd = {row_header: r}
        for c in cols:
            vals = cells[r].get(c, set())
            rowd[c] = " ".join(sorted(vals)) if vals else ""
        data.append(rowd)
    df = pd.DataFrame(data)
    out = os.path.join(IPA_OUT, f"{out_prefix}_{kind}_inventory.csv")
    df.to_csv(out, index=False)
    return out

def save_markdown_table(cells, out_prefix, kind, row_title, row_order, col_order):
    rows = ordered_keys(cells.keys(), row_order)
    cols = ordered_keys({col for row in cells.values() for col in row.keys()}, col_order)
    lines = []
    if kind == 'consonants':
        lines.append(f"# {out_prefix} consonant inventory")
        lines.append("place ->")
        lines.append("| Manner ↓ | " + " | ".join(cols) + " |")
    else:
        lines.append(f"# {out_prefix} vowel inventory")
        lines.append("backness ->")
        lines.append("| Height ↓ | " + " | ".join(cols) + " |")
    lines.append("|---|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        values = []
        for c in cols:
            vals = cells[r].get(c, set())
            values.append(" ".join(sorted(vals)) if vals else "")
        lines.append("| " + r + " | " + " | ".join(values) + " |")
    out = os.path.join(IPA_OUT, f"{out_prefix}_{kind}_inventory.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return out

def main():
    target_path = os.path.join(ROOT, "input", "MGM_AFA4_2nd.txt")
    model_01_path = os.path.join(IPA_OUT, "previously-run", "01b_ipa_transcriptions.csv")
    model_02_path = os.path.join(IPA_OUT, "previously-run", "02_ipa_transcriptions.csv")

    source_texts = {
        "target": read_target_transcriptions(target_path),
        "01": read_model_transcriptions(model_01_path),
        "02": read_model_transcriptions(model_02_path),
    }

    if not any(source_texts.values()):
        print("No source transcriptions found for inventory generation", file=sys.stderr)
        return 1

    inventories = build_inventories(source_texts)
    for src, (cons_cells, vowel_cells) in inventories.items():
        out1 = save_matrix_cells(cons_cells, src, 'consonants', 'Manner ↓', CONSONANT_MANNER_ORDER, CONSONANT_PLACE_ORDER)
        out2 = save_matrix_cells(vowel_cells, src, 'vowels', 'Height ↓', VOWEL_HEIGHT_ORDER, VOWEL_BACKNESS_ORDER)
        md1 = save_markdown_table(cons_cells, src, 'consonants', 'Manner ↓', CONSONANT_MANNER_ORDER, CONSONANT_PLACE_ORDER)
        md2 = save_markdown_table(vowel_cells, src, 'vowels', 'Height ↓', VOWEL_HEIGHT_ORDER, VOWEL_BACKNESS_ORDER)
        print(f"Wrote inventories for {src}: {out1}, {out2}, {md1}, {md2}")
    return 0

if __name__ == '__main__':
    sys.exit(main())
