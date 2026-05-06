"""
Extract and classify all IPA characters from transcriptions into consonants, vowels, and diacritics.

Outputs three dataframes:
1. consonants_inventory.csv - Each unique consonant variation with IPA classification
2. vowels_inventory.csv - Each unique vowel variation with tone, length, and classification
3. diacritics_inventory.csv - All diacritics and their environments
"""
from pathlib import Path
import pandas as pd
import unicodedata
import sys
from collections import defaultdict, Counter


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "input" / "MGM_AFA4_2nd.txt"
OUTPUT_DIR = BASE_DIR / "ipa" / "output"


# IPA consonant inventory with classification
CONSONANT_CHART = {
    # Plosives
    'p': {'manner': 'plosive', 'place': 'bilabial', 'voiced': False},
    'b': {'manner': 'plosive', 'place': 'bilabial', 'voiced': True},
    't': {'manner': 'plosive', 'place': 'alveolar', 'voiced': False},
    'd': {'manner': 'plosive', 'place': 'alveolar', 'voiced': True},
    'ɖ': {'manner': 'plosive', 'place': 'retroflex', 'voiced': True},
    'c': {'manner': 'plosive', 'place': 'palatal', 'voiced': False},
    'k': {'manner': 'plosive', 'place': 'velar', 'voiced': False},
    'g': {'manner': 'plosive', 'place': 'velar', 'voiced': True},
    'q': {'manner': 'plosive', 'place': 'uvular', 'voiced': False},
    'ʔ': {'manner': 'plosive', 'place': 'glottal', 'voiced': False},
    # Nasals
    'm': {'manner': 'nasal', 'place': 'bilabial', 'voiced': True},
    'n': {'manner': 'nasal', 'place': 'alveolar', 'voiced': True},
    'ŋ': {'manner': 'nasal', 'place': 'velar', 'voiced': True},
    # Fricatives
    'f': {'manner': 'fricative', 'place': 'labiodental', 'voiced': False},
    'v': {'manner': 'fricative', 'place': 'labiodental', 'voiced': True},
    'θ': {'manner': 'fricative', 'place': 'dental', 'voiced': False},
    'ð': {'manner': 'fricative', 'place': 'dental', 'voiced': True},
    's': {'manner': 'fricative', 'place': 'alveolar', 'voiced': False},
    'z': {'manner': 'fricative', 'place': 'alveolar', 'voiced': True},
    'ʃ': {'manner': 'fricative', 'place': 'postalveolar', 'voiced': False},
    'ʒ': {'manner': 'fricative', 'place': 'postalveolar', 'voiced': True},
    'ç': {'manner': 'fricative', 'place': 'palatal', 'voiced': False},
    'x': {'manner': 'fricative', 'place': 'velar', 'voiced': False},
    'ɣ': {'manner': 'fricative', 'place': 'velar', 'voiced': True},
    'h': {'manner': 'fricative', 'place': 'glottal', 'voiced': False},
    # Affricates
    't͡ʃ': {'manner': 'affricate', 'place': 'postalveolar', 'voiced': False},
    'd͡ʒ': {'manner': 'affricate', 'place': 'postalveolar', 'voiced': True},
    # Approximants
    'j': {'manner': 'approximant', 'place': 'palatal', 'voiced': True},
    'w': {'manner': 'approximant', 'place': 'labial-velar', 'voiced': True},
    'ɹ': {'manner': 'approximant', 'place': 'alveolar', 'voiced': True},
    'r': {'manner': 'tap', 'place': 'alveolar', 'voiced': True},
    'l': {'manner': 'approximant', 'place': 'alveolar', 'voiced': True},
}

# IPA vowel chart with classification
VOWEL_CHART = {
    'i': {'height': 'close', 'backness': 'front', 'rounded': False},
    'y': {'height': 'close', 'backness': 'front', 'rounded': True},
    'ɨ': {'height': 'close', 'backness': 'central', 'rounded': False},
    'ʉ': {'height': 'close', 'backness': 'central', 'rounded': True},
    'ɯ': {'height': 'close', 'backness': 'back', 'rounded': False},
    'u': {'height': 'close', 'backness': 'back', 'rounded': True},
    'e': {'height': 'close-mid', 'backness': 'front', 'rounded': False},
    'ø': {'height': 'close-mid', 'backness': 'front', 'rounded': True},
    'ə': {'height': 'mid', 'backness': 'central', 'rounded': False},
    'ɘ': {'height': 'close-mid', 'backness': 'central', 'rounded': False},
    'ɜ': {'height': 'open-mid', 'backness': 'central', 'rounded': False},
    'ʌ': {'height': 'open-mid', 'backness': 'back', 'rounded': False},
    'ɔ': {'height': 'open-mid', 'backness': 'back', 'rounded': True},
    'ɛ': {'height': 'open-mid', 'backness': 'front', 'rounded': False},
    'œ': {'height': 'open-mid', 'backness': 'front', 'rounded': True},
    'a': {'height': 'open', 'backness': 'front', 'rounded': False},
    'ɑ': {'height': 'open', 'backness': 'back', 'rounded': False},
    'ɒ': {'height': 'open', 'backness': 'back', 'rounded': True},
    'I': {'height': 'near-close', 'backness': 'front', 'rounded': False},
    'Y': {'height': 'near-close', 'backness': 'front', 'rounded': True},
    'ʊ': {'height': 'near-close', 'backness': 'back', 'rounded': True},
    'ʏ': {'height': 'near-close', 'backness': 'front', 'rounded': True},
}

# Tone marks (combining characters)
TONE_MARKS = {
    '\u0300': 'falling',          # à (combining grave accent)
    '\u0301': 'rising',           # á (combining acute accent)
    '\u0304': 'flat',             # ā (combining macron)
    '\u0302': 'rising-falling',   # â (combining circumflex)
    '\u0306': 'short',            # ă (combining breve)
}

# Other diacritics
DIACRITICS = {
    'ʰ': 'aspiration',
    'ː': 'long',
    '̃': 'nasalized',
}


def parse_transcription(transcription: str):
    """Parse a transcription into characters, handling affricates, ties, and modifiers."""
    # Replace affricates with placeholder to preserve them
    transcription = transcription.replace('t͡ʃ', '__TAFFRICATE__')
    transcription = transcription.replace('d͡ʒ', '__DAFFRICATE__')
    
    chars = []
    i = 0
    while i < len(transcription):
        if transcription[i:i+13] == '__TAFFRICATE__':
            chars.append('t͡ʃ')
            i += 13
        elif transcription[i:i+13] == '__DAFFRICATE__':
            chars.append('d͡ʒ')
            i += 13
        else:
            # Collect base + combining diacritics + modifier letters as one grapheme
            char = transcription[i]
            i += 1
            # Collect combining marks (category M*) and modifier letters (Lm)
            while i < len(transcription):
                cat = unicodedata.category(transcription[i])
                if cat.startswith('M') or cat == 'Lm':
                    char += transcription[i]
                    i += 1
                else:
                    break
            if char.strip():  # Skip whitespace
                chars.append(char)
    return chars


def extract_tone(char: str) -> str:
    """Extract tone from combining diacritics."""
    nfd = unicodedata.normalize('NFD', char)
    for d in nfd[1:]:
        if d in TONE_MARKS:
            return TONE_MARKS[d]
    return 'none'


def find_consonant_base(char: str) -> str:
    """Find the base consonant in a character with diacritics."""
    # Try exact match first
    if char in CONSONANT_CHART:
        return char
    
    # Try base character (remove combining marks)
    nfd = unicodedata.normalize('NFD', char)
    base = nfd[0]
    if base in CONSONANT_CHART:
        return base
    
    # Try substring matches (for affricates with diacritics)
    for cons in CONSONANT_CHART:
        if cons in char:
            return cons
    
    return None


def find_vowel_base(char: str) -> str:
    """Find the base vowel in a character with diacritics."""
    # Try exact match first
    if char in VOWEL_CHART:
        return char
    
    # Try base character (remove combining marks)
    nfd = unicodedata.normalize('NFD', char)
    base = nfd[0]
    if base in VOWEL_CHART:
        return base
    
    return None


def main():
    # Read input file manually to handle irregular spacing
    transcriptions_list = []
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Skip header
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        # Split on first two whitespace sequences to get the transcription
        parts = line.split(None, 2)  # Split on whitespace, max 3 parts
        if len(parts) >= 3:
            transcriptions_list.append(parts[2])
    
    transcriptions = transcriptions_list
    
    # Extract all characters
    all_chars = set()
    char_frequencies = Counter()
    consonant_contexts = defaultdict(lambda: defaultdict(int))
    
    for trans in transcriptions:
        # Remove metadata (after underscore)
        trans_clean = trans.split('_')[0] if '_' in trans else trans
        chars = parse_transcription(trans_clean)
        for i, char in enumerate(chars):
            if char and char not in (' ', '\t', '\n'):
                all_chars.add(char)
                char_frequencies[char] += 1
                
                # Track consonant contexts for diacritics
                cons_base = find_consonant_base(char)
                if cons_base:
                    for diac in DIACRITICS.keys():
                        if diac in char:
                            consonant_contexts[diac][cons_base] += 1
    
    # Classify consonants
    consonants_data = []
    for char in sorted(all_chars):
        base_consonant = find_consonant_base(char)
        
        if base_consonant:
            info = CONSONANT_CHART[base_consonant]
            has_aspiration = 'ʰ' in char
            has_long = 'ː' in char
            
            # Only add if it's a consonant (not a vowel or other)
            if not find_vowel_base(char):  # Make sure it's not also a vowel
                consonants_data.append({
                    'character': char,
                    'base_consonant': base_consonant,
                    'manner': info['manner'],
                    'place': info['place'],
                    'voiced': info['voiced'],
                    'aspiration': has_aspiration,
                    'elongation': has_long,
                    'frequency': char_frequencies[char],
                })
    
    # Classify vowels
    vowels_data = []
    for char in sorted(all_chars):
        base_vowel = find_vowel_base(char)
        
        if base_vowel:
            info = VOWEL_CHART[base_vowel]
            tone = extract_tone(char)
            has_long = 'ː' in char
            has_nasalized = '̃' in char
            
            vowels_data.append({
                'character': char,
                'base_vowel': base_vowel,
                'height': info['height'],
                'backness': info['backness'],
                'rounded': info['rounded'],
                'tone': tone,
                'elongation': has_long,
                'nasalized': has_nasalized,
                'frequency': char_frequencies[char],
            })
    
    # Diacritics and environments
    diacritics_data = []
    for diac, cons_dict in consonant_contexts.items():
        env_str = '; '.join(f"{cons}(×{count})" for cons, count in sorted(cons_dict.items(), key=lambda x: -x[1]))
        diacritics_data.append({
            'diacritic': diac,
            'meaning': DIACRITICS.get(diac, 'unknown'),
            'consonant_environments': env_str,
            'total_occurrences': sum(cons_dict.values()),
        })
    
    # Create dataframes
    consonants_df = pd.DataFrame(consonants_data).drop_duplicates(subset=['character']).sort_values('frequency', ascending=False).reset_index(drop=True)
    vowels_df = pd.DataFrame(vowels_data).drop_duplicates(subset=['character']).sort_values('frequency', ascending=False).reset_index(drop=True)
    diacritics_df = pd.DataFrame(diacritics_data)
    
    # Save to CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    consonants_df.to_csv(OUTPUT_DIR / 'consonants_inventory.csv', index=False)
    vowels_df.to_csv(OUTPUT_DIR / 'vowels_inventory.csv', index=False)
    diacritics_df.to_csv(OUTPUT_DIR / 'diacritics_inventory.csv', index=False)
    
    print("=" * 80)
    print("IPA CHARACTER INVENTORY EXTRACTION")
    print("=" * 80)
    print(f"\nConsonants found: {len(consonants_df)}")
    print(consonants_df.to_string(index=False))
    
    print(f"\n\nVowels found: {len(vowels_df)}")
    print(vowels_df.to_string(index=False))
    
    print(f"\n\nDiacritics found: {len(diacritics_df)}")
    print(diacritics_df.to_string(index=False))
    
    print(f"\n\nOutputs saved to {OUTPUT_DIR}:")
    print(f"- consonants_inventory.csv")
    print(f"- vowels_inventory.csv")
    print(f"- diacritics_inventory.csv")


if __name__ == "__main__":
    main()
