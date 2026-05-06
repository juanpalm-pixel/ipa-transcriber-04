"""
Extract and classify IPA characters from model transcriptions, separated by model.

Reads from ipa_transcriptions.csv and generates separate inventories for each model:
- consonant_inventory_<model>.csv
- vowel_inventory_<model>.csv
- diacritics_inventory_<model>.csv
"""
from pathlib import Path
import pandas as pd
import unicodedata
from collections import defaultdict, Counter


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "ipa" / "output" / "ipa_transcriptions.csv"
OUTPUT_DIR = BASE_DIR / "ipa" / "output"


# IPA consonant inventory with classification
CONSONANT_CHART = {
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
    'm': {'manner': 'nasal', 'place': 'bilabial', 'voiced': True},
    'n': {'manner': 'nasal', 'place': 'alveolar', 'voiced': True},
    'ŋ': {'manner': 'nasal', 'place': 'velar', 'voiced': True},
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
    't͡ʃ': {'manner': 'affricate', 'place': 'postalveolar', 'voiced': False},
    'd͡ʒ': {'manner': 'affricate', 'place': 'postalveolar', 'voiced': True},
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

TONE_MARKS = {
    '\u0300': 'falling',
    '\u0301': 'rising',
    '\u0304': 'flat',
    '\u0302': 'rising-falling',
    '\u0306': 'short',
}

DIACRITICS = {
    'ʰ': 'aspiration',
    'ː': 'long',
    '̃': 'nasalized',
}


def parse_transcription(transcription: str):
    """Parse a transcription into characters, handling affricates, ties, and modifiers."""
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
            char = transcription[i]
            i += 1
            while i < len(transcription):
                cat = unicodedata.category(transcription[i])
                if cat.startswith('M') or cat == 'Lm':
                    char += transcription[i]
                    i += 1
                else:
                    break
            if char.strip():
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
    if char in CONSONANT_CHART:
        return char
    
    nfd = unicodedata.normalize('NFD', char)
    base = nfd[0]
    if base in CONSONANT_CHART:
        return base
    
    for cons in CONSONANT_CHART:
        if cons in char:
            return cons
    
    return None


def find_vowel_base(char: str) -> str:
    """Find the base vowel in a character with diacritics."""
    if char in VOWEL_CHART:
        return char
    
    nfd = unicodedata.normalize('NFD', char)
    base = nfd[0]
    if base in VOWEL_CHART:
        return base
    
    return None


def process_model_transcriptions(df, model_name, trans_col):
    """Process transcriptions for a single model."""
    model_df = df[df['ipa_model'] == model_name]
    transcriptions = model_df[trans_col].fillna('').astype(str)
    
    all_chars = set()
    char_frequencies = Counter()
    consonant_contexts = defaultdict(lambda: defaultdict(int))
    
    for trans in transcriptions:
        chars = parse_transcription(trans)
        for char in chars:
            if char and char not in (' ', '\t', '\n'):
                all_chars.add(char)
                char_frequencies[char] += 1
                
                cons_base = find_consonant_base(char)
                if cons_base:
                    for diac in DIACRITICS.keys():
                        if diac in char:
                            consonant_contexts[diac][cons_base] += 1
    
    # Consonants
    consonants_data = []
    for char in sorted(all_chars):
        base_consonant = find_consonant_base(char)
        if base_consonant and not find_vowel_base(char):
            info = CONSONANT_CHART[base_consonant]
            has_aspiration = 'ʰ' in char
            has_long = 'ː' in char
            
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
    
    # Vowels
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
    
    # Diacritics
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
    
    return consonants_df, vowels_df, diacritics_df


def main():
    # Read model transcriptions
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        return
    
    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.strip()
    
    # Find transcription column
    trans_col = None
    for col in df.columns:
        if col.lower() in ('ipa_transcriptions', 'ipa_transcription'):
            trans_col = col
            break
    
    if not trans_col:
        print(f"ERROR: Could not find transcription column. Available: {list(df.columns)}")
        return
    
    # Get unique models
    models = df['ipa_model'].unique()
    print(f"Found {len(models)} model(s): {', '.join(sorted(models))}\n")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each model
    for model in sorted(models):
        print(f"\nProcessing {model}...")
        consonants_df, vowels_df, diacritics_df = process_model_transcriptions(df, model, trans_col)
        
        # Sanitize model name for filename
        safe_model = model.lower().replace(' ', '_').replace('/', '_')
        
        consonants_file = OUTPUT_DIR / f"consonant_inventory_{safe_model}.csv"
        vowels_file = OUTPUT_DIR / f"vowel_inventory_{safe_model}.csv"
        diacritics_file = OUTPUT_DIR / f"diacritics_inventory_{safe_model}.csv"
        
        consonants_df.to_csv(consonants_file, index=False)
        vowels_df.to_csv(vowels_file, index=False)
        diacritics_df.to_csv(diacritics_file, index=False)
        
        print(f"  Consonants: {len(consonants_df)} variations")
        print(f"  Vowels: {len(vowels_df)} variations")
        print(f"  Diacritics: {len(diacritics_df)} types")
        
        if len(consonants_df) > 0:
            print(f"  Top consonants: {', '.join(consonants_df.head(3)['character'].tolist())}")
        if len(vowels_df) > 0:
            print(f"  Top vowels: {', '.join(vowels_df.head(3)['character'].tolist())}")
    
    print("\n" + "=" * 80)
    print("MODEL IPA INVENTORY EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to {OUTPUT_DIR}:")
    for model in sorted(models):
        safe_model = model.lower().replace(' ', '_').replace('/', '_')
        print(f"  {model}:")
        print(f"    - consonant_inventory_{safe_model}.csv")
        print(f"    - vowel_inventory_{safe_model}.csv")
        print(f"    - diacritics_inventory_{safe_model}.csv")


if __name__ == "__main__":
    main()
