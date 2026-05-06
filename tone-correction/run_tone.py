"""
Tone Detection Stage - Pitch Analysis and Tone Classification
Analyzes pitch contours to identify tones in Rengmitca
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR / "tone-correction"
IPA_RESULTS = BASE_DIR / "ipa" / "output" / "ipa_transcriptions.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
RESULTS_FILE = OUTPUT_DIR / "tone_analysis.csv"

OUTPUT_DIR.mkdir(exist_ok=True)

print("Tone Detection Stage - CPU Mode")


class LibrosaToneAnalyzer:
    """Tone analysis using librosa pitch tracking"""
    
    def __init__(self):
        self.name = "librosa-pitch"
        
    def load(self):
        print(f"✓ Loaded {self.name} (CPU mode)")
        return True
    
    def analyze(self, audio_path):
        """Analyze pitch contour and classify tone"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Extract F0 using YIN algorithm
            f0 = librosa.yin(y, fmin=75, fmax=400, sr=sr)
            
            # Remove unvoiced regions (f0 = 0)
            valid_f0 = f0[f0 > 0]
            
            if len(valid_f0) == 0:
                return {
                    'detected_tones': 'UNKNOWN',
                    'pitch_contour': [],
                    'tone_category': 'UNKNOWN',
                    'mean_f0': 0.0,
                    'f0_std': 0.0,
                    'f0_range': 0.0
                }
            
            # Calculate features
            mean_f0 = np.mean(valid_f0)
            std_f0 = np.std(valid_f0)
            min_f0 = np.min(valid_f0)
            max_f0 = np.max(valid_f0)
            f0_range = max_f0 - min_f0
            
            # Normalize F0 to identify contour pattern
            if len(valid_f0) > 2:
                norm_f0 = (valid_f0 - min_f0) / (f0_range if f0_range > 0 else 1)
                
                # Simple tone classification based on contour
                start = np.mean(norm_f0[:len(norm_f0)//3])
                middle = np.mean(norm_f0[len(norm_f0)//3:2*len(norm_f0)//3])
                end = np.mean(norm_f0[2*len(norm_f0)//3:])
                
                # Classify tone pattern
                if end > start + 0.2:
                    tone_category = "RISING"
                elif end < start - 0.2:
                    tone_category = "FALLING"
                elif abs(middle - start) > 0.2 and abs(end - start) < 0.15:
                    if middle > start:
                        tone_category = "RISING-FALLING"
                    else:
                        tone_category = "FALLING-RISING"
                elif mean_f0 > 200:
                    tone_category = "HIGH"
                elif mean_f0 < 120:
                    tone_category = "LOW"
                else:
                    tone_category = "MID"
            else:
                tone_category = "SHORT"
            
            return {
                'detected_tones': tone_category,
                'pitch_contour': valid_f0.tolist()[:50],  # Limit array size
                'tone_category': tone_category,
                'mean_f0': float(mean_f0),
                'f0_std': float(std_f0),
                'f0_range': float(f0_range),
                'min_f0': float(min_f0),
                'max_f0': float(max_f0)
            }
            
        except Exception as e:
            print(f"  Error analyzing {audio_path}: {e}")
            return {
                'detected_tones': 'ERROR',
                'pitch_contour': [],
                'tone_category': 'ERROR',
                'mean_f0': 0.0,
                'f0_std': 0.0,
                'f0_range': 0.0
            }


class ParsemouthToneAnalyzer:
    """Tone analysis using Parselmouth/PRAAT"""
    
    def __init__(self):
        self.name = "parselmouth"
        self.parselmouth = None
        
    def load(self):
        try:
            import parselmouth
            self.parselmouth = parselmouth
            print(f"✓ Loaded {self.name} (CPU mode)")
            return True
        except ImportError:
            print(f"✗ {self.name} not available (install: pip install praat-parselmouth)")
            return False
    
    def analyze(self, audio_path):
        """Analyze pitch using Parselmouth"""
        try:
            # Load sound
            snd = self.parselmouth.Sound(str(audio_path))
            
            # Extract pitch
            pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=400)
            
            # Get pitch values
            pitch_values = pitch.selected_array['frequency']
            valid_pitch = pitch_values[pitch_values > 0]
            
            if len(valid_pitch) == 0:
                return {
                    'detected_tones': 'UNKNOWN',
                    'pitch_contour': [],
                    'tone_category': 'UNKNOWN',
                    'mean_f0': 0.0
                }
            
            mean_f0 = np.mean(valid_pitch)
            
            # Simple classification
            if mean_f0 > 200:
                tone_category = "HIGH"
            elif mean_f0 < 120:
                tone_category = "LOW"
            else:
                tone_category = "MID"
            
            return {
                'detected_tones': tone_category,
                'pitch_contour': valid_pitch.tolist()[:50],
                'tone_category': tone_category,
                'mean_f0': float(mean_f0)
            }
            
        except Exception as e:
            print(f"  Error with parselmouth: {e}")
            return {
                'detected_tones': 'ERROR',
                'pitch_contour': [],
                'tone_category': 'ERROR',
                'mean_f0': 0.0
            }


def main():
    """Run tone detection on IPA-transcribed segments"""
    
    print("=" * 80)
    print("TONE DETECTION - Pitch Analysis (CPU Mode)")
    print("=" * 80)
    print()
    
    # Load IPA results
    if not IPA_RESULTS.exists():
        print(f"ERROR: IPA results not found: {IPA_RESULTS}")
        print("Please run IPA transcription first!")
        return
    
    ipa_df = pd.read_csv(IPA_RESULTS)
    print(f"Loaded {len(ipa_df)} IPA transcriptions")
    
    # Get unique segments
    unique_segments = ipa_df.drop_duplicates(subset=['segment_filename'])
    print(f"Analyzing {len(unique_segments)} unique segments")
    print()
    
    # Initialize analyzers
    analyzers = [
        LibrosaToneAnalyzer(),
    ]
    
    # Try to add Parselmouth
    pm_analyzer = ParsemouthToneAnalyzer()
    if pm_analyzer.load():
        analyzers.append(pm_analyzer)
    
    # Load analyzers
    print("Loading analyzers...")
    loaded_analyzers = []
    for analyzer in analyzers:
        if analyzer.load():
            loaded_analyzers.append(analyzer)
    print()
    
    if not loaded_analyzers:
        print("ERROR: No analyzers loaded")
        return
    
    # Run tone analysis
    all_results = []
    
    for analyzer in loaded_analyzers:
        print(f"\n{'=' * 80}")
        print(f"Running {analyzer.name}...")
        print('=' * 80)
        
        start_time = datetime.now()
        
        for _, row in tqdm(unique_segments.iterrows(), total=len(unique_segments), desc=analyzer.name):
            audio_path = Path(row['audio_path'])
            
            if not audio_path.exists():
                continue
            
            # Analyze tone
            result = analyzer.analyze(audio_path)
            
            # Combine with segment info
            combined = {
                'segment_filename': row['segment_filename'],
                'speaker_id': row['speaker_id'],
                'start_time_ms': row['start_time_ms'],
                'end_time_ms': row['end_time_ms'],
                'duration_ms': row['duration_ms'],
                'ipa_transcription': row['ipa_transcription'],
                'tone_model': analyzer.name,
                'detected_tones': result['detected_tones'],
                'tone_category': result['tone_category'],
                'mean_f0': result.get('mean_f0', 0.0),
                'f0_std': result.get('f0_std', 0.0),
                'f0_range': result.get('f0_range', 0.0),
                'audio_path': str(audio_path)
            }
            
            all_results.append(combined)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        print(f"\n✓ Analyzed {len(unique_segments)} segments")
        print(f"  Processing time: {processing_time:.2f} seconds")
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(RESULTS_FILE, index=False)
        
        print(f"\n{'=' * 80}")
        print(f"✓ Results saved to {RESULTS_FILE}")
        
        # Tone inventory
        print("\nTone Inventory:")
        for model in df['tone_model'].unique():
            model_df = df[df['tone_model'] == model]
            print(f"\n{model}:")
            tone_counts = model_df['tone_category'].value_counts()
            for tone, count in tone_counts.items():
                pct = (count / len(model_df)) * 100
                print(f"  {tone}: {count} ({pct:.1f}%)")
        
        # Save tone inventory
        inventory = {}
        for model in df['tone_model'].unique():
            model_df = df[df['tone_model'] == model]
            inventory[model] = model_df['tone_category'].value_counts().to_dict()
        
        inventory_file = OUTPUT_DIR / "tone_inventory.json"
        with open(inventory_file, 'w') as f:
            json.dump(inventory, f, indent=2)
        
        print(f"\n✓ Tone inventory saved to {inventory_file}")
    else:
        print("\n⚠ No tone analysis results generated")
    
    print("\n" + "=" * 80)
    print("TONE DETECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
