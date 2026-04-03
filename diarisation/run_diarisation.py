"""
Diarisation Stage - Speaker Identification
Assigns speaker labels to segmented audio clips.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import torchaudio
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json

# Configuration
SEGMENTATION_RESULTS = Path("../segmentation/segmentation_results.csv")
OUTPUT_DIR = Path("output")
MODELS_DIR = Path("models")
RESULTS_FILE = "diarisation_results.csv"

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# HuggingFace token
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    print("WARNING: HF_TOKEN not found in environment")


class PyAnnoteDiariser:
    """Speaker diarisation using pyannote.audio"""
    
    def __init__(self, model_name="pyannote/speaker-diarization-3.1"):
        self.name = model_name.split('/')[-1]
        self.model_name = model_name
        self.pipeline = None
        
    def load(self):
        try:
            from pyannote.audio import Pipeline
            
            self.pipeline = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=HF_TOKEN
            )
            
            if torch.cuda.is_available():
                self.pipeline.to(torch.device("cuda"))
            
            print(f"✓ Loaded {self.name}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load {self.name}: {e}")
            return False
    
    def diarise_segment(self, audio_path, num_speakers=3):
        """Diarise a single audio segment"""
        try:
            # Run diarisation
            diarization = self.pipeline(
                str(audio_path),
                num_speakers=num_speakers
            )
            
            # Get dominant speaker for this segment
            speaker_durations = {}
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if speaker not in speaker_durations:
                    speaker_durations[speaker] = 0
                speaker_durations[speaker] += turn.duration
            
            # Get speaker with most time
            if speaker_durations:
                dominant_speaker = max(speaker_durations.items(), key=lambda x: x[1])
                return {
                    'speaker_id': dominant_speaker[0],
                    'confidence': dominant_speaker[1] / sum(speaker_durations.values()),
                    'all_speakers': speaker_durations
                }
            else:
                return {
                    'speaker_id': 'UNKNOWN',
                    'confidence': 0.0,
                    'all_speakers': {}
                }
                
        except Exception as e:
            print(f"  Error processing {audio_path}: {e}")
            return {
                'speaker_id': 'ERROR',
                'confidence': 0.0,
                'all_speakers': {}
            }


class SimplePitchDiariser:
    """Simple speaker diarisation based on pitch (F0) analysis"""
    
    def __init__(self):
        self.name = "simple-pitch"
        self.speaker_thresholds = {
            'female': (150, 350),  # Hz range for female speaker
            'male': (75, 150)      # Hz range for male speakers
        }
        
    def load(self):
        print(f"✓ Loaded {self.name}")
        return True
    
    def diarise_segment(self, audio_path):
        """Diarise based on fundamental frequency"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Extract pitch using librosa
            f0 = librosa.yin(y, fmin=75, fmax=400, sr=sr)
            
            # Get valid (non-zero) f0 values
            valid_f0 = f0[f0 > 0]
            
            if len(valid_f0) == 0:
                return {
                    'speaker_id': 'UNKNOWN',
                    'confidence': 0.0,
                    'mean_f0': 0.0
                }
            
            mean_f0 = np.median(valid_f0)
            
            # Classify based on pitch
            if mean_f0 >= 150:
                speaker_id = 'SPEAKER_FEMALE'
                confidence = min(1.0, (mean_f0 - 150) / 200)
            else:
                speaker_id = 'SPEAKER_MALE'
                confidence = min(1.0, (150 - mean_f0) / 75)
            
            return {
                'speaker_id': speaker_id,
                'confidence': float(confidence),
                'mean_f0': float(mean_f0)
            }
            
        except Exception as e:
            print(f"  Error processing {audio_path}: {e}")
            return {
                'speaker_id': 'ERROR',
                'confidence': 0.0,
                'mean_f0': 0.0
            }


class EnergyBasedDiariser:
    """Diarisation based on energy and spectral features"""
    
    def __init__(self):
        self.name = "energy-spectral"
        
    def load(self):
        print(f"✓ Loaded {self.name}")
        return True
    
    def diarise_segment(self, audio_path):
        """Diarise based on energy and spectral features"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Extract features
            # 1. RMS energy
            rms = librosa.feature.rms(y=y)[0]
            mean_rms = np.mean(rms)
            
            # 2. Spectral centroid (brightness)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            mean_centroid = np.mean(spectral_centroid)
            
            # 3. MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mean_mfcc = np.mean(mfccs[0])
            
            # Simple rule-based classification
            # Higher spectral centroid typically indicates female voice
            if mean_centroid > 2000:
                speaker_id = 'SPEAKER_FEMALE'
                confidence = min(1.0, mean_centroid / 3000)
            else:
                speaker_id = 'SPEAKER_MALE'
                confidence = min(1.0, (3000 - mean_centroid) / 3000)
            
            return {
                'speaker_id': speaker_id,
                'confidence': float(confidence),
                'mean_rms': float(mean_rms),
                'mean_centroid': float(mean_centroid),
                'mean_mfcc': float(mean_mfcc)
            }
            
        except Exception as e:
            print(f"  Error processing {audio_path}: {e}")
            return {
                'speaker_id': 'ERROR',
                'confidence': 0.0,
                'mean_rms': 0.0,
                'mean_centroid': 0.0,
                'mean_mfcc': 0.0
            }


def main():
    """Run diarisation on all segmented clips"""
    
    print("=" * 80)
    print("SPEAKER DIARISATION - Testing Multiple Models")
    print("=" * 80)
    print()
    
    # Load segmentation results
    if not SEGMENTATION_RESULTS.exists():
        print(f"ERROR: Segmentation results not found: {SEGMENTATION_RESULTS}")
        print("Please run segmentation first!")
        return
    
    seg_df = pd.read_csv(SEGMENTATION_RESULTS)
    print(f"Loaded {len(seg_df)} segments from {seg_df['model_name'].nunique()} segmentation models")
    print()
    
    # Select which segmentation model to use (use the first one for now)
    # In production, you might want to select the best performing one
    selected_seg_model = seg_df['model_name'].iloc[0]
    print(f"Using segments from: {selected_seg_model}")
    
    seg_df_filtered = seg_df[seg_df['model_name'] == selected_seg_model].copy()
    print(f"Processing {len(seg_df_filtered)} segments")
    print()
    
    # Initialize diarisation models
    models = [
        SimplePitchDiariser(),
        EnergyBasedDiariser(),
    ]
    
    # Try to add pyannote models
    try:
        models.append(PyAnnoteDiariser("pyannote/speaker-diarization-3.1"))
    except:
        print("Note: pyannote/speaker-diarization-3.1 not available")
    
    # Load models
    print("Loading models...")
    loaded_models = []
    for model in models:
        if model.load():
            loaded_models.append(model)
    print()
    
    if not loaded_models:
        print("ERROR: No models loaded successfully")
        return
    
    # Run diarisation
    all_results = []
    
    for model in loaded_models:
        print(f"\n{'=' * 80}")
        print(f"Running {model.name}...")
        print('=' * 80)
        
        start_time = datetime.now()
        
        results_for_model = []
        
        for idx, row in tqdm(seg_df_filtered.iterrows(), total=len(seg_df_filtered), desc=model.name):
            audio_path = Path(row['full_path'])
            
            if not audio_path.exists():
                print(f"  WARNING: File not found: {audio_path}")
                continue
            
            # Run diarisation
            result = model.diarise_segment(audio_path)
            
            # Combine with segment info
            combined = {
                'segment_filename': row['filename'],
                'start_time_ms': row['start_time_ms'],
                'end_time_ms': row['end_time_ms'],
                'duration_ms': row['duration_ms'],
                'segmentation_model': row['model_name'],
                'diarisation_model': model.name,
                'speaker_id': result['speaker_id'],
                'confidence': result['confidence'],
                'audio_path': str(audio_path)
            }
            
            # Add extra features if available
            for key, value in result.items():
                if key not in ['speaker_id', 'confidence']:
                    combined[f'extra_{key}'] = value
            
            results_for_model.append(combined)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Add processing time to all results
        for r in results_for_model:
            r['processing_time_s'] = processing_time
        
        all_results.extend(results_for_model)
        
        # Print summary
        if results_for_model:
            df_temp = pd.DataFrame(results_for_model)
            speaker_counts = df_temp['speaker_id'].value_counts()
            
            print(f"\n✓ Processed {len(results_for_model)} segments")
            print(f"  Processing time: {processing_time:.2f} seconds")
            print(f"  Speaker distribution:")
            for speaker, count in speaker_counts.items():
                pct = (count / len(results_for_model)) * 100
                print(f"    {speaker}: {count} ({pct:.1f}%)")
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        df = df.sort_values(['diarisation_model', 'start_time_ms'])
        df.to_csv(RESULTS_FILE, index=False)
        
        print(f"\n{'=' * 80}")
        print(f"✓ Results saved to {RESULTS_FILE}")
        print(f"  Total results: {len(all_results)}")
        
        # Overall summary
        print("\nOverall Speaker Distribution:")
        for model in df['diarisation_model'].unique():
            print(f"\n  {model}:")
            model_df = df[df['diarisation_model'] == model]
            speaker_counts = model_df['speaker_id'].value_counts()
            for speaker, count in speaker_counts.items():
                pct = (count / len(model_df)) * 100
                print(f"    {speaker}: {count} ({pct:.1f}%)")
    else:
        print("\n⚠ No results generated")
    
    print("\n" + "=" * 80)
    print("DIARISATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
