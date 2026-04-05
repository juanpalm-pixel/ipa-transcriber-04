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
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
SEGMENTATION_DIR = PROJECT_ROOT / "segmentation"
SEGMENTATION_RESULTS = SEGMENTATION_DIR / "segmentation_results.csv"
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"
DIARISED_AUDIO_DIR = OUTPUT_DIR / "by_model"
RESULTS_FILE = BASE_DIR / "diarisation_results.csv"

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
DIARISED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Device configuration - CPU only
device = "cpu"
print(f"Using device: {device} (CPU-only mode)")

# HuggingFace token
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    print("WARNING: HF_TOKEN not found in environment")


def sanitize_label(label):
    """Make speaker/model labels safe for file and folder names."""
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(label))
    return safe.strip("_") or "UNKNOWN"


def build_unique_output_path(model_name, speaker_id, start_time_ms, end_time_ms):
    """Build output path with collision-safe suffixes."""
    safe_model = sanitize_label(model_name)
    safe_speaker = sanitize_label(speaker_id)
    out_dir = DIARISED_AUDIO_DIR / safe_model / safe_speaker
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{int(start_time_ms)}_{int(end_time_ms)}_{safe_speaker}"
    candidate = out_dir / f"{base_name}.wav"
    if not candidate.exists():
        return candidate

    i = 1
    while True:
        alt_candidate = out_dir / f"{base_name}_alt{i}.wav"
        if not alt_candidate.exists():
            return alt_candidate
        i += 1


def save_segment_copy(source_audio_path, model_name, speaker_id, start_time_ms, end_time_ms):
    """Copy a segment into diarisation output structure."""
    y, sr = sf.read(source_audio_path)
    out_path = build_unique_output_path(model_name, speaker_id, start_time_ms, end_time_ms)
    temp_path = out_path.with_name(f"{out_path.stem}.__tmp__.wav")

    sf.write(temp_path, y, sr)
    os.replace(temp_path, out_path)
    return out_path.relative_to(PROJECT_ROOT)


def save_audio_slice(
    source_audio_path,
    model_name,
    speaker_id,
    slice_start_ms,
    slice_end_ms,
    filename_start_ms=None,
    filename_end_ms=None
):
    """Save a time slice from source audio to diarisation output structure."""
    y, sr = sf.read(source_audio_path)
    start_sample = int(max(0, slice_start_ms) * sr / 1000)
    end_sample = int(max(slice_start_ms, slice_end_ms) * sr / 1000)
    end_sample = min(end_sample, len(y))

    if end_sample <= start_sample:
        return None

    clip = y[start_sample:end_sample]
    if filename_start_ms is None:
        filename_start_ms = slice_start_ms
    if filename_end_ms is None:
        filename_end_ms = slice_end_ms

    out_path = build_unique_output_path(model_name, speaker_id, filename_start_ms, filename_end_ms)
    temp_path = out_path.with_name(f"{out_path.stem}.__tmp__.wav")

    sf.write(temp_path, clip, sr)
    os.replace(temp_path, out_path)
    return out_path.relative_to(PROJECT_ROOT)


class PyAnnoteDiariser:
    """Speaker diarisation using pyannote.audio"""
    
    def __init__(self, model_name="pyannote/speaker-diarization-3.1"):
        self.name = model_name.split('/')[-1]
        self.model_name = model_name
        self.pipeline = None
        
    def load(self):
        try:
            from pyannote.audio import Pipeline
            
            # Load pipeline in CPU mode
            try:
                self.pipeline = Pipeline.from_pretrained(
                    self.model_name,
                    token=HF_TOKEN
                )
            except TypeError:
                self.pipeline = Pipeline.from_pretrained(
                    self.model_name,
                    use_auth_token=HF_TOKEN
                )
            
            # Ensure CPU mode
            self.pipeline.to(torch.device("cpu"))
            
            print(f"✓ Loaded {self.name} (CPU mode)")
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
            
            # Get dominant speaker and per-turn speaker segments
            speaker_durations = {}
            speaker_segments = []
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if speaker not in speaker_durations:
                    speaker_durations[speaker] = 0
                duration = max(0.0, float(turn.duration))
                speaker_durations[speaker] += duration

                start_ms = int(max(0.0, float(turn.start)) * 1000)
                end_ms = int(max(float(turn.start), float(turn.end)) * 1000)
                if end_ms <= start_ms:
                    continue

                speaker_segments.append({
                    'speaker_id': speaker,
                    'start_ms': start_ms,
                    'end_ms': end_ms,
                    'duration_ms': end_ms - start_ms
                })
            
            # Get speaker with most time
            if speaker_durations:
                dominant_speaker = max(speaker_durations.items(), key=lambda x: x[1])
                return {
                    'speaker_id': dominant_speaker[0],
                    'confidence': dominant_speaker[1] / sum(speaker_durations.values()),
                    'all_speakers': speaker_durations,
                    'speaker_segments': speaker_segments
                }
            else:
                return {
                    'speaker_id': 'UNKNOWN',
                    'confidence': 0.0,
                    'all_speakers': {},
                    'speaker_segments': []
                }
                
        except Exception as e:
            print(f"  Error processing {audio_path}: {e}")
            return {
                'speaker_id': 'ERROR',
                'confidence': 0.0,
                'all_speakers': {},
                'speaker_segments': []
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
                speaker_id = 'FEMALE'
                confidence = min(1.0, (mean_f0 - 150) / 200)
            else:
                speaker_id = 'MALE'
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
                speaker_id = 'FEMALE'
                confidence = min(1.0, mean_centroid / 3000)
            else:
                speaker_id = 'MALE'
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
            audio_path = Path(row["full_path"])
            if not audio_path.is_absolute():
                audio_path = (SEGMENTATION_DIR / audio_path).resolve()
            
            if not audio_path.exists():
                print(f"  WARNING: File not found: {audio_path}")
                continue
            
            # Run diarisation
            result = model.diarise_segment(audio_path)
            
            segment_start_ms = int(row['start_time_ms'])
            segment_end_ms = int(row['end_time_ms'])
            segment_duration_ms = int(row['duration_ms'])

            # Pyannote can produce multiple speaker turns in the same segmented chunk.
            speaker_segments = result.get('speaker_segments') or []
            if isinstance(model, PyAnnoteDiariser) and speaker_segments:
                for seg in speaker_segments:
                    abs_start_ms = segment_start_ms + int(seg['start_ms'])
                    abs_end_ms = segment_start_ms + int(seg['end_ms'])
                    if abs_end_ms <= abs_start_ms:
                        continue

                    out_rel_path = save_audio_slice(
                        audio_path,
                        model.name,
                        seg['speaker_id'],
                        int(seg['start_ms']),
                        int(seg['end_ms']),
                        abs_start_ms,
                        abs_end_ms
                    )
                    if out_rel_path is None:
                        continue

                    combined = {
                        'segment_filename': row['filename'],
                        'start_time_ms': abs_start_ms,
                        'end_time_ms': abs_end_ms,
                        'duration_ms': abs_end_ms - abs_start_ms,
                        'segmentation_model': row['model_name'],
                        'diarisation_model': model.name,
                        'speaker_id': seg['speaker_id'],
                        'confidence': result['confidence'],
                        'audio_path': str(out_rel_path)
                    }

                    for key, value in result.items():
                        if key not in ['speaker_id', 'confidence', 'speaker_segments']:
                            combined[f'extra_{key}'] = value

                    results_for_model.append(combined)
            else:
                out_rel_path = save_segment_copy(
                    audio_path,
                    model.name,
                    result['speaker_id'],
                    segment_start_ms,
                    segment_end_ms
                )

                combined = {
                    'segment_filename': row['filename'],
                    'start_time_ms': segment_start_ms,
                    'end_time_ms': segment_end_ms,
                    'duration_ms': segment_duration_ms,
                    'segmentation_model': row['model_name'],
                    'diarisation_model': model.name,
                    'speaker_id': result['speaker_id'],
                    'confidence': result['confidence'],
                    'audio_path': str(out_rel_path)
                }

                # Add extra features if available
                for key, value in result.items():
                    if key not in ['speaker_id', 'confidence', 'speaker_segments']:
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
