"""
Segmentation Stage - Audio Word-Level Segmentation
Tests multiple models for segmenting audio into individual words.
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

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"
MODELS_DIR = SCRIPT_DIR / "models"
RESULTS_FILE = SCRIPT_DIR / "segmentation_results.csv"

# Single source of truth for model segmentation controls.
MODEL_SEGMENT_KWARGS = {
    "silero-vad": {
        "threshold": 0.5,  # Higher => more strict (default is 0.5 for Silero VAD)
        "min_silence_duration_ms": 280,
        "min_speech_duration_ms": 300,
    },
    "whisper-base-cancel": {
        "threshold": 0.7, # Higher => more strict (default is 0.6 for no_speech_threshold in Whisper)
        "min_speech_duration_ms": 300,
        "min_silence_duration_ms": 300,
    },
    "simple-vad": {
        "threshold_db": -25, # Closer to 0 => more stric (default is -30 dB for energy-based VAD)
        "min_silence_duration": 0.28,
        "min_segment_duration": 0.3,
    },
}

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Device configuration - CPU only
device = "cpu"
print(f"Using device: {device} (CPU-only mode)")


class SileroVADSegmenter:
    """Segmentation using Silero VAD model"""
    
    def __init__(self):
        self.name = "silero-vad"
        self.model = None
        
    def load(self):
        try:
            # Force CPU mode
            torch.set_num_threads(1)  # Optimize for CPU
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.get_speech_timestamps = utils[0]
            # Ensure CPU mode
            self.model = self.model.cpu()
            print(f"✓ Loaded {self.name} (CPU mode)")
            return True
        except Exception as e:
            print(f"✗ Failed to load {self.name}: {e}")
            return False
    
    def segment(
        self,
        audio_path,
        threshold=None,
        min_speech_duration_ms=None,
        min_silence_duration_ms=None,
    ):
        """Segment audio using VAD"""
        try:
            if threshold is None or min_speech_duration_ms is None or min_silence_duration_ms is None:
                raise ValueError("Missing Silero segmentation parameters. Configure MODEL_SEGMENT_KWARGS['silero-vad'].")

            # Load audio
            wav, sr = librosa.load(audio_path, sr=16000, mono=True)
            wav_tensor = torch.FloatTensor(wav)  # CPU tensor
            
            # Get speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                wav_tensor,
                self.model,
                sampling_rate=16000,
                threshold=threshold,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                return_seconds=False
            )
            
            # Convert to results format
            segments = []
            for i, ts in enumerate(speech_timestamps):
                start_sample = ts['start']
                end_sample = ts['end']
                start_ms = int((start_sample / 16000) * 1000)
                end_ms = int((end_sample / 16000) * 1000)
                duration_ms = end_ms - start_ms
                
                # Extract segment
                segment_wav = wav[start_sample:end_sample]
                
                # Save segment
                filename = f"{start_ms}_{end_ms}.wav"
                output_path = OUTPUT_DIR / self.name / filename
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                sf.write(output_path, segment_wav, 16000)
                
                segments.append({
                    'filename': filename,
                    'start_time_ms': start_ms,
                    'end_time_ms': end_ms,
                    'duration_ms': duration_ms,
                    'model_name': self.name,
                    'full_path': str(output_path)
                })
            
            return segments
            
        except Exception as e:
            print(f"Error in {self.name} segmentation: {e}")
            return []


class WhisperSegmenter:
    """Segmentation using Whisper model with word timestamps"""
    
    def __init__(self, model_size="base"):
        self.name = f"whisper-{model_size}"
        self.model_size = model_size
        self.model = None
        
    def load(self):
        try:
            import whisper
            # Force CPU mode
            self.model = whisper.load_model(self.model_size, device="cpu")
            print(f"✓ Loaded {self.name} (CPU mode)")
            return True
        except Exception as e:
            print(f"✗ Failed to load {self.name}: {e}")
            return False
    
    def segment(
        self,
        audio_path,
        threshold=None,
        min_speech_duration_ms=None,
        min_silence_duration_ms=None,
    ):
        """Segment audio using Whisper word timestamps"""
        try:
            if threshold is None or min_speech_duration_ms is None or min_silence_duration_ms is None:
                raise ValueError("Missing Whisper segmentation parameters. Configure MODEL_SEGMENT_KWARGS['whisper-base'].")

            import whisper
            
            # Transcribe with word timestamps
            result = self.model.transcribe(
                str(audio_path),
                word_timestamps=True,
                language=None,  # Auto-detect
                no_speech_threshold=threshold
            )
            
            # Load audio for extraction
            wav, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            segments = []
            current_words = []
            current_start_ms = None
            current_end_ms = None

            def flush_segment():
                nonlocal current_words, current_start_ms, current_end_ms
                if not current_words or current_start_ms is None or current_end_ms is None:
                    current_words = []
                    current_start_ms = None
                    current_end_ms = None
                    return None

                duration_ms = current_end_ms - current_start_ms
                if duration_ms < min_speech_duration_ms:
                    current_words = []
                    current_start_ms = None
                    current_end_ms = None
                    return None

                start_sample = int((current_start_ms / 1000) * sr)
                end_sample = int((current_end_ms / 1000) * sr)
                segment_wav = wav[start_sample:end_sample]

                filename = f"{current_start_ms}_{current_end_ms}.wav"
                output_path = OUTPUT_DIR / self.name / filename
                output_path.parent.mkdir(parents=True, exist_ok=True)

                sf.write(output_path, segment_wav, 16000)

                segment_record = {
                    'filename': filename,
                    'start_time_ms': current_start_ms,
                    'end_time_ms': current_end_ms,
                    'duration_ms': duration_ms,
                    'model_name': self.name,
                    'word_text': ' '.join(w.get('word', '').strip() for w in current_words).strip(),
                    'full_path': str(output_path)
                }

                current_words = []
                current_start_ms = None
                current_end_ms = None
                return segment_record
            
            # Extract word-level segments
            for segment in result['segments']:
                if 'words' in segment:
                    for word_info in segment['words']:
                        start_s = word_info['start']
                        end_s = word_info['end']
                        start_ms = int(start_s * 1000)
                        end_ms = int(end_s * 1000)
                        if not current_words:
                            current_words = [word_info]
                            current_start_ms = start_ms
                            current_end_ms = end_ms
                            continue

                        gap_ms = start_ms - current_end_ms
                        if gap_ms <= min_silence_duration_ms:
                            current_words.append(word_info)
                            current_end_ms = max(current_end_ms, end_ms)
                            continue

                        flushed = flush_segment()
                        if flushed:
                            segments.append(flushed)

                        current_words = [word_info]
                        current_start_ms = start_ms
                        current_end_ms = end_ms

            flushed = flush_segment()
            if flushed:
                segments.append(flushed)
            
            return segments
            
        except Exception as e:
            print(f"Error in {self.name} segmentation: {e}")
            return []


class SimpleVADSegmenter:
    """Simple energy-based VAD segmentation"""
    
    def __init__(self):
        self.name = "simple-vad"
        
    def load(self):
        print(f"✓ Loaded {self.name}")
        return True
    
    def segment(self, audio_path, threshold_db=None, min_silence_duration=None, min_segment_duration=None):
        """Segment using energy-based VAD"""
        try:
            if threshold_db is None or min_silence_duration is None or min_segment_duration is None:
                raise ValueError("Missing Simple VAD parameters. Configure MODEL_SEGMENT_KWARGS['simple-vad'].")

            # Load audio
            wav, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Convert to dB
            audio_db = librosa.amplitude_to_db(np.abs(wav), ref=np.max)
            
            # Find speech segments
            is_speech = audio_db > threshold_db

            # If no speech is detected at this threshold, return early.
            if not np.any(is_speech):
                return []

            # Close short silent gaps so we require a minimum silence to split segments.
            min_silence_samples = int(min_silence_duration * sr)
            if min_silence_samples > 0:
                speech_idx = np.flatnonzero(is_speech)
                for left_idx, right_idx in zip(speech_idx[:-1], speech_idx[1:]):
                    gap = right_idx - left_idx - 1
                    if 0 < gap < min_silence_samples:
                        is_speech[left_idx + 1:right_idx] = True
            
            # Find boundaries
            boundaries = np.diff(is_speech.astype(int))
            starts = np.where(boundaries == 1)[0] + 1
            ends = np.where(boundaries == -1)[0] + 1

            # Handle speech that starts at sample 0 or extends to the end of the file.
            if is_speech[0]:
                starts = np.insert(starts, 0, 0)
            if is_speech[-1]:
                ends = np.append(ends, len(is_speech))
            
            # Align starts and ends
            if len(starts) > 0 and len(ends) > 0:
                if starts[0] > ends[0]:
                    ends = ends[1:]
                if len(starts) > len(ends):
                    starts = starts[:len(ends)]
                if len(ends) > len(starts):
                    ends = ends[:len(starts)]
            
            segments = []
            min_segment_samples = int(min_segment_duration * sr)
            
            for start_sample, end_sample in zip(starts, ends):
                # Check minimum duration
                if end_sample - start_sample < min_segment_samples:
                    continue
                
                start_ms = int((start_sample / sr) * 1000)
                end_ms = int((end_sample / sr) * 1000)
                duration_ms = end_ms - start_ms
                
                # Extract segment
                segment_wav = wav[start_sample:end_sample]
                
                # Save segment
                filename = f"{start_ms}_{end_ms}.wav"
                output_path = OUTPUT_DIR / self.name / filename
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                sf.write(output_path, segment_wav, 16000)
                
                segments.append({
                    'filename': filename,
                    'start_time_ms': start_ms,
                    'end_time_ms': end_ms,
                    'duration_ms': duration_ms,
                    'model_name': self.name,
                    'full_path': str(output_path)
                })
            
            return segments
            
        except Exception as e:
            print(f"Error in {self.name} segmentation: {e}")
            return []


def main():
    """Run all segmentation models"""
    
    print("=" * 80)
    print("AUDIO SEGMENTATION - Testing Multiple Models")
    print("=" * 80)
    print()
    
    # Find input audio
    input_files = list(INPUT_DIR.glob("*.wav"))
    if not input_files:
        print(f"ERROR: No .wav files found in {INPUT_DIR}")
        return
    
    audio_file = input_files[0]
    print(f"Input audio: {audio_file}")
    print()
    
    # Get audio info
    info = sf.info(audio_file)
    print(f"Sample rate: {info.samplerate} Hz")
    print(f"Duration: {info.duration:.2f} seconds")
    print(f"Channels: {info.channels}")
    print()
    
    # Initialize models
    models = [
        SileroVADSegmenter(),
        WhisperSegmenter("base"),
        SimpleVADSegmenter()
    ]

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
    
    # Run segmentation with each model
    all_results = []
    
    for model in loaded_models:
        print(f"\n{'=' * 80}")
        print(f"Running {model.name}...")
        print('=' * 80)
        
        start_time = datetime.now()
        segment_kwargs = MODEL_SEGMENT_KWARGS.get(model.name)
        if segment_kwargs is None:
            print(f"ERROR: No segmentation config found for model '{model.name}'")
            continue
        segments = model.segment(audio_file, **segment_kwargs)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        print(f"✓ Found {len(segments)} segments")
        print(f"  Processing time: {processing_time:.2f} seconds")
        
        if segments:
            durations = [s['duration_ms'] for s in segments]
            print(f"  Average segment duration: {np.mean(durations):.0f} ms")
            print(f"  Min duration: {np.min(durations):.0f} ms")
            print(f"  Max duration: {np.max(durations):.0f} ms")
            
            # Add metadata
            for seg in segments:
                seg['processing_time_s'] = processing_time
                seg['input_file'] = audio_file.name
        
        all_results.extend(segments)
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        df = df.sort_values(['model_name', 'start_time_ms'])
        df.to_csv(RESULTS_FILE, index=False)
        print(f"\n✓ Results saved to {RESULTS_FILE}")
        print(f"  Total segments across all models: {len(all_results)}")
        
        # Summary by model
        print("\nSummary by model:")
        summary = df.groupby('model_name').agg({
            'filename': 'count',
            'duration_ms': ['mean', 'min', 'max'],
            'processing_time_s': 'first'
        }).round(2)
        print(summary)
    else:
        print("\n⚠ No segments generated")
    
    print("\n" + "=" * 80)
    print("SEGMENTATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
