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
INPUT_DIR = Path("../input")
OUTPUT_DIR = Path("output")
MODELS_DIR = Path("models")
RESULTS_FILE = "segmentation_results.csv"

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


class SileroVADSegmenter:
    """Segmentation using Silero VAD model"""
    
    def __init__(self):
        self.name = "silero-vad"
        self.model = None
        
    def load(self):
        try:
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.get_speech_timestamps = utils[0]
            self.model.to(device)
            print(f"✓ Loaded {self.name}")
            return True
        except Exception as e:
            print(f"✗ Failed to load {self.name}: {e}")
            return False
    
    def segment(self, audio_path, min_speech_duration_ms=300, min_silence_duration_ms=300):
        """Segment audio using VAD"""
        try:
            # Load audio
            wav, sr = librosa.load(audio_path, sr=16000, mono=True)
            wav_tensor = torch.FloatTensor(wav).to(device)
            
            # Get speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                wav_tensor,
                self.model,
                sampling_rate=16000,
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
            self.model = whisper.load_model(self.model_size, device=device)
            print(f"✓ Loaded {self.name}")
            return True
        except Exception as e:
            print(f"✗ Failed to load {self.name}: {e}")
            return False
    
    def segment(self, audio_path):
        """Segment audio using Whisper word timestamps"""
        try:
            import whisper
            
            # Transcribe with word timestamps
            result = self.model.transcribe(
                str(audio_path),
                word_timestamps=True,
                language=None  # Auto-detect
            )
            
            # Load audio for extraction
            wav, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            segments = []
            
            # Extract word-level segments
            for segment in result['segments']:
                if 'words' in segment:
                    for word_info in segment['words']:
                        start_s = word_info['start']
                        end_s = word_info['end']
                        start_ms = int(start_s * 1000)
                        end_ms = int(end_s * 1000)
                        duration_ms = end_ms - start_ms
                        
                        # Extract audio segment
                        start_sample = int(start_s * sr)
                        end_sample = int(end_s * sr)
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
                            'word_text': word_info.get('word', ''),
                            'full_path': str(output_path)
                        })
            
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
    
    def segment(self, audio_path, threshold_db=-40, min_silence_duration=0.3, min_segment_duration=0.2):
        """Segment using energy-based VAD"""
        try:
            # Load audio
            wav, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Convert to dB
            audio_db = librosa.amplitude_to_db(np.abs(wav), ref=np.max)
            
            # Find speech segments
            is_speech = audio_db > threshold_db
            
            # Find boundaries
            boundaries = np.diff(is_speech.astype(int))
            starts = np.where(boundaries == 1)[0]
            ends = np.where(boundaries == -1)[0]
            
            # Align starts and ends
            if len(starts) > 0 and len(ends) > 0:
                if starts[0] > ends[0]:
                    ends = ends[1:]
                if len(starts) > len(ends):
                    starts = starts[:len(ends)]
                if len(ends) > len(starts):
                    ends = ends[:len(starts)]
            
            segments = []
            min_silence_samples = int(min_silence_duration * sr)
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
        segments = model.segment(audio_file)
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
