"""
IPA Transcription Stage - Zero-Shot Phonetic Transcription
Tests multiple models for IPA transcription of segmented audio.
"""

import os
import sys
import warnings
import subprocess
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
from transformers import Wav2Vec2ForCTC, AutoProcessor

# Configuration: project paths and output locations used by this stage.
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
EXTRACTED_SEGMENTS_DIR = PROJECT_ROOT / "segmentation" / "output" / "simple-vad" 
OUTPUT_DIR = PROJECT_ROOT / "ipa" / "output"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_FILE = "ipa_transcriptions.csv"

# Ensure required directories exist before any model/data operations.
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Device configuration: force CPU for reproducibility and low-resource runs.
device = "cpu"
print(f"Using device: {device} (CPU-only mode)")

# Optional Hugging Face token for gated/private model access.
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    print("WARNING: HF_TOKEN not found in environment")


class WhisperIPATranscriber:
    """IPA transcription using Whisper models with IPA adaptation"""
    
    def __init__(self, model_id="neurlang/ipa-whisper-small"):
        self.name = model_id.split('/')[-1]
        self.model_id = model_id
        self.model = None
        self.processor = None
        
    def load(self):
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            
            print(f"Loading {self.name}...")
            self.processor = WhisperProcessor.from_pretrained(self.model_id, token=HF_TOKEN)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_id,
                token=HF_TOKEN
            )
            self.model = self.model.to(device)
            self.model.eval()
            
            print(f"✓ Loaded {self.name} (CPU mode)")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load {self.name}: {e}")
            return False
    
    def transcribe(self, audio_path):
        """Transcribe audio to IPA"""
        try:
            # Load audio
            speech, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Process audio
            input_features = self.processor(
                speech, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(device)
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features)
            
            # Decode
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            return {
                'ipa_transcription': transcription,
                'confidence': 1.0  # Whisper doesn't provide confidence
            }
            
        except Exception as e:
            print(f"  Error transcribing with {self.name}: {e}")
            return {
                'ipa_transcription': '',
                'confidence': 0.0
            }


class AllosaurusTranscriber:
    """IPA transcription using Allosaurus"""
    
    def __init__(self):
        self.name = "allosaurus"
        self.model = None
        
    def load(self):
        try:
            from allosaurus.app import read_recognizer
            
            print(f"Loading {self.name}...")
            self.model = read_recognizer()
            
            print(f"✓ Loaded {self.name} (CPU mode)")
            return True
            
        except ModuleNotFoundError:
            print(f"✗ {self.name} is not installed in this environment; skipping this model.")
            print("  Note: allosaurus installation may fail on Python 3.14 without build tools.")
            return False
        except Exception as e:
            print(f"✗ Failed to load {self.name}: {e}")
            return False
    
    def transcribe(self, audio_path):
        """Transcribe audio to IPA using Allosaurus"""
        try:
            # Allosaurus expects file path
            transcription = self.model.recognize(str(audio_path), lang_id='ipa')
            
            return {
                'ipa_transcription': transcription,
                'confidence': 1.0
            }
            
        except Exception as e:
            print(f"  Error transcribing with {self.name}: {e}")
            return {
                'ipa_transcription': '',
                'confidence': 0.0
            }


class G2PTranscriber:
    """IPA transcription using G2P models"""
    
    def __init__(self, model_id="fdemelo/g2p-multilingual-byt5-tiny-8l-ipa-childes"):
        self.name = model_id.split('/')[-1]
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        
    def load(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            print(f"Loading {self.name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=HF_TOKEN)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id,
                token=HF_TOKEN
            )
            self.model = self.model.to(device)
            self.model.eval()
            
            print(f"✓ Loaded {self.name} (CPU mode)")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load {self.name}: {e}")
            return False
    
    def transcribe(self, audio_path):
        """Transcribe using speech recognition + G2P"""
        try:
            # First, get text transcription using basic Whisper
            import whisper
            whisper_model = whisper.load_model("tiny", device=device)
            result = whisper_model.transcribe(str(audio_path))
            text = result['text']
            
            if not text.strip():
                return {'ipa_transcription': '', 'confidence': 0.0}
            
            # Convert text to IPA using G2P
            inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(device)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=128)
            
            ipa = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                'ipa_transcription': ipa,
                'confidence': 0.8  # Lower confidence for two-step process
            }
            
        except Exception as e:
            print(f"  Error transcribing with {self.name}: {e}")
            return {
                'ipa_transcription': '',
                'confidence': 0.0
            }

class SimpleWhisperIPATranscriber:
    """Simple IPA transcription using base Whisper"""
    
    def __init__(self, model_size="base"):
        self.name = f"whisper-{model_size}-simple"
        self.model_size = model_size
        self.model = None
        
    def load(self):
        try:
            import whisper
            
            print(f"Loading {self.name}...")
            self.model = whisper.load_model(self.model_size, device=device)
            
            print(f"✓ Loaded {self.name} (CPU mode)")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load {self.name}: {e}")
            return False
    
    def transcribe(self, audio_path):
        """Transcribe audio (text, not IPA, but useful for comparison)"""
        try:
            result = self.model.transcribe(str(audio_path), language=None)
            text = result['text']
            
            return {
                'ipa_transcription': text,  # Not IPA, but text transcription
                'confidence': 1.0
            }
            
        except Exception as e:
            print(f"  Error transcribing with {self.name}: {e}")
            return {
                'ipa_transcription': '',
                'confidence': 0.0
            }


def main():
    """Run IPA transcription on extracted audio segments."""
    
    print("=" * 80)
    print("IPA TRANSCRIPTION - Testing Multiple Models (CPU Mode)")
    print("=" * 80)
    print()
    
    # Step 1: Validate that extracted audio segments exist.
    if not EXTRACTED_SEGMENTS_DIR.exists():
        print(f"ERROR: Extracted segments directory not found: {EXTRACTED_SEGMENTS_DIR}")
        print("Please run extract_segments.py first!")
        return
    
    # Step 2: Enumerate all WAV files to be transcribed.
    audio_files = sorted(list(EXTRACTED_SEGMENTS_DIR.glob("*.wav")))
    
    if not audio_files:
        print(f"ERROR: No WAV files found in {EXTRACTED_SEGMENTS_DIR}")
        return
    
    print(f"Found {len(audio_files)} extracted segments")
    print()
    
    # Step 3: Build a uniform table so downstream code can stay model-agnostic.
    # Time/speaker fields are placeholders for this extracted-segment workflow.
    segment_data = []
    for audio_path in audio_files:
        segment_data.append({
            'segment_filename': audio_path.name,
            'audio_path': str(audio_path),
            'start_time_ms': 0,  # Not applicable for extracted segments
            'end_time_ms': 0,
            'duration_ms': 0,
            'speaker_id': 'N/A',
            'diarisation_model': 'N/A'
        })
    
    diar_df_filtered = pd.DataFrame(segment_data)
    print(f"Processing {len(diar_df_filtered)} segments")
    
    # Step 4: Choose which transcription models to run.
    # Keep this list small on CPU to control total runtime.
    models = [
        # SimpleWhisperIPATranscriber("tiny"),  # Fastest Whisper
         AllosaurusTranscriber(),  # Fast and IPA-specific
         WhisperIPATranscriber("neurlang/ipa-whisper-small"),
         G2PTranscriber(), 
    ]
    
   
    # Step 5: Load selected models and keep only the ones that initialize successfully.
    print("Loading models...")
    loaded_models = []
    for model in models:
        if model.load():
            loaded_models.append(model)
    print()
    
    if not loaded_models:
        print("ERROR: No models loaded successfully")
        print("Make sure required packages are installed:")
        return
    
    # Step 6: Run transcription model-by-model over every extracted segment.
    all_results = []
    
    for model in loaded_models:
        print(f"\n{'=' * 80}")
        print(f"Running {model.name}...")
        print('=' * 80)
        
        start_time = datetime.now()
        
        results_for_model = []
        
        for idx, row in tqdm(diar_df_filtered.iterrows(), total=len(diar_df_filtered), desc=model.name):
            # Resolve relative paths safely to absolute paths.
            audio_path = Path(row['audio_path'])
            if not audio_path.is_absolute():
                audio_path = PROJECT_ROOT / audio_path
            
            if not audio_path.exists():
                print(f"  WARNING: File not found: {audio_path}")
                continue
            
            # Run the model-specific transcription routine on this segment.
            result = model.transcribe(audio_path)
            
            # Merge source segment metadata with model output for analysis/comparison.
            combined = {
                'segment_filename': row['segment_filename'],
                'start_time_ms': row['start_time_ms'],
                'end_time_ms': row['end_time_ms'],
                'duration_ms': row['duration_ms'],
                'speaker_id': row['speaker_id'],
                'diarisation_model': row['diarisation_model'],
                'ipa_model': model.name,
                'ipa_transcription': result['ipa_transcription'],
                'confidence': result['confidence'],
                'audio_path': str(audio_path)
            }
            
            results_for_model.append(combined)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Attach end-to-end runtime for this model to each produced row.
        for r in results_for_model:
            r['processing_time_s'] = processing_time
        
        all_results.extend(results_for_model)
        
        # Print quick quality/coverage summary for this model.
        if results_for_model:
            df_temp = pd.DataFrame(results_for_model)
            non_empty = len(df_temp[df_temp['ipa_transcription'].str.len() > 0])
            
            print(f"\n✓ Processed {len(results_for_model)} segments")
            print(f"  Processing time: {processing_time:.2f} seconds")
            print(f"  Non-empty transcriptions: {non_empty} ({(non_empty/len(results_for_model)*100):.1f}%)")
            print(f"  Sample transcriptions:")
            for i, row in df_temp.head(3).iterrows():
                print(f"    {row['segment_filename']}: {row['ipa_transcription'][:50]}...")
    
    # Step 7: Save all collected transcriptions to CSV and print final metrics.
    if all_results:
        df = pd.DataFrame(all_results)
        df = df.sort_values(['ipa_model', 'start_time_ms'])
        results_path = OUTPUT_DIR / RESULTS_FILE
        df.to_csv(results_path, index=False)
        
        print(f"\n{'=' * 80}")
        print(f"✓ Results saved to {results_path}")
        print(f"  Total transcriptions: {len(all_results)}")
        
        # Overall summary
        print("\nTranscription Success Rate by Model:")
        for model in df['ipa_model'].unique():
            model_df = df[df['ipa_model'] == model]
            non_empty = len(model_df[model_df['ipa_transcription'].str.len() > 0])
            pct = (non_empty / len(model_df)) * 100
            print(f"  {model}: {non_empty}/{len(model_df)} ({pct:.1f}%)")
    else:
        print("\n⚠ No transcriptions generated")
    
    print("\n" + "=" * 80)
    print("IPA TRANSCRIPTION COMPLETE")
    print("=" * 80)
    print("\nNOTE: For CPU-only mode, we tested the fastest models.")
    print("To test more models, uncomment additional models in the code.")


if __name__ == "__main__":
    main()
