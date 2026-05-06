"""
Extract audio segments from a WAV file based on timestamps in a TXT file.
"""

import os
import re
import librosa
import soundfile as sf
import pandas as pd
from pathlib import Path

def extract_segments(wav_file, txt_file, output_dir):
    """
    Extract audio segments from a WAV file based on timestamps in a TXT file.
    
    Parameters:
    -----------
    wav_file : str
        Path to the input WAV file
    txt_file : str
        Path to the input TXT file with START, END, TRANSCRIPTION columns
    output_dir : str
        Directory where extracted segments will be saved
    """
    
    def sanitize_filename(filename):
        """Remove or replace invalid filename characters."""
        # Replace invalid characters with underscore
        invalid_chars = r'[<>:"/\\|?*]'
        return re.sub(invalid_chars, '_', filename)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the TXT file
    print(f"Reading timestamps from {txt_file}...")
    df = pd.read_csv(txt_file, sep='\t', skiprows=1, names=['START', 'END', 'TRANSCRIPTION'])
    
    # Load the WAV file
    print(f"Loading audio file {wav_file}...")
    audio, sr = librosa.load(wav_file, sr=None)
    
    print(f"Sample rate: {sr} Hz")
    print(f"Total duration: {len(audio) / sr:.2f} seconds")
    print(f"Number of segments to extract: {len(df)}")
    
    # Extract segments
    extracted_count = 0
    for idx, row in df.iterrows():
        start_time = float(row['START'])
        end_time = float(row['END'])
        transcription = str(row['TRANSCRIPTION']).strip()
        
        # Convert time to sample indices
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Extract segment
        segment = audio[start_sample:end_sample]
        
        # Create output filename with sanitized transcription
        sanitized_transcription = sanitize_filename(transcription)
        output_filename = f"{idx+1:03d}_{sanitized_transcription}.wav"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save segment
        sf.write(output_path, segment, sr)
        
        extracted_count += 1
        
        if (idx + 1) % 50 == 0:
            print(f"  Extracted {idx + 1}/{len(df)} segments...")
    
    print(f"\nSuccessfully extracted {extracted_count} segments to {output_dir}")

if __name__ == "__main__":
    # Define paths
    base_dir = r"c:\Users\pablo\OneDrive\Desktop\Functions\ipa-transcribers\ipa-transcriber(4.0)"
    wav_file = os.path.join(base_dir, "input", "MGM_AFA4_2nd.WAV")
    txt_file = os.path.join(base_dir, "input", "MGM_AFA4_2nd.txt")
    output_dir = os.path.join(base_dir, "output", "extracted_segments")
    
    extract_segments(wav_file, txt_file, output_dir)
