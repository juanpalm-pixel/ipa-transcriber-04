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

    def read_timestamp_rows(txt_path):
        """Parse START, END, TRANSCRIPTION rows from the mixed-delimiter text file."""
        rows = []
        with open(txt_path, 'r', encoding='utf-8-sig') as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                if line_number == 1 and line.upper().startswith('START'):
                    continue

                parts = line.split(maxsplit=2)
                if len(parts) < 3:
                    raise ValueError(
                        f"Could not parse timestamp row {line_number} in {txt_path}: {raw_line.rstrip()}"
                    )

                start_text, end_text, transcription = parts
                rows.append({
                    'START': float(start_text),
                    'END': float(end_text),
                    'TRANSCRIPTION': transcription,
                })

        return rows

    def write_textgrid(txt_path, rows, duration_seconds, textgrid_path):
        """Write the timestamps to a Praat TextGrid with one interval tier."""
        intervals = []
        current_time = 0.0

        for row in rows:
            start_time = float(row['START'])
            end_time = float(row['END'])
            transcription = str(row['TRANSCRIPTION']).strip()

            if start_time > current_time:
                intervals.append((current_time, start_time, ''))

            intervals.append((start_time, end_time, transcription))
            current_time = end_time

        if current_time < duration_seconds:
            intervals.append((current_time, duration_seconds, ''))

        with open(textgrid_path, 'w', encoding='utf-8') as handle:
            handle.write('File type = "ooTextFile"\n')
            handle.write('Object class = "TextGrid"\n\n')
            handle.write(f'{0.0:.6f} {duration_seconds:.6f}\n')
            handle.write('<exists>\n')
            handle.write('1\n')
            handle.write('"IntervalTier" "transcription" ')
            handle.write(f'{0.0:.6f} {duration_seconds:.6f}\n')
            handle.write(f'{len(intervals)}\n')

            for interval_start, interval_end, label in intervals:
                escaped_label = label.replace('"', '""')
                handle.write(f'{interval_start:.6f}\n')
                handle.write(f'{interval_end:.6f}\n')
                handle.write(f'"{escaped_label}"\n')

        print(f"Wrote TextGrid to {textgrid_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Remove any previously extracted clips so reruns produce a clean segment set.
    for existing_file in Path(output_dir).glob('*.wav'):
        existing_file.unlink()
    
    # Read the TXT file
    print(f"Reading timestamps from {txt_file}...")
    rows = read_timestamp_rows(txt_file)
    
    # Load the WAV file
    print(f"Loading audio file {wav_file}...")
    audio, sr = librosa.load(wav_file, sr=None)
    audio_duration = len(audio) / sr
    
    print(f"Sample rate: {sr} Hz")
    print(f"Total duration: {audio_duration:.2f} seconds")
    print(f"Number of segments to extract: {len(rows)}")

    textgrid_dir = Path(output_dir).parent / "textgrids"
    textgrid_dir.mkdir(parents=True, exist_ok=True)
    textgrid_path = textgrid_dir / f"{Path(txt_file).stem}.TextGrid"
    write_textgrid(txt_file, rows, audio_duration, textgrid_path)
    
    # Extract segments
    extracted_count = 0
    records = []
    for idx, row in enumerate(rows):
        start_time = row['START']
        end_time = row['END']
        transcription = str(row['TRANSCRIPTION']).strip()

        if end_time <= start_time:
            raise ValueError(
                f"Invalid timestamp range at row {idx + 1}: start={start_time}, end={end_time}"
            )
        
        # Convert time to sample indices
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        start_sample = max(0, min(start_sample, len(audio)))
        end_sample = max(0, min(end_sample, len(audio)))
        
        # Extract segment
        segment = audio[start_sample:end_sample]
        
        # Create output filename with sanitized transcription
        sanitized_transcription = sanitize_filename(transcription)
        output_filename = f"{idx+1:03d}_{sanitized_transcription}.wav"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save segment
        sf.write(output_path, segment, sr)
        duration_seconds = end_time - start_time
        records.append({
            'index': idx + 1,
            'filename': output_filename,
            'start_time': start_time,
            'end_time': end_time,
            'duration_seconds': duration_seconds,
            'full_path': str(output_path)
        })
        
        extracted_count += 1
        
        if (idx + 1) % 50 == 0:
            print(f"  Extracted {idx + 1}/{len(rows)} segments...")
    
    print(f"\nSuccessfully extracted {extracted_count} segments to {output_dir}")

    # Write manifest CSV with start/end times for all segments
    manifest_path = Path(output_dir).parent / f"{Path(txt_file).stem}_segments.csv"
    df = pd.DataFrame.from_records(records)
    df.to_csv(manifest_path, index=False, encoding='utf-8-sig')
    print(f"Wrote segments manifest to {manifest_path}")

if __name__ == "__main__":
    # Define paths
    base_dir = r"c:\Users\pablo\OneDrive\Desktop\Functions\ipa-transcribers\ipa-transcriber(4.0)"
    wav_file = os.path.join(base_dir, "input", "MGM_AFA4_2nd.WAV")
    txt_file = os.path.join(base_dir, "input", "MGM_AFA4_2nd.txt")
    output_dir = os.path.join(base_dir, "output", "extracted_segments")
    
    extract_segments(wav_file, txt_file, output_dir)
