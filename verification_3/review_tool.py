"""
Verification Tool 3 - IPA Transcription Review
Side-by-side comparison of IPA transcriptions from multiple models
"""

import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import soundfile as sf
import numpy as np
import json
from datetime import datetime

RESULTS_FILE = Path("../ipa/ipa_transcriptions.csv")
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(exist_ok=True)

class IPAReviewTool:
    def __init__(self, root):
        self.root = root
        self.root.title("IPA Transcription Verification Tool")
        self.root.geometry("1600x900")
        
        self.df = None
        self.current_segment_idx = 0
        self.segments = []
        self.corrections = []
        
        self.setup_ui()
        self.load_data()
    
    def setup_ui(self):
        # Top controls
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)
        
        ttk.Button(top_frame, text="<< Previous", command=self.prev_segment).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Play Audio", command=self.play_audio).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Next >>", command=self.next_segment).pack(side=tk.LEFT, padx=5)
        
        self.progress_label = ttk.Label(top_frame, text="0 / 0")
        self.progress_label.pack(side=tk.LEFT, padx=20)
        
        ttk.Button(top_frame, text="Export Corrections", command=self.export_corrections).pack(side=tk.RIGHT, padx=5)
        
        # Segment info
        info_frame = ttk.LabelFrame(self.root, text="Segment Info", padding="10")
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.info_text = tk.Text(info_frame, height=3, width=120)
        self.info_text.pack(fill=tk.X)
        
        # Transcriptions comparison
        comp_frame = ttk.LabelFrame(self.root, text="IPA Transcriptions (All Models)", padding="10")
        comp_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.comparison_text = tk.Text(comp_frame, height=15, width=120, font=("Courier", 11))
        self.comparison_text.pack(fill=tk.BOTH, expand=True)
        
        # Manual correction
        corr_frame = ttk.LabelFrame(self.root, text="Manual Correction", padding="10")
        corr_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(corr_frame, text="Corrected IPA:").pack(side=tk.LEFT, padx=5)
        self.corrected_var = tk.StringVar()
        ttk.Entry(corr_frame, textvariable=self.corrected_var, width=80).pack(side=tk.LEFT, padx=5)
        ttk.Button(corr_frame, text="Save Correction", command=self.save_correction).pack(side=tk.LEFT, padx=5)
    
    def load_data(self):
        try:
            self.df = pd.read_csv(RESULTS_FILE)
            self.segments = self.df['segment_filename'].unique().tolist()
            
            if self.segments:
                self.update_display()
                messagebox.showinfo("Success", f"Loaded {len(self.segments)} segments with IPA transcriptions")
            else:
                messagebox.showwarning("Warning", "No segments found")
                
        except FileNotFoundError:
            messagebox.showerror("Error", f"Results file not found: {RESULTS_FILE}")
            self.root.quit()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")
            self.root.quit()
    
    def update_display(self):
        if not self.segments:
            return
        
        segment = self.segments[self.current_segment_idx]
        segment_df = self.df[self.df['segment_filename'] == segment]
        
        self.progress_label.config(text=f"{self.current_segment_idx + 1} / {len(self.segments)}")
        
        # Update info
        first_row = segment_df.iloc[0]
        self.info_text.delete(1.0, tk.END)
        info = f"Segment: {segment}\nSpeaker: {first_row['speaker_id']} | Time: {first_row['start_time_ms']}-{first_row['end_time_ms']}ms | Duration: {first_row['duration_ms']}ms"
        self.info_text.insert(1.0, info)
        
        # Update comparisons
        self.comparison_text.delete(1.0, tk.END)
        self.comparison_text.insert(1.0, "Model Transcriptions:\n" + "="*80 + "\n\n")
        
        for _, row in segment_df.iterrows():
            model = row['ipa_model']
            transcription = row['ipa_transcription']
            confidence = row['confidence']
            self.comparison_text.insert(tk.END, f"{model:30s} | Conf: {confidence:.2f}\n")
            self.comparison_text.insert(tk.END, f"  {transcription}\n\n")
        
        self.current_audio_path = first_row['audio_path']
        self.corrected_var.set("")
    
    def prev_segment(self):
        if self.current_segment_idx > 0:
            self.current_segment_idx -= 1
            self.update_display()
    
    def next_segment(self):
        if self.current_segment_idx < len(self.segments) - 1:
            self.current_segment_idx += 1
            self.update_display()
    
    def play_audio(self):
        try:
            import pygame
            audio, sr = sf.read(self.current_audio_path)
            pygame.mixer.init(frequency=sr)
            
            import io, wave
            audio_bytes = (audio * 32767).astype(np.int16).tobytes()
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sr)
                wav_file.writeframes(audio_bytes)
            buffer.seek(0)
            pygame.mixer.music.load(buffer)
            pygame.mixer.music.play()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to play audio: {e}")
    
    def save_correction(self):
        corrected = self.corrected_var.get().strip()
        if not corrected:
            messagebox.showwarning("Warning", "Enter corrected IPA transcription")
            return
        
        segment = self.segments[self.current_segment_idx]
        self.corrections.append({
            'segment': segment,
            'corrected_ipa': corrected,
            'timestamp': datetime.now().isoformat()
        })
        
        messagebox.showinfo("Saved", "Correction saved!")
        self.next_segment()
    
    def export_corrections(self):
        if not self.corrections:
            messagebox.showinfo("Info", "No corrections to export")
            return
        
        output_file = OUTPUT_DIR / f"ipa_corrections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(self.corrections, f, indent=2)
        
        messagebox.showinfo("Success", f"Exported {len(self.corrections)} corrections to:\n{output_file}")

def main():
    root = tk.Tk()
    app = IPAReviewTool(root)
    root.mainloop()

if __name__ == "__main__":
    main()
