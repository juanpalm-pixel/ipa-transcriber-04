"""
Verification Tool 4 - Tone Review
Visualize pitch contours and review tone classifications
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import soundfile as sf
import numpy as np
import json
from datetime import datetime

RESULTS_FILE = Path("../tone-correction/tone_analysis.csv")
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(exist_ok=True)

class ToneReviewTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Tone Verification Tool")
        self.root.geometry("1400x900")
        
        self.df = None
        self.current_idx = 0
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
        
        ttk.Button(top_frame, text="Generate Report", command=self.generate_report).pack(side=tk.RIGHT, padx=5)
        
        # Info frame
        info_frame = ttk.LabelFrame(self.root, text="Segment Info", padding="10")
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.info_text = tk.Text(info_frame, height=3, width=120)
        self.info_text.pack(fill=tk.X)
        
        # Pitch contour plot
        plot_frame = ttk.LabelFrame(self.root, text="Pitch Contour", padding="10")
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.fig, self.ax = plt.subplots(figsize=(12, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tone stats
        stats_frame = ttk.LabelFrame(self.root, text="Tone Statistics", padding="10")
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=6, width=120)
        self.stats_text.pack(fill=tk.X)
    
    def load_data(self):
        try:
            self.df = pd.read_csv(RESULTS_FILE)
            
            if len(self.df) > 0:
                self.update_display()
                self.update_statistics()
                messagebox.showinfo("Success", f"Loaded {len(self.df)} tone analyses")
            else:
                messagebox.showwarning("Warning", "No tone data found")
                
        except FileNotFoundError:
            messagebox.showerror("Error", f"Results file not found: {RESULTS_FILE}")
            self.root.quit()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")
            self.root.quit()
    
    def update_display(self):
        if self.df is None or len(self.df) == 0:
            return
        
        row = self.df.iloc[self.current_idx]
        
        self.progress_label.config(text=f"{self.current_idx + 1} / {len(self.df)}")
        
        # Update info
        self.info_text.delete(1.0, tk.END)
        info = f"""Segment: {row['segment_filename']} | Speaker: {row['speaker_id']}
IPA: {row['ipa_transcription']} | Tone: {row['tone_category']}
F0: Mean={row['mean_f0']:.1f}Hz, Std={row.get('f0_std', 0):.1f}Hz, Range={row.get('f0_range', 0):.1f}Hz"""
        self.info_text.insert(1.0, info)
        
        # Plot pitch contour
        self.ax.clear()
        
        # Note: pitch_contour might be stored as string, need to parse
        try:
            import ast
            if isinstance(row.get('pitch_contour'), str):
                # Pitch contour not in CSV, skip plotting
                self.ax.text(0.5, 0.5, f"Tone: {row['tone_category']}\nMean F0: {row['mean_f0']:.1f} Hz",
                           ha='center', va='center', fontsize=14, transform=self.ax.transAxes)
            else:
                self.ax.text(0.5, 0.5, f"Tone: {row['tone_category']}\nMean F0: {row['mean_f0']:.1f} Hz",
                           ha='center', va='center', fontsize=14, transform=self.ax.transAxes)
        except:
            self.ax.text(0.5, 0.5, f"Tone: {row['tone_category']}\nMean F0: {row['mean_f0']:.1f} Hz",
                       ha='center', va='center', fontsize=14, transform=self.ax.transAxes)
        
        self.ax.set_title(f"Pitch Analysis - {row['segment_filename']}")
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('F0 (Hz)')
        self.ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
        
        self.current_audio_path = row['audio_path']
    
    def update_statistics(self):
        if self.df is None:
            return
        
        stats = "Overall Tone Distribution:\n"
        tone_counts = self.df['tone_category'].value_counts()
        
        for tone, count in tone_counts.items():
            pct = (count / len(self.df)) * 100
            stats += f"  {tone}: {count} ({pct:.1f}%)\n"
        
        stats += f"\nAverage F0: {self.df['mean_f0'].mean():.1f} Hz\n"
        stats += f"F0 Range: {self.df['mean_f0'].min():.1f} - {self.df['mean_f0'].max():.1f} Hz"
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats)
    
    def prev_segment(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_display()
    
    def next_segment(self):
        if self.current_idx < len(self.df) - 1:
            self.current_idx += 1
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
    
    def generate_report(self):
        if self.df is None:
            return
        
        output_file = OUTPUT_DIR / f"tone_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TONE ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total segments: {len(self.df)}\n\n")
            
            f.write("Tone Distribution:\n")
            tone_counts = self.df['tone_category'].value_counts()
            for tone, count in tone_counts.items():
                pct = (count / len(self.df)) * 100
                f.write(f"  {tone}: {count} ({pct:.1f}%)\n")
            
            f.write(f"\nPitch Statistics:\n")
            f.write(f"  Mean F0: {self.df['mean_f0'].mean():.1f} Hz\n")
            f.write(f"  F0 Range: {self.df['mean_f0'].min():.1f} - {self.df['mean_f0'].max():.1f} Hz\n")
            f.write(f"  Std F0: {self.df['mean_f0'].std():.1f} Hz\n")
        
        messagebox.showinfo("Success", f"Report saved to:\n{output_file}")

def main():
    root = tk.Tk()
    app = ToneReviewTool(root)
    root.mainloop()

if __name__ == "__main__":
    main()
