"""
Verification Tool 2 - Interactive Diarisation Review
Allows manual review and adjustment of speaker labels
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import json
from collections import defaultdict

# Configuration
RESULTS_FILE = Path("../diarisation/diarisation_results.csv")
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(exist_ok=True)

class DiarisationReviewTool:
    """Interactive tool for reviewing diarisation results"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Diarisation Verification Tool")
        self.root.geometry("1400x900")
        
        self.df = None
        self.current_model = None
        self.current_speaker = None
        self.current_idx = 0
        self.current_audio = None
        self.current_sr = None
        self.corrections = []
        
        self.setup_ui()
        self.load_data()
    
    def setup_ui(self):
        """Setup user interface"""
        
        # Top frame - Model and Speaker selection
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)
        
        ttk.Label(top_frame, text="Model:").pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(top_frame, textvariable=self.model_var, width=25)
        self.model_combo.pack(side=tk.LEFT, padx=5)
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_change)
        
        ttk.Label(top_frame, text="Speaker:").pack(side=tk.LEFT, padx=15)
        self.speaker_var = tk.StringVar()
        self.speaker_combo = ttk.Combobox(top_frame, textvariable=self.speaker_var, width=20)
        self.speaker_combo.pack(side=tk.LEFT, padx=5)
        self.speaker_combo.bind('<<ComboboxSelected>>', self.on_speaker_change)
        
        ttk.Button(top_frame, text="Generate Report", command=self.generate_report).pack(side=tk.RIGHT, padx=5)
        ttk.Button(top_frame, text="Export Corrections", command=self.export_corrections).pack(side=tk.RIGHT, padx=5)
        
        # Info frame
        info_frame = ttk.LabelFrame(self.root, text="Segment Info", padding="10")
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.info_text = tk.Text(info_frame, height=4, width=100)
        self.info_text.pack(fill=tk.X)
        
        # Waveform frame
        wave_frame = ttk.LabelFrame(self.root, text="Waveform", padding="10")
        wave_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.fig, self.ax = plt.subplots(figsize=(12, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=wave_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Control frame
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)
        
        ttk.Button(control_frame, text="<< Previous", command=self.prev_segment).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Play Audio", command=self.play_audio).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Next >>", command=self.next_segment).pack(side=tk.LEFT, padx=5)
        
        self.progress_label = ttk.Label(control_frame, text="0 / 0")
        self.progress_label.pack(side=tk.LEFT, padx=20)
        
        # Speaker reassignment frame
        reassign_frame = ttk.LabelFrame(self.root, text="Speaker Reassignment", padding="10")
        reassign_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(reassign_frame, text="Reassign to:").pack(side=tk.LEFT, padx=5)
        
        self.new_speaker_var = tk.StringVar()
        ttk.Entry(reassign_frame, textvariable=self.new_speaker_var, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(reassign_frame, text="Reassign", command=self.reassign_speaker).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(reassign_frame, text="Quick assign:").pack(side=tk.LEFT, padx=15)
        ttk.Button(reassign_frame, text="Female", command=lambda: self.quick_assign("FEMALE")).pack(side=tk.LEFT, padx=2)
        ttk.Button(reassign_frame, text="Male 1", command=lambda: self.quick_assign("MALE_1")).pack(side=tk.LEFT, padx=2)
        ttk.Button(reassign_frame, text="Male 2", command=lambda: self.quick_assign("MALE_2")).pack(side=tk.LEFT, padx=2)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(self.root, text="Speaker Statistics", padding="10")
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=6, width=100)
        self.stats_text.pack(fill=tk.X)
    
    def load_data(self):
        """Load diarisation results"""
        try:
            self.df = pd.read_csv(RESULTS_FILE)
            
            # Get unique models
            models = self.df['diarisation_model'].unique().tolist()
            self.model_combo['values'] = models
            
            if models:
                self.model_var.set(models[0])
                self.on_model_change()
            
            messagebox.showinfo("Success", f"Loaded {len(self.df)} diarised segments from {len(models)} models")
            
        except FileNotFoundError:
            messagebox.showerror("Error", f"Results file not found: {RESULTS_FILE}\nRun diarisation first!")
            self.root.quit()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")
            self.root.quit()
    
    def on_model_change(self, event=None):
        """Handle model selection change"""
        self.current_model = self.model_var.get()
        
        # Update speaker list
        model_df = self.df[self.df['diarisation_model'] == self.current_model]
        speakers = model_df['speaker_id'].unique().tolist()
        self.speaker_combo['values'] = speakers
        
        if speakers:
            self.speaker_var.set(speakers[0])
            self.on_speaker_change()
        
        self.update_statistics()
    
    def on_speaker_change(self, event=None):
        """Handle speaker selection change"""
        self.current_speaker = self.speaker_var.get()
        self.current_idx = 0
        self.update_display()
    
    def update_display(self):
        """Update display for current segment"""
        if self.df is None or self.current_model is None or self.current_speaker is None:
            return
        
        # Get segments for current model and speaker
        speaker_df = self.df[
            (self.df['diarisation_model'] == self.current_model) &
            (self.df['speaker_id'] == self.current_speaker)
        ]
        
        if len(speaker_df) == 0:
            return
        
        # Get current segment
        segment = speaker_df.iloc[self.current_idx]
        
        # Update progress
        self.progress_label.config(text=f"{self.current_idx + 1} / {len(speaker_df)}")
        
        # Update info
        self.info_text.delete(1.0, tk.END)
        info = f"""Filename: {segment['segment_filename']}
Time: {segment['start_time_ms']}ms - {segment['end_time_ms']}ms | Duration: {segment['duration_ms']}ms
Speaker: {segment['speaker_id']} | Confidence: {segment['confidence']:.3f}
Model: {segment['diarisation_model']}"""
        self.info_text.insert(1.0, info)
        
        # Load and display waveform
        try:
            audio_path = Path(segment['audio_path'])
            if audio_path.exists():
                self.current_audio, self.current_sr = sf.read(audio_path)
                
                # Plot waveform
                self.ax.clear()
                time = np.arange(len(self.current_audio)) / self.current_sr
                self.ax.plot(time, self.current_audio, color='steelblue', linewidth=0.5)
                self.ax.set_xlabel('Time (s)')
                self.ax.set_ylabel('Amplitude')
                self.ax.set_title(f'Segment: {segment["segment_filename"]} | Speaker: {segment["speaker_id"]}')
                self.ax.grid(True, alpha=0.3)
                self.canvas.draw()
            else:
                self.ax.clear()
                self.ax.text(0.5, 0.5, 'Audio file not found', 
                           ha='center', va='center', transform=self.ax.transAxes)
                self.canvas.draw()
                
        except Exception as e:
            print(f"Error loading audio: {e}")
            self.ax.clear()
            self.ax.text(0.5, 0.5, f'Error: {e}', 
                       ha='center', va='center', transform=self.ax.transAxes)
            self.canvas.draw()
    
    def update_statistics(self):
        """Update statistics display"""
        if self.df is None or self.current_model is None:
            return
        
        model_df = self.df[self.df['diarisation_model'] == self.current_model]
        
        # Speaker distribution
        speaker_counts = model_df['speaker_id'].value_counts()
        
        stats = f"""Model: {self.current_model}
Total segments: {len(model_df)}
Speaker distribution:\n"""
        
        for speaker, count in speaker_counts.items():
            pct = (count / len(model_df)) * 100
            avg_conf = model_df[model_df['speaker_id'] == speaker]['confidence'].mean()
            stats += f"  {speaker}: {count} segments ({pct:.1f}%) | Avg confidence: {avg_conf:.3f}\n"
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats)
    
    def prev_segment(self):
        """Go to previous segment"""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_display()
    
    def next_segment(self):
        """Go to next segment"""
        speaker_df = self.df[
            (self.df['diarisation_model'] == self.current_model) &
            (self.df['speaker_id'] == self.current_speaker)
        ]
        if self.current_idx < len(speaker_df) - 1:
            self.current_idx += 1
            self.update_display()
    
    def play_audio(self):
        """Play current audio segment"""
        if self.current_audio is None:
            messagebox.showwarning("Warning", "No audio loaded")
            return
        
        try:
            import pygame
            pygame.mixer.init(frequency=self.current_sr)
            
            # Create temporary wav in memory
            import io
            import wave
            
            # Convert to bytes
            audio_bytes = (self.current_audio * 32767).astype(np.int16).tobytes()
            
            # Create wav buffer
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.current_sr)
                wav_file.writeframes(audio_bytes)
            
            buffer.seek(0)
            pygame.mixer.music.load(buffer)
            pygame.mixer.music.play()
            
        except ImportError:
            messagebox.showerror("Error", "pygame not installed. Install with: pip install pygame")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to play audio: {e}")
    
    def reassign_speaker(self):
        """Reassign current segment to new speaker"""
        new_speaker = self.new_speaker_var.get().strip()
        if not new_speaker:
            messagebox.showwarning("Warning", "Enter new speaker ID")
            return
        
        speaker_df = self.df[
            (self.df['diarisation_model'] == self.current_model) &
            (self.df['speaker_id'] == self.current_speaker)
        ]
        segment = speaker_df.iloc[self.current_idx]
        
        correction = {
            'model_name': self.current_model,
            'segment_filename': segment['segment_filename'],
            'old_speaker': segment['speaker_id'],
            'new_speaker': new_speaker,
            'confidence': segment['confidence'],
            'timestamp': datetime.now().isoformat()
        }
        
        self.corrections.append(correction)
        messagebox.showinfo("Reassigned", f"Segment reassigned: {segment['speaker_id']} → {new_speaker}")
        
        # Move to next
        self.next_segment()
    
    def quick_assign(self, speaker_label):
        """Quick reassignment to predefined speaker"""
        self.new_speaker_var.set(speaker_label)
        self.reassign_speaker()
    
    def export_corrections(self):
        """Export corrections to file"""
        if not self.corrections:
            messagebox.showinfo("Info", "No corrections to export")
            return
        
        # Export as JSON
        json_file = OUTPUT_DIR / f"speaker_corrections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w') as f:
            json.dump(self.corrections, f, indent=2)
        
        # Export as CSV
        csv_file = OUTPUT_DIR / f"speaker_corrections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        pd.DataFrame(self.corrections).to_csv(csv_file, index=False)
        
        messagebox.showinfo("Success", f"Exported {len(self.corrections)} corrections to:\n{json_file}\n{csv_file}")
    
    def generate_report(self):
        """Generate text report"""
        if self.df is None:
            return
        
        output_file = OUTPUT_DIR / f"diarisation_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DIARISATION REVIEW REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for model in self.df['diarisation_model'].unique():
                model_df = self.df[self.df['diarisation_model'] == model]
                
                f.write(f"\nModel: {model}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total segments: {len(model_df)}\n\n")
                
                f.write("Speaker Distribution:\n")
                speaker_counts = model_df['speaker_id'].value_counts()
                for speaker, count in speaker_counts.items():
                    pct = (count / len(model_df)) * 100
                    avg_conf = model_df[model_df['speaker_id'] == speaker]['confidence'].mean()
                    f.write(f"  {speaker}: {count} ({pct:.1f}%) | Avg confidence: {avg_conf:.3f}\n")
            
            if self.corrections:
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("MANUAL CORRECTIONS\n")
                f.write("=" * 80 + "\n\n")
                
                for corr in self.corrections:
                    f.write(f"Segment: {corr['segment_filename']}\n")
                    f.write(f"  {corr['old_speaker']} → {corr['new_speaker']}\n")
                    f.write(f"  Time: {corr['timestamp']}\n\n")
        
        messagebox.showinfo("Success", f"Report saved to:\n{output_file}")


def main():
    """Main function"""
    root = tk.Tk()
    app = DiarisationReviewTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
