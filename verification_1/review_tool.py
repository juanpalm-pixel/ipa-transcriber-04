"""
Verification Tool 1 - Interactive Segmentation Review
Allows manual review and adjustment of audio segments
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

# Configuration
RESULTS_FILE = Path("../segmentation/segmentation_results.csv")
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(exist_ok=True)

class SegmentReviewTool:
    """Interactive tool for reviewing segmentation results"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Segmentation Verification Tool")
        self.root.geometry("1200x800")
        
        self.df = None
        self.current_model = None
        self.current_idx = 0
        self.current_audio = None
        self.current_sr = None
        self.corrections = []
        
        self.setup_ui()
        self.load_data()
    
    def setup_ui(self):
        """Setup user interface"""
        
        # Top frame - Model selection
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)
        
        ttk.Label(top_frame, text="Model:").pack(side=tk.LEFT, padx=5)
        
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(top_frame, textvariable=self.model_var, width=30)
        self.model_combo.pack(side=tk.LEFT, padx=5)
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_change)
        
        ttk.Button(top_frame, text="Generate Report", command=self.generate_report).pack(side=tk.RIGHT, padx=5)
        ttk.Button(top_frame, text="Export Corrections", command=self.export_corrections).pack(side=tk.RIGHT, padx=5)
        
        # Info frame
        info_frame = ttk.LabelFrame(self.root, text="Segment Info", padding="10")
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.info_text = tk.Text(info_frame, height=4, width=80)
        self.info_text.pack(fill=tk.X)
        
        # Waveform frame
        wave_frame = ttk.LabelFrame(self.root, text="Waveform", padding="10")
        wave_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.fig, self.ax = plt.subplots(figsize=(10, 3))
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
        
        # Adjustment frame
        adjust_frame = ttk.LabelFrame(self.root, text="Adjustments (Advanced)", padding="10")
        adjust_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(adjust_frame, text="Mark as:").pack(side=tk.LEFT, padx=5)
        ttk.Button(adjust_frame, text="Good", command=lambda: self.mark_segment("good")).pack(side=tk.LEFT, padx=2)
        ttk.Button(adjust_frame, text="Too Short", command=lambda: self.mark_segment("too_short")).pack(side=tk.LEFT, padx=2)
        ttk.Button(adjust_frame, text="Too Long", command=lambda: self.mark_segment("too_long")).pack(side=tk.LEFT, padx=2)
        ttk.Button(adjust_frame, text="Bad Quality", command=lambda: self.mark_segment("bad_quality")).pack(side=tk.LEFT, padx=2)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(self.root, text="Statistics", padding="10")
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=4, width=80)
        self.stats_text.pack(fill=tk.X)
    
    def load_data(self):
        """Load segmentation results"""
        try:
            self.df = pd.read_csv(RESULTS_FILE)
            
            # Get unique models
            models = self.df['model_name'].unique().tolist()
            self.model_combo['values'] = models
            
            if models:
                self.model_var.set(models[0])
                self.on_model_change()
            
            messagebox.showinfo("Success", f"Loaded {len(self.df)} segments from {len(models)} models")
            
        except FileNotFoundError:
            messagebox.showerror("Error", f"Results file not found: {RESULTS_FILE}\nRun segmentation first!")
            self.root.quit()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")
            self.root.quit()
    
    def on_model_change(self, event=None):
        """Handle model selection change"""
        self.current_model = self.model_var.get()
        self.current_idx = 0
        self.update_display()
        self.update_statistics()
    
    def update_display(self):
        """Update display for current segment"""
        if self.df is None or self.current_model is None:
            return
        
        # Get segments for current model
        model_df = self.df[self.df['model_name'] == self.current_model]
        
        if len(model_df) == 0:
            return
        
        # Get current segment
        segment = model_df.iloc[self.current_idx]
        
        # Update progress
        self.progress_label.config(text=f"{self.current_idx + 1} / {len(model_df)}")
        
        # Update info
        self.info_text.delete(1.0, tk.END)
        info = f"""Filename: {segment['filename']}
Start: {segment['start_time_ms']}ms | End: {segment['end_time_ms']}ms | Duration: {segment['duration_ms']}ms
Model: {segment['model_name']}
Full Path: {segment['full_path']}"""
        self.info_text.insert(1.0, info)
        
        # Load and display waveform
        try:
            audio_path = Path(segment['full_path'])
            if audio_path.exists():
                self.current_audio, self.current_sr = sf.read(audio_path)
                
                # Plot waveform
                self.ax.clear()
                time = np.arange(len(self.current_audio)) / self.current_sr
                self.ax.plot(time, self.current_audio, color='steelblue', linewidth=0.5)
                self.ax.set_xlabel('Time (s)')
                self.ax.set_ylabel('Amplitude')
                self.ax.set_title(f'Segment Waveform: {segment["filename"]}')
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
        
        model_df = self.df[self.df['model_name'] == self.current_model]
        
        stats = f"""Total segments: {len(model_df)}
Average duration: {model_df['duration_ms'].mean():.0f}ms (std: {model_df['duration_ms'].std():.0f}ms)
Range: {model_df['duration_ms'].min():.0f}ms - {model_df['duration_ms'].max():.0f}ms
Processing time: {model_df['processing_time_s'].iloc[0]:.2f}s"""
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats)
    
    def prev_segment(self):
        """Go to previous segment"""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_display()
    
    def next_segment(self):
        """Go to next segment"""
        model_df = self.df[self.df['model_name'] == self.current_model]
        if self.current_idx < len(model_df) - 1:
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
    
    def mark_segment(self, label):
        """Mark current segment with label"""
        model_df = self.df[self.df['model_name'] == self.current_model]
        segment = model_df.iloc[self.current_idx]
        
        correction = {
            'model_name': self.current_model,
            'filename': segment['filename'],
            'start_time_ms': segment['start_time_ms'],
            'end_time_ms': segment['end_time_ms'],
            'label': label,
            'timestamp': datetime.now().isoformat()
        }
        
        self.corrections.append(correction)
        messagebox.showinfo("Marked", f"Segment marked as: {label}")
        
        # Move to next
        self.next_segment()
    
    def export_corrections(self):
        """Export corrections to file"""
        if not self.corrections:
            messagebox.showinfo("Info", "No corrections to export")
            return
        
        output_file = OUTPUT_DIR / f"corrections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.corrections, f, indent=2)
        
        messagebox.showinfo("Success", f"Exported {len(self.corrections)} corrections to:\n{output_file}")
    
    def generate_report(self):
        """Generate text report"""
        if self.df is None:
            return
        
        output_file = OUTPUT_DIR / f"review_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SEGMENTATION REVIEW REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for model in self.df['model_name'].unique():
                model_df = self.df[self.df['model_name'] == model]
                
                f.write(f"\nModel: {model}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total segments: {len(model_df)}\n")
                f.write(f"Average duration: {model_df['duration_ms'].mean():.0f}ms\n")
                f.write(f"Std duration: {model_df['duration_ms'].std():.0f}ms\n")
                f.write(f"Min duration: {model_df['duration_ms'].min():.0f}ms\n")
                f.write(f"Max duration: {model_df['duration_ms'].max():.0f}ms\n")
                f.write(f"Processing time: {model_df['processing_time_s'].iloc[0]:.2f}s\n")
            
            if self.corrections:
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("MANUAL CORRECTIONS\n")
                f.write("=" * 80 + "\n\n")
                
                for corr in self.corrections:
                    f.write(f"File: {corr['filename']} | Label: {corr['label']} | Time: {corr['timestamp']}\n")
        
        messagebox.showinfo("Success", f"Report saved to:\n{output_file}")


def main():
    """Main function"""
    root = tk.Tk()
    app = SegmentReviewTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
