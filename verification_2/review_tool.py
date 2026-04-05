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
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
RESULTS_FILE = PROJECT_ROOT / "diarisation" / "diarisation_results.csv"
OUTPUT_DIR = BASE_DIR / "reports"
HOTKEYS_FILE = OUTPUT_DIR / "hotkeys.json"
OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_HOTKEYS = {
    "prev_segment": "Left",
    "next_segment": "Right",
    "play_audio": "space",
    "delete_file": "q",
    "female": "z",
    "male_1": "x",
    "male_2": "c"
}

HOTKEY_LABELS = {
    "prev_segment": "Previous segment",
    "next_segment": "Next segment",
    "play_audio": "Play audio",
    "delete_file": "Delete file",
    "female": "Female",
    "male_1": "Male 1",
    "male_2": "Male 2"
}

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
        self.hotkeys = self.load_hotkeys()
        self.bound_hotkey_sequences = []
        
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
        ttk.Button(top_frame, text="Hotkeys", command=self.open_hotkeys_editor).pack(side=tk.RIGHT, padx=5)
        
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
        self.bind_hotkeys()
    
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

    def load_hotkeys(self):
        """Load hotkeys from disk, falling back to defaults."""
        hotkeys = DEFAULT_HOTKEYS.copy()
        if HOTKEYS_FILE.exists():
            try:
                with open(HOTKEYS_FILE, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    for action in DEFAULT_HOTKEYS:
                        val = loaded.get(action)
                        if isinstance(val, str) and val.strip():
                            hotkeys[action] = val.strip()
            except Exception as e:
                print(f"Failed to load hotkeys from {HOTKEYS_FILE}: {e}")

        self.save_hotkeys(hotkeys)
        return hotkeys

    def save_hotkeys(self, hotkeys=None):
        """Persist hotkeys to JSON file."""
        if hotkeys is None:
            hotkeys = self.hotkeys

        with open(HOTKEYS_FILE, 'w', encoding='utf-8') as f:
            json.dump(hotkeys, f, indent=2)

    def key_to_sequence(self, key_text):
        """Convert user key text into a Tk binding sequence."""
        if not isinstance(key_text, str):
            return None

        raw = key_text.strip()
        if not raw:
            return None

        if raw.startswith('<') and raw.endswith('>'):
            return raw

        parts = [p for p in raw.replace(' ', '').split('+') if p]
        if not parts:
            return None

        special = {
            'left': 'Left',
            'right': 'Right',
            'up': 'Up',
            'down': 'Down',
            'space': 'space',
            'delete': 'Delete',
            'del': 'Delete',
            'enter': 'Return',
            'return': 'Return',
            'backspace': 'BackSpace',
            'tab': 'Tab',
            'esc': 'Escape',
            'escape': 'Escape'
        }
        modifier_map = {
            'ctrl': 'Control',
            'control': 'Control',
            'alt': 'Alt',
            'shift': 'Shift'
        }

        if len(parts) == 1:
            key = parts[0]
            return f"<{special.get(key.lower(), key)}>"

        mods = []
        for mod in parts[:-1]:
            mapped = modifier_map.get(mod.lower())
            if mapped and mapped not in mods:
                mods.append(mapped)

        key = special.get(parts[-1].lower(), parts[-1])
        if mods:
            return f"<{ '-'.join(mods) }-{key}>"
        return f"<{key}>"

    def bind_hotkeys(self):
        """Bind all configured hotkeys, replacing old bindings."""
        for seq in self.bound_hotkey_sequences:
            self.root.unbind_all(seq)
        self.bound_hotkey_sequences = []

        for action, key_text in self.hotkeys.items():
            seq = self.key_to_sequence(key_text)
            if not seq:
                continue
            self.root.bind_all(seq, lambda event, a=action: self.handle_hotkey(a, event))
            self.bound_hotkey_sequences.append(seq)

    def handle_hotkey(self, action, event=None):
        """Dispatch hotkey actions for diarisation review."""
        focus_widget = self.root.focus_get()
        if isinstance(focus_widget, (tk.Entry, ttk.Entry)):
            return

        actions = {
            'prev_segment': self.prev_segment,
            'next_segment': self.next_segment,
            'play_audio': self.play_audio,
            'female': lambda: self.quick_assign('FEMALE'),
            'male_1': lambda: self.quick_assign('MALE_1'),
            'male_2': lambda: self.quick_assign('MALE_2')
        }

        action_fn = actions.get(action)
        if action_fn is not None:
            action_fn()

    def open_hotkeys_editor(self):
        """Open a dialog to edit and save keyboard shortcuts."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit Hotkeys")
        dialog.geometry("480x460")
        dialog.transient(self.root)
        dialog.grab_set()

        container = ttk.Frame(dialog, padding="10")
        container.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            container,
            text="Edit keys (examples: t, Delete, Left, Ctrl+t)."
        ).pack(anchor=tk.W, pady=(0, 10))

        rows = ttk.Frame(container)
        rows.pack(fill=tk.BOTH, expand=True)

        editors = {}
        for idx, action in enumerate(DEFAULT_HOTKEYS.keys()):
            ttk.Label(rows, text=HOTKEY_LABELS.get(action, action), width=22).grid(row=idx, column=0, sticky='w', padx=(0, 8), pady=3)
            var = tk.StringVar(value=self.hotkeys.get(action, DEFAULT_HOTKEYS[action]))
            ttk.Entry(rows, textvariable=var, width=24).grid(row=idx, column=1, sticky='w', pady=3)
            editors[action] = var

        def save_and_close():
            updated = {}
            seen = {}

            for action, var in editors.items():
                key_text = var.get().strip()
                seq = self.key_to_sequence(key_text)
                if not key_text or seq is None:
                    messagebox.showerror("Invalid hotkey", f"Invalid key for {HOTKEY_LABELS.get(action, action)}")
                    return

                sig = seq.lower()
                if sig in seen:
                    other = HOTKEY_LABELS.get(seen[sig], seen[sig])
                    current = HOTKEY_LABELS.get(action, action)
                    messagebox.showerror("Duplicate hotkey", f"{current} duplicates {other} ({key_text})")
                    return

                seen[sig] = action
                updated[action] = key_text

            try:
                self.hotkeys = updated
                self.save_hotkeys(self.hotkeys)
                self.bind_hotkeys()
                dialog.destroy()
                messagebox.showinfo("Success", f"Hotkeys saved to:\n{HOTKEYS_FILE}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save hotkeys: {e}")

        button_row = ttk.Frame(container)
        button_row.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(button_row, text="Save", command=save_and_close).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_row, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT)

    def resolve_audio_path(self, csv_path_value):
        """Resolve CSV audio_path to absolute path."""
        audio_path = Path(str(csv_path_value))
        if not audio_path.is_absolute():
            audio_path = PROJECT_ROOT / audio_path
        return audio_path.resolve()

    def to_project_relative(self, absolute_path):
        """Store paths in CSV relative to project root."""
        return str(Path(absolute_path).resolve().relative_to(PROJECT_ROOT))

    def build_reassigned_output_path(self, segment, new_speaker):
        """Build collision-safe target path for reassigned speaker audio."""
        model_name = str(segment['diarisation_model'])
        start_time_ms = int(segment['start_time_ms'])
        end_time_ms = int(segment['end_time_ms'])

        out_dir = PROJECT_ROOT / "diarisation" / "output" / "by_model" / model_name / new_speaker
        out_dir.mkdir(parents=True, exist_ok=True)

        base_name = f"{start_time_ms}_{end_time_ms}_{new_speaker}"
        candidate = out_dir / f"{base_name}.wav"
        if not candidate.exists():
            return candidate

        i = 1
        while True:
            alt_candidate = out_dir / f"{base_name}_alt{i}.wav"
            if not alt_candidate.exists():
                return alt_candidate
            i += 1
    
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
            if not audio_path.is_absolute():
                # Diarisation CSV stores paths relative to the project root.
                audio_path = PROJECT_ROOT / audio_path

            audio_path = audio_path.resolve()
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
        """Reassign current segment to new speaker and persist to CSV/files."""
        new_speaker = self.new_speaker_var.get().strip()
        if not new_speaker:
            messagebox.showwarning("Warning", "Enter new speaker ID")
            return
        
        speaker_df = self.df[
            (self.df['diarisation_model'] == self.current_model) &
            (self.df['speaker_id'] == self.current_speaker)
        ]
        if len(speaker_df) == 0:
            messagebox.showwarning("Warning", "No segment selected")
            return

        if self.current_idx >= len(speaker_df):
            self.current_idx = len(speaker_df) - 1

        segment = speaker_df.iloc[self.current_idx]
        row_index = segment.name

        old_speaker = str(segment['speaker_id'])
        old_audio_rel = str(segment['audio_path'])
        old_audio_path = self.resolve_audio_path(old_audio_rel)
        if not old_audio_path.exists():
            messagebox.showerror("Error", f"Audio file not found:\n{old_audio_path}")
            return

        new_audio_path = self.build_reassigned_output_path(segment, new_speaker)
        temp_path = new_audio_path.with_name(f"{new_audio_path.stem}.__tmp__.wav")

        try:
            y, sr = sf.read(old_audio_path)
            sf.write(temp_path, y, sr)

            if not temp_path.exists() or temp_path.stat().st_size == 0:
                raise RuntimeError("Reassigned file was not written correctly")

            os.replace(temp_path, new_audio_path)

            self.df.at[row_index, 'speaker_id'] = new_speaker
            self.df.at[row_index, 'audio_path'] = self.to_project_relative(new_audio_path)
            self.df.to_csv(RESULTS_FILE, index=False)

            # Delete old file only if no other rows still reference it.
            remaining_refs = (self.df['audio_path'].astype(str) == old_audio_rel).sum()
            if remaining_refs == 0 and old_audio_path.exists() and old_audio_path.resolve() != new_audio_path.resolve():
                old_audio_path.unlink()

        except Exception as e:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            messagebox.showerror("Error", f"Failed to reassign speaker: {e}")
            return
        
        correction = {
            'model_name': self.current_model,
            'segment_filename': segment['segment_filename'],
            'old_speaker': old_speaker,
            'new_speaker': new_speaker,
            'confidence': segment['confidence'],
            'old_audio_path': old_audio_rel,
            'new_audio_path': self.to_project_relative(new_audio_path),
            'timestamp': datetime.now().isoformat()
        }
        
        self.corrections.append(correction)

        # Refresh speaker buckets and stay near the updated row.
        self.on_model_change()
        self.current_speaker = new_speaker
        self.speaker_var.set(new_speaker)
        self.on_speaker_change()

        updated_df = self.df[
            (self.df['diarisation_model'] == self.current_model) &
            (self.df['speaker_id'] == new_speaker)
        ]
        if row_index in updated_df.index:
            self.current_idx = list(updated_df.index).index(row_index)
            self.update_display()

        messagebox.showinfo("Reassigned", f"Segment reassigned: {old_speaker} → {new_speaker}")
    
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
