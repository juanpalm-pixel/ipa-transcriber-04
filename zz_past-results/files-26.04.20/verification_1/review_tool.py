"""
Verification Tool 1 - Interactive Segmentation Review
Allows manual review and adjustment of audio segments
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import json
import tempfile

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
SEGMENTATION_DIR = SCRIPT_DIR.parent / "segmentation"
RESULTS_FILE = SEGMENTATION_DIR / "segmentation_results.csv"
OUTPUT_DIR = SCRIPT_DIR / "reports"
HOTKEYS_FILE = SCRIPT_DIR / "hotkeys.json"
OUTPUT_DIR.mkdir(exist_ok=True)

# The following is another back-up, the script runs the hotkeys.json first, and then this as a backup in case the file is missing or malformed. Users can edit hotkeys.json to customize their shortcuts, and the script will persist any changes they make through the UI. Note it chooses hotkeys.json first over the default
DEFAULT_HOTKEYS = { # where the left is the name of the action, and the right is the default key (can be customized by user)
    "prev_segment": "Left",
    "next_segment": "Right",
    "play_audio": "space",
    "delete_file": "q",
    "focus_trim_start": "e",
    "focus_trim_end": "r",
    "trim_and_save": "t",
    "extract_new_segment": "y",
    "mark_good": "z",
    "mark_too_short": "x", # Note, we dont want to write numbers in the form of "n" in the default hotkeys because Tkinter can be weird about them
    "mark_too_long": "c",
    "mark_bad_quality": "v"
}

HOTKEY_LABELS = {
    "prev_segment": "Previous segment",
    "next_segment": "Next segment",
    "play_audio": "Play audio",
    "delete_file": "Delete file",
    "focus_trim_start": "Focus trim start",
    "focus_trim_end": "Focus trim end",
    "trim_and_save": "Trim and save",
    "extract_new_segment": "Extract new segment",
    "mark_good": "Mark good",
    "mark_too_short": "Mark too short",
    "mark_too_long": "Mark too long",
    "mark_bad_quality": "Mark bad quality"
}

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
        self.trim_start_var = None
        self.trim_end_var = None
        self.trim_start_entry = None
        self.trim_end_entry = None
        self.extract_offsets = {}
        self.hotkeys = self.load_hotkeys()
        self.bound_hotkey_sequences = []
        
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
        ttk.Button(top_frame, text="Hotkeys", command=self.open_hotkeys_editor).pack(side=tk.RIGHT, padx=5)
        
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

        ttk.Separator(adjust_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Label(adjust_frame, text="Trim start ms:").pack(side=tk.LEFT, padx=5)
        self.trim_start_var = tk.StringVar(value="0")
        self.trim_start_entry = ttk.Entry(adjust_frame, textvariable=self.trim_start_var, width=8)
        self.trim_start_entry.pack(side=tk.LEFT, padx=2)
        self.trim_start_entry.bind('<Return>', self.on_trim_entry_return)

        ttk.Label(adjust_frame, text="Trim end ms:").pack(side=tk.LEFT, padx=5)
        self.trim_end_var = tk.StringVar(value="0")
        self.trim_end_entry = ttk.Entry(adjust_frame, textvariable=self.trim_end_var, width=8)
        self.trim_end_entry.pack(side=tk.LEFT, padx=2)
        self.trim_end_entry.bind('<Return>', self.on_trim_entry_return)

        ttk.Button(adjust_frame, text="Trim and Save", command=self.trim_and_save_segment).pack(side=tk.LEFT, padx=5)
        ttk.Button(adjust_frame, text="Extract New Segment", command=self.extract_new_segment).pack(side=tk.LEFT, padx=5)
        ttk.Button(adjust_frame, text="Delete File", command=self.delete_segment_file).pack(side=tk.LEFT, padx=5)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(self.root, text="Statistics", padding="10")
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=4, width=80)
        self.stats_text.pack(fill=tk.X)
        self.bind_hotkeys()
    
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

    def on_trim_entry_return(self, event=None):
        """Leave trim input field when Enter is pressed."""
        self.root.focus_set()
        return "break"

    def focus_trim_start(self):
        if self.trim_start_entry is not None:
            self.trim_start_entry.focus_set()
            self.trim_start_entry.selection_range(0, tk.END)

    def focus_trim_end(self):
        if self.trim_end_entry is not None:
            self.trim_end_entry.focus_set()
            self.trim_end_entry.selection_range(0, tk.END)

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
        """Dispatch hotkey actions while avoiding accidental edits in text fields."""
        focus_widget = self.root.focus_get()
        in_entry = isinstance(focus_widget, (tk.Entry, ttk.Entry))
        allowed_when_editing = {'focus_trim_start', 'focus_trim_end'}

        if in_entry and action not in allowed_when_editing:
            return

        actions = {
            'prev_segment': self.prev_segment,
            'next_segment': self.next_segment,
            'play_audio': self.play_audio,
            'delete_file': self.delete_segment_file,
            'focus_trim_start': self.focus_trim_start,
            'focus_trim_end': self.focus_trim_end,
            'trim_and_save': self.trim_and_save_segment,
            'extract_new_segment': self.extract_new_segment,
            'mark_good': lambda: self.mark_segment('good'),
            'mark_too_short': lambda: self.mark_segment('too_short'),
            'mark_too_long': lambda: self.mark_segment('too_long'),
            'mark_bad_quality': lambda: self.mark_segment('bad_quality')
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
            text="Edit keys (examples: t, Delete, Left, Ctrl+t). Press Enter in trim fields to exit input."
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
    
    def on_model_change(self, event=None):
        """Handle model selection change"""
        self.current_model = self.model_var.get()
        self.current_idx = 0
        self.update_display()
        self.update_statistics()

    def resolve_audio_path(self, full_path):
        """Resolve a CSV audio path against the segmentation folder."""
        audio_path = Path(str(full_path))
        if not audio_path.is_absolute():
            audio_path = (SEGMENTATION_DIR / audio_path).resolve()
        return audio_path

    def save_results(self):
        """Persist the current dataframe back to the segmentation CSV."""
        if self.df is None:
            return

        self.df.to_csv(RESULTS_FILE, index=False)

    def segment_key(self, segment):
        """Stable key used to track repeated extracts from the same source segment."""
        return (
            str(segment['model_name']),
            str(segment['filename']),
            int(segment['start_time_ms']),
            int(segment['end_time_ms'])
        )

    def to_json_safe(self, value):
        """Convert pandas/numpy values to JSON-safe native Python types."""
        if isinstance(value, dict):
            return {str(k): self.to_json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.to_json_safe(v) for v in value]
        if isinstance(value, tuple):
            return [self.to_json_safe(v) for v in value]
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.ndarray):
            return [self.to_json_safe(v) for v in value.tolist()]
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        if pd.isna(value):
            return None
        return value

    def refresh_models(self):
        """Refresh model dropdown after edits."""
        if self.df is None or len(self.df) == 0:
            self.model_combo['values'] = []
            self.current_model = None
            self.current_idx = 0
            self.current_audio = None
            self.current_sr = None
            self.info_text.delete(1.0, tk.END)
            self.stats_text.delete(1.0, tk.END)
            self.progress_label.config(text="0 / 0")
            self.ax.clear()
            self.ax.text(0.5, 0.5, 'No segments available', ha='center', va='center', transform=self.ax.transAxes)
            self.canvas.draw()
            return

        models = self.df['model_name'].unique().tolist()
        self.model_combo['values'] = models

        if self.current_model not in models:
            self.model_var.set(models[0])
            self.current_model = models[0]
            self.current_idx = 0

        if self.current_model is not None:
            self.update_display()
            self.update_statistics()

    def get_active_segment(self):
        """Return the current segment row and its dataframe index."""
        if self.df is None or self.current_model is None:
            return None, None, None

        model_df = self.df[self.df['model_name'] == self.current_model]
        if len(model_df) == 0:
            return None, None, None

        if self.current_idx >= len(model_df):
            self.current_idx = len(model_df) - 1

        row_index = model_df.index[self.current_idx]
        segment = self.df.loc[row_index]
        return segment, model_df, row_index
    
    def update_display(self):
        """Update display for current segment"""
        if self.df is None or self.current_model is None:
            return
        
        # Get segments for current model
        model_df = self.df[self.df['model_name'] == self.current_model]
        
        if len(model_df) == 0:
            return

        if self.current_idx >= len(model_df):
            self.current_idx = len(model_df) - 1
        
        # Get current segment
        segment = model_df.iloc[self.current_idx]

        if self.trim_start_var is not None and self.trim_end_var is not None:
            self.trim_start_var.set("0")
            self.trim_end_var.set(str(int(segment['duration_ms'])))
        
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
            audio_path = self.resolve_audio_path(segment['full_path'])

            if audio_path.exists():
                self.current_audio, self.current_sr = sf.read(audio_path)
                
                # Plot waveform
                self.ax.clear()
                time = np.arange(len(self.current_audio)) / self.current_sr
                self.ax.plot(time, self.current_audio, color='steelblue', linewidth=0.5)
                self.ax.set_xlabel('Time (ms)')
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

    def delete_segment_file(self):
        """Delete the current audio file and remove its CSV row."""
        segment, model_df, row_index = self.get_active_segment()
        if segment is None:
            messagebox.showwarning("Warning", "No segment selected")
            return

        audio_path = self.resolve_audio_path(segment['full_path'])
        if not messagebox.askyesno("Delete File", f"Delete this file and remove it from the CSV?\n\n{audio_path}"):
            return

        try:
            deleted_filename = segment['filename']
            if audio_path.exists():
                audio_path.unlink()

            self.df = self.df.drop(index=row_index)
            self.save_results()
            self.current_audio = None
            self.current_sr = None

            self.corrections.append({
                'model_name': self.current_model,
                'filename': deleted_filename,
                'old_filename': deleted_filename,
                'new_filename': None,
                'label': 'delete',
                'timestamp': datetime.now().isoformat()
            })

            remaining = self.df[self.df['model_name'] == self.current_model]
            if len(remaining) == 0:
                self.refresh_models()
            else:
                if self.current_idx >= len(remaining):
                    self.current_idx = len(remaining) - 1
                self.update_display()
                self.update_statistics()

            messagebox.showinfo("Success", "File deleted and CSV updated")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete segment: {e}")

    def trim_and_save_segment(self):
        """Trim the current audio, save the new file, then replace the CSV row."""
        segment, model_df, row_index = self.get_active_segment()
        if segment is None:
            messagebox.showwarning("Warning", "No segment selected")
            return

        if self.current_audio is None or self.current_sr is None:
            messagebox.showwarning("Warning", "No audio loaded")
            return

        try:
            trim_start_ms = int(float(self.trim_start_var.get()))
            trim_end_ms = int(float(self.trim_end_var.get()))
        except Exception:
            messagebox.showerror("Error", "Trim start and end must be numbers")
            return

        segment_duration_ms = int(segment['duration_ms'])
        if trim_start_ms < 0 or trim_end_ms <= trim_start_ms or trim_end_ms > segment_duration_ms:
            messagebox.showerror(
                "Error",
                f"Trim values must satisfy 0 <= start < end <= {segment_duration_ms} ms"
            )
            return

        start_sample = int((trim_start_ms / 1000) * self.current_sr)
        end_sample = int((trim_end_ms / 1000) * self.current_sr)
        trimmed_audio = self.current_audio[start_sample:end_sample]

        if trimmed_audio.size == 0:
            messagebox.showerror("Error", "Trimmed audio is empty")
            return

        original_path = self.resolve_audio_path(segment['full_path'])
        old_filename = str(segment['filename'])
        new_start_ms = int(segment['start_time_ms']) + trim_start_ms
        new_end_ms = int(segment['start_time_ms']) + trim_end_ms
        new_filename = f"{new_start_ms}_{new_end_ms}.wav"
        final_path = original_path.parent / new_filename
        temp_path = final_path.with_name(f"{final_path.stem}.__tmp__.wav")

        if final_path.exists() and final_path != original_path:
            if not messagebox.askyesno("Overwrite", f"{final_path.name} already exists. Replace it?"):
                return

        try:
            sf.write(temp_path, trimmed_audio, self.current_sr)

            if not temp_path.exists() or temp_path.stat().st_size == 0:
                raise RuntimeError("Trimmed file was not written correctly")

            os.replace(temp_path, final_path)

            if original_path.exists() and original_path != final_path:
                original_path.unlink()

            self.df.loc[row_index, 'filename'] = new_filename
            self.df.loc[row_index, 'start_time_ms'] = new_start_ms
            self.df.loc[row_index, 'end_time_ms'] = new_end_ms
            self.df.loc[row_index, 'duration_ms'] = new_end_ms - new_start_ms
            self.df.loc[row_index, 'full_path'] = str(final_path.relative_to(SEGMENTATION_DIR))

            self.save_results()
            self.corrections.append({
                'model_name': self.current_model,
                'filename': new_filename,
                'old_filename': old_filename,
                'new_filename': new_filename,
                'start_time_ms': new_start_ms,
                'end_time_ms': new_end_ms,
                'label': 'trim',
                'timestamp': datetime.now().isoformat()
            })
            self.current_audio = trimmed_audio
            self.current_sr = self.current_sr
            self.current_idx = self.df[self.df['model_name'] == self.current_model].index.get_loc(row_index)
            self.update_display()
            self.update_statistics()

            messagebox.showinfo("Success", f"Trimmed file saved as:\n{final_path}")

        except Exception as e:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            messagebox.showerror("Error", f"Failed to trim and save segment: {e}")

    def extract_new_segment(self):
        """Create a new trimmed file and keep the original segment for further extraction."""
        segment, model_df, row_index = self.get_active_segment()
        if segment is None:
            messagebox.showwarning("Warning", "No segment selected")
            return

        if self.current_audio is None or self.current_sr is None:
            messagebox.showwarning("Warning", "No audio loaded")
            return

        try:
            trim_start_ms = int(float(self.trim_start_var.get()))
            trim_end_ms = int(float(self.trim_end_var.get()))
        except Exception:
            messagebox.showerror("Error", "Trim start and end must be numbers")
            return

        segment_duration_ms = int(segment['duration_ms'])
        if trim_start_ms < 0 or trim_end_ms <= trim_start_ms or trim_end_ms > segment_duration_ms:
            messagebox.showerror(
                "Error",
                f"Trim values must satisfy 0 <= start < end <= {segment_duration_ms} ms"
            )
            return

        start_sample = int((trim_start_ms / 1000) * self.current_sr)
        end_sample = int((trim_end_ms / 1000) * self.current_sr)
        trimmed_audio = self.current_audio[start_sample:end_sample]

        if trimmed_audio.size == 0:
            messagebox.showerror("Error", "Trimmed audio is empty")
            return

        original_path = self.resolve_audio_path(segment['full_path'])
        old_filename = str(segment['filename'])

        base_start_ms = int(segment['start_time_ms'])
        base_end_ms = int(segment['end_time_ms'])
        new_start_ms = base_start_ms + trim_start_ms
        new_end_ms = base_start_ms + trim_end_ms
        new_filename = f"{new_start_ms}_{new_end_ms}.wav"
        final_path = original_path.parent / new_filename

        if final_path.exists():
            i = 1
            while True:
                candidate = original_path.parent / f"{new_start_ms}_{new_end_ms}_alt{i}.wav"
                if not candidate.exists():
                    final_path = candidate
                    break
                i += 1

        temp_path = final_path.with_name(f"{final_path.stem}.__tmp__.wav")
        source_key = self.segment_key(segment)
        source_abs_pos = self.df.index.get_loc(row_index)
        offset = self.extract_offsets.get(source_key, 0)
        insert_pos = source_abs_pos + 1 + offset

        try:
            sf.write(temp_path, trimmed_audio, self.current_sr)

            if not temp_path.exists() or temp_path.stat().st_size == 0:
                raise RuntimeError("Extracted file was not written correctly")

            os.replace(temp_path, final_path)

            new_row = segment.copy()
            new_row['filename'] = final_path.name
            new_row['start_time_ms'] = new_start_ms
            new_row['end_time_ms'] = new_end_ms
            new_row['duration_ms'] = new_end_ms - new_start_ms
            new_row['full_path'] = str(final_path.relative_to(SEGMENTATION_DIR))

            top = self.df.iloc[:insert_pos]
            bottom = self.df.iloc[insert_pos:]
            self.df = pd.concat([top, pd.DataFrame([new_row]), bottom], ignore_index=True)
            self.extract_offsets[source_key] = offset + 1

            self.save_results()
            self.corrections.append({
                'model_name': self.current_model,
                'filename': final_path.name,
                'old_filename': old_filename,
                'new_filename': final_path.name,
                'start_time_ms': new_start_ms,
                'end_time_ms': new_end_ms,
                'label': 'extract',
                'timestamp': datetime.now().isoformat()
            })

            self.refresh_models()

            model_df_after = self.df[self.df['model_name'] == self.current_model].reset_index(drop=True)
            source_matches = model_df_after[
                (model_df_after['filename'] == old_filename) &
                (model_df_after['start_time_ms'] == base_start_ms) &
                (model_df_after['end_time_ms'] == base_end_ms)
            ]
            if len(source_matches) > 0:
                self.current_idx = int(source_matches.index[0])

            self.update_display()
            self.update_statistics()

            messagebox.showinfo(
                "Success",
                f"New segment created and inserted after source:\n{final_path}\nOriginal segment kept."
            )

        except Exception as e:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            messagebox.showerror("Error", f"Failed to extract new segment: {e}")
    
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
            'filename': str(segment['filename']),
            'start_time_ms': int(segment['start_time_ms']),
            'end_time_ms': int(segment['end_time_ms']),
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

        temp_path = None
        try:
            safe_corrections = self.to_json_safe(self.corrections)

            with tempfile.NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                delete=False,
                dir=OUTPUT_DIR,
                suffix='.tmp'
            ) as tmp_file:
                temp_path = Path(tmp_file.name)
                json.dump(safe_corrections, tmp_file, indent=2)
                tmp_file.flush()

            os.replace(temp_path, output_file)
            messagebox.showinfo("Success", f"Exported {len(self.corrections)} corrections to:\n{output_file}")
        except Exception as e:
            if temp_path is not None and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            messagebox.showerror("Error", f"Failed to export corrections: {e}")
    
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
                    if corr['label'] in ('trim', 'delete', 'extract'):
                        old_name = corr.get('old_filename')
                        new_name = corr.get('new_filename')
                        f.write(f"Old File: {old_name} | New File: {new_name}\n")
                    f.write("\n")
        
        messagebox.showinfo("Success", f"Report saved to:\n{output_file}")


def main():
    """Main function"""
    root = tk.Tk()
    app = SegmentReviewTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
