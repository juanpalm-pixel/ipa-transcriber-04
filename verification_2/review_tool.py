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
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_FILE = PROJECT_ROOT / "diarisation" / "diarisation_results.csv"
OUTPUT_DIR = SCRIPT_DIR / "reports"
HOTKEYS_FILE = SCRIPT_DIR / "hotkeys.json"
OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_HOTKEYS = {
    "prev_segment": "Left",
    "next_segment": "Right",
    "play_audio": "space",
    "delete_file": "q",
    "focus_trim_start": "e",
    "focus_trim_end": "r",
    "trim_and_save": "t",
    "extract_new_segment": "y",
    "female": "z",
    "male": "x",
    "male_1": "c",
    "male_2": "v"
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
    "female": "Female",
    "male": "Male",
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
        self.current_segmentation_model = None
        self.current_model = None
        self.current_speaker = None
        self.current_idx = 0
        self.current_audio = None
        self.current_sr = None
        self.corrections = []
        self.trim_start_var = None
        self.trim_end_var = None
        self.trim_start_entry = None
        self.trim_end_entry = None
        self.hotkeys = self.load_hotkeys()
        self.bound_hotkey_sequences = []
        
        self.setup_ui()
        self.load_data()
    
    def setup_ui(self):
        """Setup user interface"""
        
        # Top frame - Model and Speaker selection
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)
        
        ttk.Label(top_frame, text="Segmentation model:").pack(side=tk.LEFT, padx=5)

        self.segmentation_model_var = tk.StringVar()
        self.segmentation_model_combo = ttk.Combobox(top_frame, textvariable=self.segmentation_model_var, width=25)
        self.segmentation_model_combo.pack(side=tk.LEFT, padx=5)
        self.segmentation_model_combo.bind('<<ComboboxSelected>>', self.on_segmentation_model_change)

        ttk.Label(top_frame, text="Diarisation model:").pack(side=tk.LEFT, padx=15)

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
        self.info_text.configure(state=tk.DISABLED)
        self.info_text.bind('<Escape>', self.release_focus)
        
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
        ttk.Button(control_frame, text="Delete File", command=self.delete_segment_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Next >>", command=self.next_segment).pack(side=tk.LEFT, padx=5)
        
        self.progress_label = ttk.Label(control_frame, text="0 / 0")
        self.progress_label.pack(side=tk.LEFT, padx=20)
        
        # Speaker reassignment frame
        reassign_frame = ttk.LabelFrame(self.root, text="Speaker Reassignment", padding="10")
        reassign_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(reassign_frame, text="Reassign to:").pack(side=tk.LEFT, padx=5)
        
        self.new_speaker_var = tk.StringVar()
        self.new_speaker_entry = ttk.Entry(reassign_frame, textvariable=self.new_speaker_var, width=20)
        self.new_speaker_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(reassign_frame, text="Reassign", command=self.reassign_speaker).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(reassign_frame, text="Quick assign:").pack(side=tk.LEFT, padx=15)
        ttk.Button(reassign_frame, text="Female", command=lambda: self.quick_assign("FEMALE")).pack(side=tk.LEFT, padx=2)
        ttk.Button(reassign_frame, text="Male", command=lambda: self.quick_assign("MALE")).pack(side=tk.LEFT, padx=2)
        ttk.Button(reassign_frame, text="Male 1", command=lambda: self.quick_assign("MALE_1")).pack(side=tk.LEFT, padx=2)
        ttk.Button(reassign_frame, text="Male 2", command=lambda: self.quick_assign("MALE_2")).pack(side=tk.LEFT, padx=2)

        ttk.Separator(reassign_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Label(reassign_frame, text="Trim start ms:").pack(side=tk.LEFT, padx=5)
        self.trim_start_var = tk.StringVar(value="0")
        self.trim_start_entry = ttk.Entry(reassign_frame, textvariable=self.trim_start_var, width=8)
        self.trim_start_entry.pack(side=tk.LEFT, padx=2)
        self.trim_start_entry.bind('<Return>', self.on_trim_entry_return)

        ttk.Label(reassign_frame, text="Trim end ms:").pack(side=tk.LEFT, padx=5)
        self.trim_end_var = tk.StringVar(value="0")
        self.trim_end_entry = ttk.Entry(reassign_frame, textvariable=self.trim_end_var, width=8)
        self.trim_end_entry.pack(side=tk.LEFT, padx=2)
        self.trim_end_entry.bind('<Return>', self.on_trim_entry_return)

        ttk.Button(reassign_frame, text="Extract New Segment", command=self.extract_new_segment).pack(side=tk.LEFT, padx=5)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(self.root, text="Speaker Statistics", padding="10")
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=6, width=100)
        self.stats_text.pack(fill=tk.X)
        self.stats_text.configure(state=tk.DISABLED)
        self.stats_text.bind('<Escape>', self.release_focus)

        self.new_speaker_entry.bind('<Escape>', self.release_focus)
        self.trim_start_entry.bind('<Escape>', self.release_focus)
        self.trim_end_entry.bind('<Escape>', self.release_focus)
        self.root.bind_all('<Escape>', self.release_focus)
        self.bind_hotkeys()
    
    def load_data(self):
        """Load diarisation results"""
        try:
            self.df = pd.read_csv(RESULTS_FILE)

            # Normalize CSV headers/values to handle accidental surrounding spaces.
            self.df.columns = [str(c).strip() for c in self.df.columns]
            for col in ['segmentation_model', 'model_name', 'diarisation_model', 'speaker_id', 'audio_path', 'segment_filename']:
                if col in self.df.columns:
                    self.df[col] = self.df[col].apply(lambda v: v.strip() if isinstance(v, str) else v)

            segmentation_column = self.get_segmentation_column()
            if segmentation_column is None:
                raise KeyError("Expected 'segmentation_model' or 'model_name' column in diarisation_results.csv")

            segmentation_models = [m for m in self.df[segmentation_column].dropna().unique().tolist() if str(m).strip()]
            self.segmentation_model_combo['values'] = segmentation_models

            if segmentation_models:
                self.segmentation_model_var.set(segmentation_models[0])
                self.on_segmentation_model_change()
            
            messagebox.showinfo("Success", f"Loaded {len(self.df)} diarised segments from {len(segmentation_models)} segmentation models")
            
        except FileNotFoundError:
            messagebox.showerror("Error", f"Results file not found: {RESULTS_FILE}\nRun diarisation first!")
            self.root.quit()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")
            self.root.quit()

    def get_segmentation_column(self):
        """Return segmentation column name used by current dataframe."""
        if self.df is None:
            return None
        if 'segmentation_model' in self.df.columns:
            return 'segmentation_model'
        if 'model_name' in self.df.columns:
            return 'model_name'
        return None

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

    def get_filtered_df(self):
        """Return rows matching the current segmentation and diarisation model selections."""
        if self.df is None:
            return None

        filtered_df = self.df
        segmentation_column = self.get_segmentation_column()
        if segmentation_column is None:
            return filtered_df.iloc[0:0]

        if self.current_segmentation_model:
            filtered_df = filtered_df[filtered_df[segmentation_column] == self.current_segmentation_model]

        if self.current_model:
            filtered_df = filtered_df[filtered_df['diarisation_model'] == self.current_model]

        return filtered_df

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
            'trim_and_save': self.trim_and_save,
            'extract_new_segment': self.extract_new_segment,
            'female': lambda: self.quick_assign('FEMALE'),
            'male': lambda: self.quick_assign('MALE'),
            'male_1': lambda: self.quick_assign('MALE_1'),
            'male_2': lambda: self.quick_assign('MALE_2')
        }

        action_fn = actions.get(action)
        if action_fn is not None:
            action_fn()

    def release_focus(self, event=None):
        """Move focus away from editable fields so hotkeys can be used."""
        self.root.focus_set()
        return "break"

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

    def focus_trim_start(self):
        """Focus trim-start field."""
        if self.trim_start_entry is not None:
            self.trim_start_entry.focus_set()
            self.trim_start_entry.selection_range(0, tk.END)

    def focus_trim_end(self):
        """Focus trim-end field."""
        if self.trim_end_entry is not None:
            self.trim_end_entry.focus_set()
            self.trim_end_entry.selection_range(0, tk.END)

    def on_trim_entry_return(self, event=None):
        """Leave trim input field when Enter is pressed."""
        return self.release_focus(event)

    def trim_and_save(self):
        """Trim the current segment in place while preserving its speaker label."""
        self.trim_current_segment()

    def trim_current_segment(self):
        """Trim the current segment to the requested window and overwrite the row."""
        if self.df is None or self.current_model is None or self.current_speaker is None:
            messagebox.showwarning("Warning", "No segment selected")
            return

        speaker_df = self.get_filtered_df()
        if speaker_df is None:
            messagebox.showwarning("Warning", "No segment selected")
            return

        speaker_df = speaker_df[
            (speaker_df['diarisation_model'] == self.current_model) &
            (speaker_df['speaker_id'] == self.current_speaker)
        ]
        if len(speaker_df) == 0:
            messagebox.showwarning("Warning", "No segment selected")
            return

        if self.current_idx >= len(speaker_df):
            self.current_idx = len(speaker_df) - 1

        if self.current_audio is None or self.current_sr is None:
            messagebox.showwarning("Warning", "No audio loaded")
            return

        segment = speaker_df.iloc[self.current_idx]
        row_index = segment.name

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

        old_audio_rel = str(segment['audio_path'])
        old_audio_path = self.resolve_audio_path(old_audio_rel)
        old_filename = str(segment['segment_filename'])
        speaker_label = str(segment['speaker_id'])

        base_start_ms = int(segment['start_time_ms'])
        new_start_ms = base_start_ms + trim_start_ms
        new_end_ms = base_start_ms + trim_end_ms

        output_dir = old_audio_path.parent
        new_filename = f"{new_start_ms}_{new_end_ms}_{speaker_label}.wav"
        final_path = output_dir / new_filename
        if final_path.exists():
            i = 1
            while True:
                candidate = output_dir / f"{new_start_ms}_{new_end_ms}_{speaker_label}_alt{i}.wav"
                if not candidate.exists():
                    final_path = candidate
                    break
                i += 1

        temp_path = final_path.with_name(f"{final_path.stem}.__tmp__.wav")

        try:
            sf.write(temp_path, trimmed_audio, self.current_sr)
            if not temp_path.exists() or temp_path.stat().st_size == 0:
                raise RuntimeError("Trimmed file was not written correctly")

            os.replace(temp_path, final_path)

            self.df.at[row_index, 'segment_filename'] = final_path.name
            self.df.at[row_index, 'start_time_ms'] = new_start_ms
            self.df.at[row_index, 'end_time_ms'] = new_end_ms
            self.df.at[row_index, 'duration_ms'] = new_end_ms - new_start_ms
            self.df.at[row_index, 'speaker_id'] = speaker_label
            self.df.at[row_index, 'audio_path'] = self.to_project_relative(final_path)
            self.df.to_csv(RESULTS_FILE, index=False)

            remaining_refs = (self.df['audio_path'].astype(str) == old_audio_rel).sum()
            if remaining_refs == 0 and old_audio_path.exists() and old_audio_path.resolve() != final_path.resolve():
                old_audio_path.unlink()

            self.corrections.append({
                'model_name': self.current_model,
                'segment_filename': final_path.name,
                'old_segment_filename': old_filename,
                'new_segment_filename': final_path.name,
                'old_audio_path': old_audio_rel,
                'new_audio_path': self.to_project_relative(final_path),
                'old_speaker': speaker_label,
                'new_speaker': speaker_label,
                'start_time_ms': new_start_ms,
                'end_time_ms': new_end_ms,
                'label': 'trim',
                'timestamp': datetime.now().isoformat()
            })

            self.update_display()

            messagebox.showinfo("Success", f"Trimmed segment saved and opened:\n{final_path}")

        except Exception as e:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            messagebox.showerror("Error", f"Failed to trim segment: {e}")

    def extract_new_segment(self):
        """Create a new trimmed diarisation segment and insert it after current row."""
        if self.df is None or self.current_model is None or self.current_speaker is None:
            messagebox.showwarning("Warning", "No segment selected")
            return

        speaker_df = self.get_filtered_df()
        if speaker_df is None:
            messagebox.showwarning("Warning", "No segment selected")
            return

        speaker_df = speaker_df[
            (speaker_df['diarisation_model'] == self.current_model) &
            (speaker_df['speaker_id'] == self.current_speaker)
        ]
        if len(speaker_df) == 0:
            messagebox.showwarning("Warning", "No segment selected")
            return

        if self.current_idx >= len(speaker_df):
            self.current_idx = len(speaker_df) - 1

        if self.current_audio is None or self.current_sr is None:
            messagebox.showwarning("Warning", "No audio loaded")
            return

        segment = speaker_df.iloc[self.current_idx]
        row_index = segment.name

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

        old_audio_path = self.resolve_audio_path(segment['audio_path'])
        old_filename = str(segment['segment_filename'])

        base_start_ms = int(segment['start_time_ms'])
        new_start_ms = base_start_ms + trim_start_ms
        new_end_ms = base_start_ms + trim_end_ms
        speaker_label = str(segment['speaker_id'])

        output_dir = old_audio_path.parent
        new_filename = f"{new_start_ms}_{new_end_ms}_{speaker_label}.wav"
        final_path = output_dir / new_filename
        if final_path.exists():
            i = 1
            while True:
                candidate = output_dir / f"{new_start_ms}_{new_end_ms}_{speaker_label}_alt{i}.wav"
                if not candidate.exists():
                    final_path = candidate
                    break
                i += 1

        temp_path = final_path.with_name(f"{final_path.stem}.__tmp__.wav")
        insert_pos = self.df.index.get_loc(row_index) + 1

        try:
            sf.write(temp_path, trimmed_audio, self.current_sr)
            if not temp_path.exists() or temp_path.stat().st_size == 0:
                raise RuntimeError("Extracted file was not written correctly")

            os.replace(temp_path, final_path)

            new_row = segment.copy()
            new_row['segment_filename'] = final_path.name
            new_row['start_time_ms'] = new_start_ms
            new_row['end_time_ms'] = new_end_ms
            new_row['duration_ms'] = new_end_ms - new_start_ms
            new_row['audio_path'] = self.to_project_relative(final_path)

            top = self.df.iloc[:insert_pos]
            bottom = self.df.iloc[insert_pos:]
            self.df = pd.concat([top, pd.DataFrame([new_row]), bottom], ignore_index=True)
            self.df.to_csv(RESULTS_FILE, index=False)

            self.corrections.append({
                'model_name': self.current_model,
                'segment_filename': final_path.name,
                'old_segment_filename': old_filename,
                'new_segment_filename': final_path.name,
                'old_audio_path': str(segment['audio_path']),
                'new_audio_path': self.to_project_relative(final_path),
                'start_time_ms': new_start_ms,
                'end_time_ms': new_end_ms,
                'label': 'extract',
                'timestamp': datetime.now().isoformat()
            })

            self.on_model_change()
            self.current_speaker = speaker_label
            self.speaker_var.set(speaker_label)
            self.on_speaker_change()

            speaker_after = self.get_filtered_df()
            if speaker_after is None:
                speaker_after = self.df.iloc[0:0]
            speaker_after = speaker_after[
                (speaker_after['diarisation_model'] == self.current_model) &
                (speaker_after['speaker_id'] == speaker_label)
            ].reset_index(drop=True)
            match = speaker_after[
                (speaker_after['segment_filename'] == final_path.name) &
                (speaker_after['start_time_ms'] == new_start_ms) &
                (speaker_after['end_time_ms'] == new_end_ms)
            ]
            if len(match) > 0:
                self.current_idx = int(match.index[0])
                self.update_display()

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

        # Update speaker list using both filters
        model_df = self.get_filtered_df()
        if model_df is None:
            return

        model_df = model_df[model_df['diarisation_model'] == self.current_model]
        speakers = model_df['speaker_id'].unique().tolist()
        self.speaker_combo['values'] = speakers
        
        if speakers:
            self.speaker_var.set(speakers[0])
            self.on_speaker_change()
        
        self.update_statistics()

    def on_segmentation_model_change(self, event=None):
        """Handle segmentation model selection change."""
        self.current_segmentation_model = self.segmentation_model_var.get()

        if self.df is None:
            return

        # Rebuild diarisation choices from segmentation-only filtering.
        # Do not apply current diarisation filter here, otherwise the dropdown can get stuck.
        model_df = self.df
        segmentation_column = self.get_segmentation_column()
        if segmentation_column and self.current_segmentation_model:
            model_df = model_df[model_df[segmentation_column] == self.current_segmentation_model]

        diarisation_models = model_df['diarisation_model'].dropna().unique().tolist()
        self.model_combo['values'] = diarisation_models

        if diarisation_models:
            if self.current_model not in diarisation_models:
                self.model_var.set(diarisation_models[0])
                self.current_model = diarisation_models[0]
            self.on_model_change()
        else:
            self.current_model = None
            self.model_var.set("")
            self.speaker_combo['values'] = []
            self.speaker_var.set("")
            self.current_speaker = None
            self.current_idx = 0
            self.update_display()

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
        
        # Get segments for current model, speaker, and segmentation model
        speaker_df = self.get_filtered_df()
        if speaker_df is None:
            return

        speaker_df = speaker_df[
            (speaker_df['diarisation_model'] == self.current_model) &
            (speaker_df['speaker_id'] == self.current_speaker)
        ]
        
        if len(speaker_df) == 0:
            return
        
        # Get current segment
        segment = speaker_df.iloc[self.current_idx]

        if self.trim_start_var is not None and self.trim_end_var is not None:
            self.trim_start_var.set("0")
            self.trim_end_var.set(str(int(segment['duration_ms'])))
        
        # Update progress
        self.progress_label.config(text=f"{self.current_idx + 1} / {len(speaker_df)}")
        
        # Update info
        self.info_text.configure(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        segmentation_column = self.get_segmentation_column()
        segmentation_value = segment.get(segmentation_column, "N/A") if segmentation_column else "N/A"
        info = f"""Filename: {segment['segment_filename']}
Time: {segment['start_time_ms']}ms - {segment['end_time_ms']}ms | Duration: {segment['duration_ms']}ms
    Segmentation model: {segmentation_value}
Speaker: {segment['speaker_id']} | Confidence: {segment['confidence']:.3f}
Model: {segment['diarisation_model']}"""
        self.info_text.insert(1.0, info)
        self.info_text.configure(state=tk.DISABLED)
        
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
        
        model_df = self.get_filtered_df()
        if model_df is None:
            return
        model_df = model_df[model_df['diarisation_model'] == self.current_model]
        
        # Speaker distribution
        speaker_counts = model_df['speaker_id'].value_counts()
        
        stats = f"""Model: {self.current_model}
Segmentation model: {self.current_segmentation_model}
Total segments: {len(model_df)}
Speaker distribution:\n"""
        
        for speaker, count in speaker_counts.items():
            pct = (count / len(model_df)) * 100
            avg_conf = model_df[model_df['speaker_id'] == speaker]['confidence'].mean()
            stats += f"  {speaker}: {count} segments ({pct:.1f}%) | Avg confidence: {avg_conf:.3f}\n"
        
        self.stats_text.configure(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats)
        self.stats_text.configure(state=tk.DISABLED)
    
    def prev_segment(self):
        """Go to previous segment"""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_display()
    
    def next_segment(self):
        """Go to next segment"""
        speaker_df = self.get_filtered_df()
        if speaker_df is None:
            return

        speaker_df = speaker_df[
            (speaker_df['diarisation_model'] == self.current_model) &
            (speaker_df['speaker_id'] == self.current_speaker)
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

    def delete_segment_file(self):
        """Delete current segment row and remove audio file when it is unreferenced."""
        if self.df is None or self.current_model is None or self.current_speaker is None:
            messagebox.showwarning("Warning", "No segment selected")
            return

        speaker_df = self.get_filtered_df()
        if speaker_df is None:
            messagebox.showwarning("Warning", "No segment selected")
            return

        speaker_df = speaker_df[
            (speaker_df['diarisation_model'] == self.current_model) &
            (speaker_df['speaker_id'] == self.current_speaker)
        ]
        if len(speaker_df) == 0:
            messagebox.showwarning("Warning", "No segment selected")
            return

        if self.current_idx >= len(speaker_df):
            self.current_idx = len(speaker_df) - 1

        segment = speaker_df.iloc[self.current_idx]
        row_index = segment.name
        segment_filename = str(segment['segment_filename'])
        audio_rel = str(segment['audio_path'])
        audio_path = self.resolve_audio_path(audio_rel)

        if not messagebox.askyesno(
            "Delete File",
            f"Delete this segment from CSV and remove file if unreferenced?\n\n{audio_path}"
        ):
            return

        try:
            self.df = self.df.drop(index=row_index).reset_index(drop=True)
            self.df.to_csv(RESULTS_FILE, index=False)

            remaining_refs = 0
            if len(self.df) > 0:
                remaining_refs = (self.df['audio_path'].astype(str) == audio_rel).sum()

            if remaining_refs == 0 and audio_path.exists():
                audio_path.unlink()

            self.corrections.append({
                'model_name': self.current_model,
                'segment_filename': segment_filename,
                'old_segment_filename': segment_filename,
                'new_segment_filename': None,
                'old_audio_path': audio_rel,
                'new_audio_path': None,
                'label': 'delete',
                'timestamp': datetime.now().isoformat()
            })

            self.current_audio = None
            self.current_sr = None

            seg_df = self.df
            segmentation_column = self.get_segmentation_column()
            if segmentation_column and self.current_segmentation_model:
                seg_df = seg_df[seg_df[segmentation_column] == self.current_segmentation_model]

            models = seg_df['diarisation_model'].unique().tolist()
            self.model_combo['values'] = models

            if not models:
                self.current_model = None
                self.current_speaker = None
                self.progress_label.config(text="0 / 0")
                self.info_text.configure(state=tk.NORMAL)
                self.info_text.delete(1.0, tk.END)
                self.info_text.configure(state=tk.DISABLED)
                self.stats_text.configure(state=tk.NORMAL)
                self.stats_text.delete(1.0, tk.END)
                self.stats_text.configure(state=tk.DISABLED)
                self.ax.clear()
                self.ax.text(0.5, 0.5, 'No segments remaining', ha='center', va='center', transform=self.ax.transAxes)
                self.canvas.draw()
                messagebox.showinfo("Success", "Segment deleted. No segments remain.")
                return

            if self.current_model not in models:
                self.current_model = models[0]
            self.model_var.set(self.current_model)

            model_df = seg_df[seg_df['diarisation_model'] == self.current_model]
            speakers = model_df['speaker_id'].unique().tolist()
            self.speaker_combo['values'] = speakers

            if self.current_speaker not in speakers:
                self.current_speaker = speakers[0]
            self.speaker_var.set(self.current_speaker)

            speaker_after = model_df[model_df['speaker_id'] == self.current_speaker]
            if len(speaker_after) > 0:
                self.current_idx = min(self.current_idx, len(speaker_after) - 1)
            else:
                self.current_idx = 0

            self.update_display()
            self.update_statistics()
            messagebox.showinfo("Success", "Segment deleted and CSV updated")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete segment: {e}")
    
    def reassign_speaker(self):
        """Reassign current segment to new speaker and persist to CSV/files."""
        new_speaker = self.new_speaker_var.get().strip()
        if not new_speaker:
            messagebox.showwarning("Warning", "Enter new speaker ID")
            return
        
        speaker_df = self.get_filtered_df()
        if speaker_df is None:
            messagebox.showwarning("Warning", "No segment selected")
            return

        speaker_df = speaker_df[
            (speaker_df['diarisation_model'] == self.current_model) &
            (speaker_df['speaker_id'] == self.current_speaker)
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
        
        segmentation_column = self.get_segmentation_column()
        correction = {
            'segmentation_model': segment.get(segmentation_column, None) if segmentation_column else None,
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

        # Refresh while keeping reviewer context on the original speaker timeline.
        anchor_start_ms = int(segment['start_time_ms'])
        previous_speaker = old_speaker

        self.on_model_change()

        available_speakers = list(self.speaker_combo['values'])
        if previous_speaker in available_speakers:
            self.current_speaker = previous_speaker
            self.speaker_var.set(previous_speaker)
        elif available_speakers:
            self.current_speaker = available_speakers[0]
            self.speaker_var.set(self.current_speaker)
        else:
            self.current_speaker = None
            self.speaker_var.set("")

        if self.current_speaker is not None:
            speaker_after = self.get_filtered_df()
            if speaker_after is not None:
                speaker_after = speaker_after[
                    (speaker_after['diarisation_model'] == self.current_model) &
                    (speaker_after['speaker_id'] == self.current_speaker)
                ]

                if len(speaker_after) > 0:
                    ordered = speaker_after.sort_values(['start_time_ms', 'end_time_ms'])
                    next_rows = ordered[ordered['start_time_ms'] > anchor_start_ms]
                    if len(next_rows) > 0:
                        target_row = next_rows.index[0]
                    else:
                        target_row = ordered.index[-1]
                    self.current_idx = speaker_after.index.get_loc(target_row)
                else:
                    self.current_idx = 0
            else:
                self.current_idx = 0
        else:
            self.current_idx = 0

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
            segmentation_column = self.get_segmentation_column()
            
            for model in self.df['diarisation_model'].unique():
                model_df = self.df[self.df['diarisation_model'] == model]
                
                f.write(f"\nModel: {model}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total segments: {len(model_df)}\n\n")
                if segmentation_column and segmentation_column in model_df.columns:
                    f.write("Segmentation models:\n")
                    for seg_model, seg_count in model_df[segmentation_column].value_counts().items():
                        f.write(f"  {seg_model}: {seg_count}\n")
                    f.write("\n")
                
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
                    old_speaker = corr.get('old_speaker')
                    new_speaker = corr.get('new_speaker')
                    if old_speaker is not None and new_speaker is not None:
                        f.write(f"  {old_speaker} → {new_speaker}\n")
                    else:
                        f.write(f"  Label: {corr.get('label', 'update')}\n")
                    f.write(f"  Time: {corr['timestamp']}\n\n")
        
        messagebox.showinfo("Success", f"Report saved to:\n{output_file}")


def main():
    """Main function"""
    root = tk.Tk()
    app = DiarisationReviewTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
