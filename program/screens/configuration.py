from typing import TYPE_CHECKING, List
import os
import re
import threading
import numpy as np
import pandas as pd
import io
from tkinter import messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from datetime import datetime
from program import helpers
from models import processing
from data.prepare import constants

if TYPE_CHECKING: from program.app import BABIDataAnalysisApp

class Screen(ttk.Frame):
    ID = "configuration"

    def __init__(self, parent: ttk.Frame, controller: "BABIDataAnalysisApp"):
        super().__init__(parent)
        self.controller = controller

        # local context
        self.model = None
        self.threshold = None
        self.min_val = None
        self.max_val = None
        self.batch_predict = None
        
        trials: dict[str, tuple[int, int]] = {}

        header_frame = ttk.Frame(self)
        header_frame.pack(fill=X, pady=(0, 20))
        header_frame.columnconfigure(1, weight=1)

        ttk.Label(header_frame, text="Configuration", font=("Helvetica", 14), anchor="center").grid(row=0, column=0, sticky=W)
        ttk.Button(header_frame, text="Back", command=lambda: controller.show_screen("start"), style=SECONDARY).grid(row=0, column=1, sticky=E, padx=(0, 5))
        ttk.Button(header_frame, text="Proceed", command=lambda: self._handle_proceed(trials)).grid(row=0, column=2, sticky=E)

        # input field
        ttk.Label(self, text="Add trial: ").pack(anchor="w", pady=(10, 0))
        input_frame = ttk.Frame(self)
        input_frame.pack(fill=X, pady=5)
        entry_box = ttk.Entry(input_frame)
        entry_box.pack(side=LEFT, fill=X, expand=True)
        add_trial_button = ttk.Button(input_frame, text="Add")
        add_trial_button.pack(side=LEFT, padx=(5, 0))
        ttk.Label(self, text="Format: NAME [HH:MM:SS(AM/PM)-HH:MM:SS(AM/PM)]", style=SECONDARY).pack(anchor="w", pady=(2, 0))
        ttk.Label(self, text="Example: Trial 1 [12:23:03PM-12:30:21PM]", style=SECONDARY).pack(anchor="w")

        trial_info_frame = ttk.Frame(self)
        trial_info_frame.pack(fill=X, pady=10)

        def _handle_add_trial():
            raw_text = entry_box.get().strip()
            if not raw_text: return messagebox.showerror("Error", "Input cannot be empty.")
            try:
                valid, data = self._validate_trial_string(raw_text)
                if not valid or data is None: raise ValueError("Incomplete information.")
                name, (start, end), (start_s, end_s) = data # todo: validation should look for overlaps etc.
                if name in trials: return messagebox.showerror("Error", f"Trial with name '{name}' already exists.")
                if start_s >= end_s: return messagebox.showerror("Error", "Start time must be earlier than end time.")
                if len(name) == 0: return messagebox.showerror("Error", "Trial name cannot be empty.")
                if len(name) > 31: return messagebox.showerror("Error", "Trial name cannot exceed 31 characters.")

                trial_frame = ttk.Frame(trial_info_frame)
                trial_frame.pack(fill=X, pady=2)

                def _remove_trial():
                    trial_frame.destroy()
                    del trials[name]
                    self.update_idletasks()

                button = ttk.Button(trial_frame, text="Remove", style=SECONDARY, command=_remove_trial)
                button.pack(side=RIGHT, padx=(5, 0))

                label = ttk.Label(trial_frame, text=f"{name}: {start} - {end}")
                label.pack(side=LEFT, fill=X, expand=True)

                trials[name] = (start_s, end_s)
                
                entry_box.delete(0, END)
            except Exception as e:
                messagebox.showerror("Error", f"Invalid format. Please use: NAME [HH:MM:SS(AM/PM)-HH:MM:SS(AM/PM)]")

        add_trial_button.configure(command=_handle_add_trial)

    @staticmethod
    def _validate_trial_string(trial_string: str):
        time_regex = r"(?:0?[1-9]|1[0-2]):[0-5][0-9]:[0-5][0-9](?:AM|PM|am|pm)"
        full_pattern = rf"^(?P<name>.*)\s*\[(?P<start>{time_regex})-(?P<end>{time_regex})\]$"
        match = re.match(full_pattern, trial_string.strip())
        
        if match:
            d = match.groupdict()
            name = d["name"].strip()
            start_time = d["start"]
            end_time = d["end"]
            return True, (name, (start_time, end_time), (Screen._time_to_seconds(start_time), Screen._time_to_seconds(end_time)))
        else:
            return False, None
        
    @staticmethod
    def _time_to_seconds(t: str, pattern: str = "%I:%M:%S%p") -> int:
        dt = datetime.strptime(t, pattern)
        return (dt.hour * 3600) + (dt.minute * 60) + dt.second
    
    @staticmethod
    def find_time_overlaps(flat_trials: list[tuple[int, int, str]]):
        flat_trials.sort(key=lambda x: x[0])  # sort by start time

        overlaps: list[tuple[str, str]] = []
        for i in range(1, len(flat_trials)):
            prev_end = flat_trials[i-1][1]
            curr_start = flat_trials[i][0]
            if curr_start <= prev_end: overlaps.append((flat_trials[i-1][2], flat_trials[i][2]))

        return overlaps

    def _handle_proceed(self, trials: dict[str, tuple[int, int]]):
        flat_trials = list((start, end, name) for name, (start, end) in trials.items())

        overlaps = self.find_time_overlaps(flat_trials)

        if len(overlaps) > 0:
            overlap_str = "\n".join([f"'{a}' and '{b}'" for a, b in overlaps])
            return messagebox.showerror("Error", f"Overlapping trial times detected between:\n{overlap_str}")

        def _process_file(file_path: str):
            extension = os.path.splitext(file_path)[1]
            try: 
                if extension != ".csv": raise ValueError(f"Unsupported file format, got {extension} expected .csv")

                from models import autoencoder
                from program.app import resource_path # ew
                self.model = autoencoder.from_file(resource_path("program/static/model.keras"))
                self.threshold = float(open(resource_path("program/static/threshold.txt")).read().strip())
                self.min_val, self.max_val = [float(v) for v in open(resource_path("program/static/scaling.txt")).read().strip().split(',')]
                self.batch_predict = autoencoder.batch_predictor(self.model, self.threshold)

                self.controller.queue.put(("status", "Parsing..."))
                df = helpers.parse_raw_capnostream_file(file_path)

                self.controller.queue.put(("status", "Converting..."))
                signal = df["co2_wave"].to_numpy()
                original_length = len(signal)

                units = processing.signal_to_units(signal, self.min_val, self.max_val)

                self.controller.queue.put(("status", "Analyzing..."))
                _, _, labels = self.batch_predict(units)
                self.controller.queue.put(("status", "Reformatting..."))

                # create analysis blocks 
                labels = np.concatenate([labels.astype(int), [0] * (original_length - len(labels) + constants.unit_length)]) # pad labels
                normative_label_indices = np.where(labels == 1)[0] # gather indices of normative labels
                insertion_block = np.full(constants.unit_length, 1) # create block of ones
                for idx in normative_label_indices: labels[idx : idx + constants.unit_length] = insertion_block
                labels = labels[:original_length] # trim back to original length if overflow
                    
                df["time_seconds"] = df["time"].transform(lambda t: self._time_to_seconds(t, pattern="%I:%M:%S %p"))

                intervals: pd.IntervalIndex = pd.IntervalIndex.from_tuples([(s, e) for s, e, _ in flat_trials], closed='both')
                trial_label_map: dict[pd.Interval, str] = dict(zip(intervals, [name for _, _, name in flat_trials]))

                df['trial_label'] = pd.cut(df['time_seconds'], bins=intervals).map(trial_label_map)
                df["CO₂ Wave Labels"] = labels 

                df.rename(columns={"co2_wave": "CO₂ Wave (mmHg)"}, inplace=True)
                df.rename(columns={"EtCO₂": "EtCO₂ (mmHg)"}, inplace=True)
                df.rename(columns={"RR": "RR (bpm)"}, inplace=True)
                df.rename(columns={"date": "Date"}, inplace=True)
                df.rename(columns={"time": "Time"}, inplace=True)

                df.drop(columns=["time_seconds"], inplace=True)
                df = df.dropna(subset=["trial_label"]) # drop rows not in any trial

                xlsx_buffer = io.BytesIO()
                # with pd.ExcelWriter(xlsx_buffer, engine='xlsxwriter') as writer:
                #     for name in trials.keys():
                #         df_subset = df[df['trial_label'] == name].drop(columns=['trial_label'])
                #         df_subset.to_excel(writer, sheet_name=name[:31], index=False)


                with pd.ExcelWriter(xlsx_buffer, engine='xlsxwriter') as writer:
                    for name in trials.keys():
                        df_subset = df[df['trial_label'] == name].drop(columns=['trial_label'])
                        
                        if not df_subset.empty:
                            sheet_name = name[:31] # Excel limit
                            df_subset.to_excel(writer, sheet_name=sheet_name, index=False)
                            
                            # --- ADJUST COLUMN WIDTHS ---
                            worksheet = writer.sheets[sheet_name]
                            for i, col in enumerate(df_subset.columns):
                                # Calculate max length of data in column or header
                                max_len = df_subset[col].astype(str).str.len().max()
                                max_len = max(max_len, len(col)) + 2
                                worksheet.set_column(i, i, max_len)

                self.controller.queue.put(("success", xlsx_buffer))

            except Exception as e: self.controller.queue.put(("error", str(e)))

        file_path = self.controller.context.file_path
        if not file_path: return messagebox.showerror("Error", "No file selected.")

        # show progress screen
        self.controller.show_screen("progress")
        progress_label = self.controller.context.progress_label

        if not progress_label: 
            messagebox.showerror("Error", "Progress label not found in context.")
            return self.controller.show_screen("start")

        thread = threading.Thread(target=_process_file, args=(file_path,), daemon=True)

        def _on_success(xlsx_buffer: io.BytesIO):
            self.controller.context.xlsx_buffer = xlsx_buffer
            self.controller.show_screen("complete")

        def _on_error(data: str):
            messagebox.showerror("Error", data)

        def _on_status(data: str):
            if progress_label: progress_label.configure(text=data)

        self.controller.run_thread(thread, { "success": _on_success, "error": _on_error, "status": _on_status })