from typing import Optional, Callable, Literal, Any
import os
import threading
import queue
import pandas as pd
import numpy as np
from tkinter import filedialog, messagebox, PhotoImage
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from program import helpers
from models import autoencoder, processing
from data.prepare import constants

type QueueTags = Literal["success", "error", "status"]
type QueueItem = tuple[QueueTags, Any]
type QueueCallback = Optional[Callable[[Any], None]]
type QueueCallbacks = dict[QueueTags, QueueCallback]

class BABIDataAnalysisApp:
    def __init__(self, root: ttk.Window) -> None:
        self.root: ttk.Window = root
        self.root.title("BABI Data Analysis")
        
        # window geometry and centering
        window_width: int = 400
        window_height: int = 300
        screen_width: int = self.root.winfo_screenwidth()
        screen_height: int = self.root.winfo_screenheight()
        center_x: int = int(screen_width / 2 - window_width / 2)
        center_y: int = int(screen_height / 2 - window_height / 2)
        self.root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
        self.root.resizable(False, False)

        self.main_frame: ttk.Frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill=BOTH, expand=True)

        self.queue: queue.Queue[QueueItem] = queue.Queue()

        self.model = autoencoder.from_file("program/static/model.keras")
        self.threshold = float(open("program/static/threshold.txt").read().strip())
        self.min, self.max = [float(v) for v in open("program/static/scaling.txt").read().strip().split(',')]

        self.batch_predict = autoencoder.batch_predictor(self.model, self.threshold)
        self.show_start_screen()

    def show_start_screen(self) -> None:
        self._reset_canvas()
        ttk.Label(self.main_frame, text="BABI Data Analysis", font=("Helvetica", 14)).pack(pady=(20, 0))
        ttk.Label(self.main_frame, text="Upload a raw capnostream (.csv) file to begin").pack(pady=(8, 0))
        ttk.Button(self.main_frame, text="Upload File", command=self._handle_file_upload).pack(pady=40)

    def _run_thread(self, thread: threading.Thread, callbacks: QueueCallbacks) -> None:
        thread.start()
        self.root.after(100, lambda: self._poll_queue(callbacks))

    def _poll_queue(self, callbacks: QueueCallbacks) -> None:
        try:
            item = self.queue.get_nowait()
            if item:
                tag, data = item
                callback = callbacks.get(tag)
                if callback: callback(data)
                if tag == "success" or tag == "error": return
        except queue.Empty:
            pass
        
        self.root.after(100, lambda: self._poll_queue(callbacks))
    
    def _handle_file_upload(self) -> None:
        file_path: str = filedialog.askopenfilename()
        if not file_path: return
        
        def _process_file(file_path: str) -> None:
            extension = os.path.splitext(file_path)[1]
            try: 
                if extension == ".xlsx": 
                    self.queue.put(("status", "Parsing..."))
                    df = helpers.parse_preprocessed_capnostream_file(file_path)
                    self.queue.put(("success", df))
                elif extension == ".csv": 
                    self.queue.put(("status", "Parsing..."))
                    df = helpers.parse_raw_capnostream_file(file_path)

                    self.queue.put(("status", "Converting..."))
                    signal = df["co2_wave"].to_numpy()
                    original_length = len(signal)

                    units = processing.signal_to_units(signal, self.min, self.max)

                    self.queue.put(("status", "Analyzing..."))
                    _, _, labels = self.batch_predict(units)
                    self.queue.put(("status", "Reformatting..."))

                    # create analysis blocks 
                    labels = np.concatenate([labels.astype(int), [0] * (original_length - len(labels) + constants.unit_length)]) # pad labels
                    normative_label_indices = np.where(labels == 1)[0] # gather indices of normative labels
                    insertion_block = np.full(constants.unit_length, 1) # create block of ones
                    for idx in normative_label_indices: labels[idx : idx + constants.unit_length] = insertion_block
                    labels = labels[:original_length] # trim back to original length if overflow
                        
                    df["co2_wave_labels"] = labels 
                    self.queue.put(("success", df))
                else: raise ValueError(f"Unsupported file format, got {extension} expected .xlsx, .csv")
            except Exception as e: self.queue.put(("error", str(e)))

        # show progress screen
        progress_label, _ = self.show_progress_screen()
        thread = threading.Thread(target=_process_file, args=(file_path,), daemon=True)

        def _on_success(df: pd.DataFrame):
            self.show_download_screen(df)

        def _on_error(data: str):
            self.show_start_screen()
            messagebox.showerror("Error", data)

        def _on_status(data: str):
            progress_label.configure(text=data)

        self._run_thread(thread, { "success": _on_success, "error": _on_error, "status": _on_status })

    def show_download_screen(self, df: pd.DataFrame):
        self._reset_canvas()

        ttk.Label(self.main_frame, text="Analysis Complete", font=("Helvetica", 14)).pack(pady=(20, 0))
        ttk.Label(self.main_frame, text="Download the modified file below.").pack(pady=(8, 0))

        def _save_csv() -> None:
            try:
                save_path: str = filedialog.asksaveasfilename(
                    initialfile="annotated-raw.csv", # todo: use real file name
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
                )
                if not save_path: return
                df.to_csv(save_path, index=False) 
                messagebox.showinfo("Success", "CSV saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save CSV: {e}")

        ttk.Button(self.main_frame, text="Download", command=_save_csv).pack(pady=(40, 0))
        ttk.Button(self.main_frame, text="Back", command=self.show_start_screen, style=SECONDARY).pack(pady=10)

    def show_progress_screen(self):
        self._reset_canvas()

        label = ttk.Label(self.main_frame, text="Processing...")
        label.pack(pady=(80, 0))

        progress = ttk.Progressbar(self.main_frame, mode='indeterminate')
        progress.pack(pady=(30, 0), fill=X)
        progress.start()

        return label, progress

    def on_closing(self):
        self.root.destroy()

    def _reset_canvas(self):
        for widget in self.main_frame.winfo_children(): widget.destroy()

if __name__ == "__main__":
    root = ttk.Window(themename="darkly")
    app = BABIDataAnalysisApp(root)
    photo = PhotoImage(file="program/static/program-icon@1x.png")
    root.iconphoto(True, photo) 
    root.protocol("WM_DELETE_WINDOW", app.on_closing) 

    root.mainloop()
