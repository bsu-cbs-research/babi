from typing import Optional, Callable, TypedDict, Literal, Any
import os
import threading
import queue
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from program.parse import capnostream
from models import autoencoder, processing

type QueueTags = Literal["success", "error", "status"]
type QueueItem = tuple[QueueTags, Any]
type QueueCallback = Optional[Callable[[Any], None]]
type QueueCallbacks = dict[QueueTags, QueueCallback]

class FileModifierApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root: tk.Tk = root
        self.root.title("Simple File Modifier")
        
        # window geometry and centering
        window_width: int = 400
        window_height: int = 300
        screen_width: int = self.root.winfo_screenwidth()
        screen_height: int = self.root.winfo_screenheight()
        center_x: int = int(screen_width / 2 - window_width / 2)
        center_y: int = int(screen_height / 2 - window_height / 2)
        self.root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
        self.root.resizable(False, False)

        self.file_path: Optional[str] = None
        self.modified_content: Optional[bytes] = None

        self.main_frame: ttk.Frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.queue: queue.Queue[QueueItem] = queue.Queue()

        self.model = autoencoder.from_file("program/static/model.keras")
        self.threshold = float(open("program/static/threshold.txt").read().strip())
        self.min, self.max = [float(v) for v in open("program/static/scaling.txt").read().strip().split(',')]

        self.batch_predict = autoencoder.batch_predictor(self.model, self.threshold)

        self.show_start_screen()

    def show_start_screen(self) -> None:
        # Clear frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        self.file_path = None
        self.modified_content = None

        label: ttk.Label = ttk.Label(self.main_frame, text="Upload a file to modify", font=("Helvetica", 14))
        label.pack(pady=20)

        upload_btn: ttk.Button = ttk.Button(self.main_frame, text="Upload File", command=self._handle_file_upload)
        upload_btn.pack(pady=10)

    # todo: combine into single run-on-thread function
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
        # Keep polling without blocking the UI
        self.root.after(100, lambda: self._poll_queue(callbacks))
    
    def _handle_file_upload(self) -> None:
        file_path: str = filedialog.askopenfilename()
        if not file_path: 
            messagebox.showerror("Error", "No file selected")
            return
        
        def _parse_file(file_path: str) -> None:
            extension = os.path.splitext(file_path)[1]
            try: 
                if extension == ".xlsx": 
                    self.queue.put(("status", "Parsing preprocessed data..."))
                    df = capnostream.parse_preprocessed_file(file_path)
                    self.queue.put(("success", df))
                elif extension == ".csv": 
                    self.queue.put(("status", "Parsing raw data..."))
                    df = capnostream.parse_raw_file(file_path)
                    self.queue.put(("success", df))
                else: raise ValueError(f"Unsupported file format, got {extension} expected .xlsx, .csv")
            except Exception as e: self.queue.put(("error", str(e)))

        # show progress screen
        progress_label, _ = self.show_progress_screen()
        
        threading.Thread(target=_parse_file, args=(file_path,), daemon=True).start()

        def _on_error(data: str) -> None:
            self.show_start_screen()
            messagebox.showerror("Error", data)

        def _on_status(data: str) -> None:
            progress_label.configure(text=data)

        self.root.after(100, lambda: self._poll_queue({ "success": self._handle_file_analysis, "error": _on_error, "status": _on_status }))

    def _handle_file_analysis(self, df: pd.DataFrame) -> None:
        units = processing.signal_to_units(df["co2_wave"].to_numpy(), self.min, self.max)
        recons, errors, labels = self.batch_predict(units) # todo: move to thread, update loading status
        print(len(labels)) # todo: time-align labels (1 for good/0 for bad)
        self.show_download_screen()

    def show_download_screen(self) -> None:
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        label: ttk.Label = ttk.Label(self.main_frame, text="File Modified!", font=("Helvetica", 14))
        label.pack(pady=20)

        if self.file_path:
            filename: str = os.path.basename(self.file_path)
            info_label: ttk.Label = ttk.Label(self.main_frame, text=f"Original: {filename}")
            info_label.pack(pady=5)

        download_btn: ttk.Button = ttk.Button(self.main_frame, text="Download Modified File", command=self.save_file)
        download_btn.pack(pady=10)

        reset_btn: ttk.Button = ttk.Button(self.main_frame, text="Back to Start", command=self.show_start_screen)
        reset_btn.pack(pady=10)

    def show_progress_screen(self):
        for widget in self.main_frame.winfo_children(): widget.destroy()

        label: ttk.Label = ttk.Label(self.main_frame, text="Processing...")
        label.pack(pady=20)

        progress: ttk.Progressbar = ttk.Progressbar(self.main_frame, mode='indeterminate')
        progress.pack(pady=10, fill=tk.X)
        progress.start()

        return label, progress

    def save_file(self) -> None:
        if self.modified_content:
            # Suggest a filename based on the original
            initial_file: str = "modified_" + os.path.basename(self.file_path) if self.file_path else "modified_file"
            
            save_path: str = filedialog.asksaveasfilename(initialfile=initial_file)
            if save_path:
                try:
                    with open(save_path, 'wb') as f:
                        f.write(self.modified_content)
                    messagebox.showinfo("Success", "File saved successfully!")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save file: {e}")

    def on_closing(self):
        """Properly shuts down the executor when the window closes."""
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FileModifierApp(root)
    photo = tk.PhotoImage(file="program/static/program-icon@1x.png")
    root.iconphoto(True, photo) 
    root.protocol("WM_DELETE_WINDOW", app.on_closing) 

    root.mainloop()
