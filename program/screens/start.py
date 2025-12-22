from typing import TYPE_CHECKING
import os
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

if TYPE_CHECKING: from program.app import BABIDataAnalysisApp

class Screen(ttk.Frame):
    ID = "start"

    def __init__(self, parent: ttk.Frame, controller: "BABIDataAnalysisApp"):
        super().__init__(parent)
        self.controller = controller

        ttk.Label(self, text="BABI Data Analysis", font=("Helvetica", 14)).pack(pady=(20, 0))
        ttk.Label(self, text="Upload a raw capnostream (.csv) file to begin").pack(pady=(8, 0))
        ttk.Button(self, text="Upload File", command=self._handle_file_upload).pack(pady=40)

    def _handle_file_upload(self):
        file_path = filedialog.askopenfilename()
        if not file_path: return
        extension = os.path.splitext(file_path)[1]
        if extension != ".csv": return messagebox.showerror("Error", f"Unsupported file format, got {extension} expected .csv")
        self.controller.context.file_path = file_path
        self.controller.show_screen("configuration")