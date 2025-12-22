from typing import TYPE_CHECKING
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

if TYPE_CHECKING: from program.app import BABIDataAnalysisApp

class Screen(ttk.Frame):
    ID = "complete"

    def __init__(self, parent: ttk.Frame, controller: "BABIDataAnalysisApp"):
        super().__init__(parent)
        self.controller = controller
        
        ttk.Label(self, text="Analysis Complete", font=("Helvetica", 14)).pack(pady=(20, 0))
        ttk.Label(self, text="Download the modified file below.").pack(pady=(8, 0))

        ttk.Button(self, text="Download", command=self._handle_download).pack(pady=(40, 0))
        ttk.Button(self, text="Back", command=self._handle_back, style=SECONDARY).pack(pady=10)

    def _handle_back(self):
        self.controller.context.reset()
        self.controller.show_screen("start")

    def _handle_download(self):
        xlsx_buffer = self.controller.context.xlsx_buffer
        if xlsx_buffer is None: return messagebox.showerror("Error", "No data available to save.")

        try:
            save_path: str = filedialog.asksaveasfilename(
                initialfile="BB-XXX-X Annotated Capnostream Data.xlsx",
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            if not save_path: return
            with open(save_path, "wb") as f:
                f.write(xlsx_buffer.getvalue())
            messagebox.showinfo("Success", "Excel file saved successfully!")
        except Exception as e: messagebox.showerror("Error", f"Failed to save Excel file: {e}")