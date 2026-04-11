from typing import TYPE_CHECKING
import os
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

if TYPE_CHECKING:
    from program.app import BABIDataAnalysisApp


class Screen(ttk.Frame):
    ID = "start"

    def __init__(self, parent: ttk.Frame, controller: "BABIDataAnalysisApp"):
        super().__init__(parent)
        self.controller = controller

        ttk.Label(
            self, text="BABI Data Analysis", font=("Helvetica", 14)
        ).pack(pady=(20, 0))
        ttk.Label(
            self,
            text=(
                "Select a folder containing one capnostream (.xlsx) file\n"
                "and one or more motion (.tsv) files to begin."
            ),
            justify="center",
        ).pack(pady=(8, 0))
        ttk.Button(
            self, text="Select Folder", command=self._handle_folder_select
        ).pack(pady=40)

    def _handle_folder_select(self):
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return

        try:
            self._quick_presence_check(folder_path)
        except ValueError as e:
            return messagebox.showerror("Error", str(e))

        self.controller.context.folder_path = folder_path
        self.controller.show_screen("configuration")

    @staticmethod
    def _quick_presence_check(folder_path: str) -> None:
        """Cheap sanity check before advancing to the configuration screen.

        Deeper structural + schema validation runs inside the pipeline thread
        so the user doesn't stall the UI on large files.
        """
        if not os.path.isdir(folder_path):
            raise ValueError(f"Not a directory: {folder_path}")

        entries = os.listdir(folder_path)
        xlsx_files = [f for f in entries if f.lower().endswith(".xlsx")]
        tsv_files = [f for f in entries if f.lower().endswith(".tsv")]

        if len(xlsx_files) == 0:
            raise ValueError(
                "No .xlsx file found in the selected folder. "
                "Expected exactly one capnostream .xlsx file."
            )
        if len(xlsx_files) > 1:
            raise ValueError(
                f"Expected exactly one .xlsx file, found {len(xlsx_files)}: "
                f"{xlsx_files}"
            )
        if len(tsv_files) == 0:
            raise ValueError(
                "No .tsv files found in the selected folder. "
                "Expected at least one motion .tsv file."
            )
