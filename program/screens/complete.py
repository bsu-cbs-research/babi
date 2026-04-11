from typing import TYPE_CHECKING
import os
from tkinter import filedialog, messagebox
import pandas as pd
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

if TYPE_CHECKING:
    from program.app import BABIDataAnalysisApp


class Screen(ttk.Frame):
    ID = "complete"

    def __init__(self, parent: ttk.Frame, controller: "BABIDataAnalysisApp"):
        super().__init__(parent)
        self.controller = controller

        ttk.Label(
            self, text="Analysis Complete", font=("Helvetica", 14)
        ).pack(pady=(20, 0))
        ttk.Label(
            self,
            text=(
                "Choose a destination folder below. A subfolder named\n"
                "'<source-folder-name>_processed' will be created containing\n"
                "co2.csv, motion.csv, and metadata.txt."
            ),
            justify="center",
        ).pack(pady=(8, 0))

        ttk.Button(self, text="Download", command=self._handle_download).pack(
            pady=(40, 0)
        )
        ttk.Button(
            self, text="Back", command=self._handle_back, style=SECONDARY
        ).pack(pady=10)

    def _handle_back(self) -> None:
        self.controller.context.reset()
        self.controller.show_screen("start")

    def _handle_download(self) -> None:
        ctx = self.controller.context
        motion_df = ctx.motion_df
        capnostream_df = ctx.capnostream_df
        folder_path = ctx.folder_path

        if motion_df is None or capnostream_df is None:
            return messagebox.showerror("Error", "No data available to save.")
        if not folder_path:
            return messagebox.showerror(
                "Error", "Source folder path missing from context."
            )

        destination = filedialog.askdirectory(
            title="Select destination folder"
        )
        if not destination:
            return

        source_name = os.path.basename(os.path.normpath(folder_path)) or "output"
        out_dir = os.path.join(destination, f"{source_name}_processed")

        if os.path.exists(out_dir):
            overwrite = messagebox.askyesno(
                "Overwrite?",
                f"{out_dir} already exists. Overwrite its contents?",
            )
            if not overwrite:
                return
        else:
            try:
                os.makedirs(out_dir)
            except OSError as e:
                return messagebox.showerror(
                    "Error", f"Could not create output directory: {e}"
                )

        try:
            capnostream_out = os.path.join(out_dir, "co2.csv")
            capnostream_df.to_csv(capnostream_out, index=False)

            motion_out = os.path.join(out_dir, "motion.csv")
            self._write_motion_csv(motion_df, motion_out)

            metadata_out = os.path.join(out_dir, "metadata.txt")
            metadata_text = self._collect_metadata_text(folder_path)
            with open(metadata_out, "w", encoding="utf-8") as f:
                f.write(metadata_text)
        except Exception as e:  # noqa: BLE001
            return messagebox.showerror("Error", f"Failed to save CSVs: {e}")

        messagebox.showinfo(
            "Success", f"Saved co2.csv, motion.csv, and metadata.txt to:\n{out_dir}"
        )

    @staticmethod
    def _write_motion_csv(motion_df: "pd.DataFrame", path: str) -> None:
        """Materialize the motion DatetimeIndex as a leading ISO-8601 `time` column."""
        df = motion_df.copy()
        df.index.name = "time"
        df = df.reset_index()
        df.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M:%S.%f")

    @staticmethod
    def _collect_metadata_text(source_folder: str) -> str:
        entries = os.listdir(source_folder)
        xlsx_files = sorted(f for f in entries if f.lower().endswith(".xlsx"))
        tsv_files = sorted(f for f in entries if f.lower().endswith(".tsv"))

        lines: list[str] = []
        lines.append("BABI Data Analysis - Extracted Source Metadata")
        lines.append(f"Source Folder: {source_folder}")
        lines.append("")

        lines.append("[Capnostream]")
        if not xlsx_files:
            lines.append("No .xlsx file found in source folder.")
        else:
            cap_path = os.path.join(source_folder, xlsx_files[0])
            lines.append(f"File: {xlsx_files[0]}")
            lines.extend(Screen._extract_capnostream_metadata(cap_path))
        lines.append("")

        lines.append("[Motion]")
        if not tsv_files:
            lines.append("No .tsv files found in source folder.")
        else:
            for file_name in tsv_files:
                motion_path = os.path.join(source_folder, file_name)
                lines.append(f"File: {file_name}")
                lines.extend(Screen._extract_motion_metadata(motion_path))
                lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _extract_capnostream_metadata(path: str) -> list[str]:
        """Read the top section of the xlsx and keep all metadata lines before data headers."""
        try:
            raw = pd.read_excel(path, header=None, nrows=40)
        except Exception as e:  # noqa: BLE001
            return [f"<Failed to read capnostream metadata: {e}>"]

        lines: list[str] = []
        for _, row in raw.iterrows():
            values = [str(v).strip() for v in row.tolist() if not pd.isna(v) and str(v).strip()]
            if not values:
                continue
            if values[0] == "Date":
                break
            lines.append("\t".join(values))

        if not lines:
            return ["<No capnostream metadata rows found>"]
        return lines

    @staticmethod
    def _extract_motion_metadata(path: str) -> list[str]:
        """Read motion metadata/header lines up to the Frame/Time data header."""
        try:
            lines: list[str] = []
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for raw_line in f:
                    line = raw_line.rstrip("\n").rstrip("\r")
                    if not line:
                        continue
                    if line.startswith("Frame\tTime"):
                        break
                    lines.append(line)
            if not lines:
                return ["<No motion metadata rows found>"]
            return lines
        except Exception as e:  # noqa: BLE001
            return [f"<Failed to read motion metadata: {e}>"]
