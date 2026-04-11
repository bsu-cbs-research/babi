from typing import TYPE_CHECKING
import os
import threading
from tkinter import messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

if TYPE_CHECKING:
    import pandas as pd
    from program.app import BABIDataAnalysisApp


class Screen(ttk.Frame):
    ID = "configuration"

    def __init__(self, parent: ttk.Frame, controller: "BABIDataAnalysisApp"):
        super().__init__(parent)
        self.controller = controller

        header_frame = ttk.Frame(self)
        header_frame.pack(fill=X, pady=(0, 20))
        header_frame.columnconfigure(1, weight=1)

        ttk.Label(
            header_frame, text="Configuration", font=("Helvetica", 14), anchor="center"
        ).grid(row=0, column=0, sticky=W)
        ttk.Button(
            header_frame,
            text="Back",
            command=lambda: controller.show_screen("start"),
            style=SECONDARY,
        ).grid(row=0, column=1, sticky=E, padx=(0, 5))
        ttk.Button(
            header_frame, text="Proceed", command=self._handle_proceed
        ).grid(row=0, column=2, sticky=E)

        # summary of the selected folder (populated in on_show)
        self.summary_label = ttk.Label(self, text="", justify="left")
        self.summary_label.pack(anchor="w", pady=(0, 10))

        # offset entry
        ttk.Label(self, text="Alignment offset (seconds):").pack(
            anchor="w", pady=(10, 0)
        )
        self.offset_var = ttk.StringVar(value="0")
        self.offset_entry = ttk.Entry(self, textvariable=self.offset_var)
        self.offset_entry.pack(fill=X, pady=5)
        ttk.Label(
            self,
            text=(
                "The capnostream timeline will be shifted by this many seconds "
                "before\nbeing aligned onto the motion timeline. Accepts integers "
                "or decimals."
            ),
            style=SECONDARY,
            justify="left",
        ).pack(anchor="w", pady=(2, 0))

        # redraw summary whenever the screen is shown
        self.bind("<Map>", lambda _evt: self._refresh_summary())

    def _refresh_summary(self) -> None:
        folder_path = self.controller.context.folder_path
        if not folder_path:
            self.summary_label.configure(text="(no folder selected)")
            return

        try:
            entries = os.listdir(folder_path)
        except OSError as e:
            self.summary_label.configure(text=f"Cannot read folder: {e}")
            return

        xlsx_files = [f for f in entries if f.lower().endswith(".xlsx")]
        tsv_files = [f for f in entries if f.lower().endswith(".tsv")]

        xlsx_display = xlsx_files[0] if xlsx_files else "(none)"
        self.summary_label.configure(
            text=(
                f"Folder: {folder_path}\n"
                f"Capnostream file: {xlsx_display}\n"
                f"Motion files: {len(tsv_files)}"
            )
        )

    def _handle_proceed(self) -> None:
        folder_path = self.controller.context.folder_path
        if not folder_path:
            messagebox.showerror("Error", "No folder selected.")
            return self.controller.show_screen("start")

        raw_offset = self.offset_var.get().strip()
        if not raw_offset:
            return messagebox.showerror("Error", "Offset cannot be empty.")
        try:
            offset = float(raw_offset)
        except ValueError:
            return messagebox.showerror(
                "Error",
                f"Offset must be an integer or decimal number, got {raw_offset!r}.",
            )

        self.controller.context.offset = offset

        # switch to progress screen BEFORE starting the worker so the label
        # is available for status updates.
        self.controller.show_screen("progress")
        progress_label = self.controller.context.progress_label
        if not progress_label:
            messagebox.showerror("Error", "Progress label not found in context.")
            return self.controller.show_screen("start")

        def _run_pipeline() -> None:
            try:
                from program.app import resource_path  # local import to avoid cycle
                import pipeline

                static_dir = resource_path(os.path.join("program", "static"))

                def _status(msg: str) -> None:
                    self.controller.queue.put(("status", msg))

                motion_df, capnostream_df = pipeline.execute(
                    folder_path,
                    offset=offset,
                    status_cb=_status,
                    static_dir=static_dir,
                )
                self.controller.queue.put(
                    ("success", (motion_df, capnostream_df))
                )
            except Exception as e:  # noqa: BLE001
                self.controller.queue.put(("error", str(e)))

        thread = threading.Thread(target=_run_pipeline, daemon=True)

        def _on_success(
            payload: tuple["pd.DataFrame", "pd.DataFrame"],
        ) -> None:
            motion_df, capnostream_df = payload
            self.controller.context.motion_df = motion_df
            self.controller.context.capnostream_df = capnostream_df
            self.controller.show_screen("complete")

        def _on_error(data: str) -> None:
            messagebox.showerror("Error", data)
            self.controller.show_screen("configuration")

        def _on_status(data: str) -> None:
            if progress_label:
                progress_label.configure(text=data)

        self.controller.run_thread(
            thread,
            {
                "success": _on_success,
                "error": _on_error,
                "status": _on_status,
            },
        )
