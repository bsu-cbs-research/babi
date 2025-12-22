from typing import TYPE_CHECKING
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

if TYPE_CHECKING: from program.app import BABIDataAnalysisApp

class Screen(ttk.Frame):
  ID = "progress"

  def __init__(self, parent: ttk.Frame, controller: "BABIDataAnalysisApp"):
    super().__init__(parent)
    self.controller = controller

    label = ttk.Label(self, text="Loading model...")
    label.pack(pady=(80, 0))

    progress = ttk.Progressbar(self, mode='indeterminate')
    progress.pack(pady=(30, 0), fill=X)
    progress.start()

    self.controller.context.progress_label = label