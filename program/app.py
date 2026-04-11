from typing import Optional, Callable, Literal, Any, cast, TYPE_CHECKING
from dataclasses import dataclass
import os
import sys
import threading
import queue
from tkinter import PhotoImage
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from program.screens import start, configuration, progress, complete

if TYPE_CHECKING:
    import pandas as pd

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

type QueueTags = Literal["success", "error", "status"]
type QueueItem = tuple[QueueTags, Any]
type QueueCallback = Optional[Callable[[Any], None]]
type QueueCallbacks = dict[QueueTags, QueueCallback]

type ScreenName = Literal["start", "configuration", "progress", "complete"]

@dataclass
class ProgramContext:
    folder_path: Optional[str] = None
    offset: Optional[float] = None
    motion_df: Optional["pd.DataFrame"] = None
    capnostream_df: Optional["pd.DataFrame"] = None
    progress_label: Optional[ttk.Label] = None

    def reset(self):
        self.folder_path = None
        self.offset = None
        self.motion_df = None
        self.capnostream_df = None
        self.progress_label = None

class BABIDataAnalysisApp:
    def __init__(self, root: ttk.Window) -> None:
        self.root: ttk.Window = root
        self.root.title("BABI Data Analysis")
        
        # window geometry and centering
        window_width: int = 500
        window_height: int = 600
        screen_width: int = self.root.winfo_screenwidth()
        screen_height: int = self.root.winfo_screenheight()
        center_x: int = int(screen_width / 2 - window_width / 2)
        center_y: int = int(screen_height / 2 - window_height / 2)
        self.root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
        self.root.resizable(True, True)

        self.container: ttk.Frame = ttk.Frame(self.root, padding="20")
        self.container.pack(fill=BOTH, expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        # context
        self.queue: queue.Queue[QueueItem] = queue.Queue()
        self.context = ProgramContext()

        # screens
        self.screens: dict[ScreenName, ttk.Frame] = {}
        screens = (start.Screen, configuration.Screen, progress.Screen, complete.Screen)

        for Screen in screens:
            name = cast(ScreenName, Screen.ID)
            screen: ttk.Frame = Screen(parent=self.container, controller=self)
            self.screens[name] = screen
            screen.grid(row=0, column=0, sticky="nsew")

        self.active_screen: Optional[ScreenName] = None
        self.show_screen("start")

    def show_screen(self, name: ScreenName) -> None:
        """Lifts the requested frame to the top."""
        screen: ttk.Frame = self.screens[name]
        screen.tkraise()
        self.active_screen = name

    def run_thread(self, thread: threading.Thread, callbacks: QueueCallbacks) -> None:
        thread.start()
        self.root.after(100, lambda: self.poll_queue(callbacks))

    def poll_queue(self, callbacks: QueueCallbacks) -> None:
        try:
            item = self.queue.get_nowait()
            if item:
                tag, data = item
                callback = callbacks.get(tag)
                if callback: callback(data)
                if tag == "success" or tag == "error": return
        except queue.Empty: pass
        
        self.root.after(100, lambda: self.poll_queue(callbacks))

    def on_closing(self):
        self.root.destroy()

if __name__ == "__main__":
    root = ttk.Window(themename="darkly")
    app = BABIDataAnalysisApp(root)
    photo = PhotoImage(file=resource_path("program/static/program-icon@1x.png"))
    root.iconphoto(True, photo) 
    root.protocol("WM_DELETE_WINDOW", app.on_closing) 

    root.mainloop()
