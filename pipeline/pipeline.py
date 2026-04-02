import labeling, alignment, data
from pipeline.containers import CapnostreamContainer, MotionContainer
import numpy as np

class ProcessingPipeline:
    def __init__(self) -> None:
        self.temp = "/tmp"
        self.validated = False
        self.processed = False
        self.labeled = False
        self.aligned = False
        self.capnostream = CapnostreamContainer()
        self.motion = MotionContainer()
        pass
    
    def _include_file(self, file: str):
        if file.endswith(".xlsx"): self.capnostream.path = file
        elif file.endswith(".tsv"): self.motion.paths.add(file)
        else:  raise ValueError("Unsupported file type: {}".format(file))
    
    def _validate(self):
        if not self.capnostream.path: raise ValueError("Capnostream data is missing")
        if len(self.motion.paths) == 0: raise ValueError("Motion data is missing")
        # validate the data (check for missing values, check for correct format, etc.)
        data.validating.capnostream(self.capnostream.path)
        data.validating.motion(self.motion.paths)
        self.validated = True
    
    def _process(self):
        if not self.validated: raise ValueError("Data is not validated")
        assert self.capnostream.path, "Capnostream data path is not set"
        assert self.motion.paths, "Motion data paths are not set"
        # process the data (e.g. filter, normalize, etc.)
        data.processing.capnostream(self.capnostream.path)
        data.processing.motion(self.motion.paths)
        self.processed = True
    
    def _label(self):
        if not self.processed: raise ValueError("Data is not processed")
        assert self.capnostream.path, "Capnostream data path is not set"

        # perform capno inference (e.g. calculate respiratory rate, tidal volume, etc.)
        self.labeled = labeling.label(np.array([]))
    
    def _align(self):
        if not self.processed: raise ValueError("Data is not processed")
        assert self.motion.paths, "Motion data paths are not set"
        self.aligned = alignment.static()

    def _output(self):
        return self
    
    def execute(self, pkg: str, offset: int = 0):
        files = os.listdir(pkg)
        self._include_file(os.path.join(pkg, next((f for f in files if f.endswith(".xlsx")), "")))
        for m in [f for f in files if f.endswith(".tsv")]: self._include_file(os.path.join(pkg, m))

        self.offset = offset

        self._validate()
        self._process()
        self._label()
        self._align()

        return self._output()

        # self.include_file("capnostream_data.xlsx").include_file("motion_data.tsv").include_offset(offset).validate().process().label().align().output()
    
    def reset(self):
        self.capnostream = CapnostreamContainer()
        self.motion = MotionContainer()
        self.offset = 0

if __name__ == "__main__":   
    import os 
    pipeline = ProcessingPipeline()
    pkg = os.path.join("data", "testing", "p1", "co2.xlsx")
    pipeline.execute(pkg, offset=360)