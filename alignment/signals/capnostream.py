from data import constants
import pandas as pd
import os
import numpy as np
from alignment.signals import utils, mocap

sr = 20

def from_file(name: str):
    file_path = os.path.join(constants.raw_data_dir, "package", name)
    raw_capnostream_df = pd.read_excel(file_path)
    data_start_row = 6
    waveform_extracted_value = raw_capnostream_df.iloc[data_start_row:, 2].reset_index(drop=True).map(lambda x: float(x) if isinstance(x, (int, float)) else -1)
    co2_not_available = raw_capnostream_df.iloc[data_start_row:, 16].reset_index(drop=True).to_numpy()
    co2_range = np.where(co2_not_available == 0)[0]
    raw_waveform = waveform_extracted_value.to_numpy()[co2_range]
    waveform = utils.upsample(raw_waveform, sr, mocap.sr)
    waveform = (waveform - waveform.mean()) / waveform.std()
    return waveform - waveform.mean()
