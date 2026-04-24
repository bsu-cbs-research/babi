import pandas as pd


def static(capnostream: pd.DataFrame, motion: pd.DataFrame, offset: int):
    capnostream["time"] = capnostream["time"] + pd.to_timedelta(offset, unit="s")
    labels_by_time = pd.Series(
        capnostream["co2_valid"].to_numpy(),
        index=pd.DatetimeIndex(capnostream["time"]),
    )
    motion_labels = (
        labels_by_time
        .reindex(motion.index, method="nearest", tolerance=pd.Timedelta("25ms"))
        .fillna(0)
        .astype(int)
    )
    motion["co2_valid"] = motion_labels.to_numpy()
    return motion, capnostream
