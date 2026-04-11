import os
import labeling, alignment, data

CO2_COLUMN_INDEX = 2

def execute(pkg: str, offset: int = 0):
    capnostream_path = os.path.join(pkg, next((f for f in os.listdir(pkg) if f.endswith(".xlsx")), ""))
    motion_paths = set([os.path.join(pkg, f) for f in os.listdir(pkg) if f.endswith(".tsv")])

    # parsing
    assert capnostream_path, "Capnostream data path is not set"
    capnostream_data = data.parsing.capnostream(capnostream_path)
    motion_data = data.parsing.motion(motion_paths)

    # labeling
    assert capnostream_data is not None, "Capnostream data is not set"
    labels = labeling.label(capnostream_data.iloc[:, CO2_COLUMN_INDEX].to_numpy())
    print(len(labels), len(capnostream_data))

    capnostream_data["co2_valid"] = labels

    # alignment
    assert labels is not None, "Labels are not set"
    assert capnostream_data is not None, "Capnostream data is not set"
    assert motion_data is not None, "Motion data is not set"
    labeled_motion, capno2 = alignment.static(capnostream_data, motion_data, offset)

    # # output
    return capnostream_data, labeled_motion, capno2

if __name__ == "__main__":   
    execute(os.path.join("data", "testing", "p1"), offset=360)
