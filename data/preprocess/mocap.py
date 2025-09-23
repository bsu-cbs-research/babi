from typing import cast
import os
import csv
import pandas as pd
from data import constants

def preprocess(overwrite: bool = False):
    """Process the raw MoCap data into a structured format that can be exported as a pickle file."""
    if not overwrite and os.path.exists(constants.preprocessed_data_paths["mocap"]):
        print(f"Preprocessed MoCap data already exists at {constants.preprocessed_data_paths['mocap']}")
        return

    metadata: dict[str, str | int | list[str]] = {}
    n_metadata_rows = 11  # number of metadata rows to read

    df = pd.DataFrame()  # dataframe to hold the processed data

    with open(constants.raw_data_paths["mocap"], 'r', newline='') as tsvfile:
        tsv_reader = csv.reader(tsvfile, delimiter='\t')
        for i, row in enumerate(tsv_reader):
            if i == n_metadata_rows: # skip row right after metadata (not needed)
                continue

            if i < n_metadata_rows: # read all metadata from the first n_metadata_rows
                information = row[1] if len(row[1:]) == 1 else row[1:]
                metadata[row[0]] = int(information) if type(information) == str and information.isdigit() else information

                if row[0] == "MARKER_NAMES": # create dataframe with makers if the first row contains marker names
                    df = pd.DataFrame(columns=["frame", "time"] + cast(list[str], metadata["MARKER_NAMES"]))
            else:
                frame_number, time = int(row[0]), float(row[1])
                marker_values = [(row[i], row[i+1], row[i+2]) for i in range(0, len(row[2:]), 3)]
                df.loc[len(df)] = [frame_number, time] + marker_values

    df.attrs = df.attrs | metadata
    df.to_pickle(constants.preprocessed_data_paths["mocap"])
    print(f"Preprocessed MoCap data saved to {constants.preprocessed_data_paths['mocap']}")