from typing import cast
import os
import csv
import pandas as pd
from data import constants

def parse(overwrite: bool = False, with_tuples: bool = False) -> None:
    """Parse the raw MoCap data into a structured format that can be exported as a pickle file."""
    parsed_path = constants.parsed_data_paths[with_tuples and "mocap-tuples" or "mocap"]
    if not overwrite and os.path.exists(parsed_path):
        print(f"Parsed MoCap data already exists at {parsed_path}")
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
                if row[0] == "MARKER_NAMES": # create dataframe with makers if the first column contains marker names
                    columns = ["frame", "time"]
                    marker_names = cast(list[str], metadata["MARKER_NAMES"])

                    if with_tuples: # each marker stores value as tuple (x, y, z)
                        columns.extend(marker_names) 
                    else: # each marker stores value as separate x, y, z columns
                        for name in marker_names:
                            for coord in ['x', 'y', 'z']:
                                columns.append(f"{name}_{coord}")

                    df = pd.DataFrame(columns=columns)
            else:
                if i in [1034, 1035, 1036]: continue # skip corrupted data rows (todo: generalize to other datasets )
                frame_number, time = int(row[0]), float(row[1])
                marker_values = [(float(row[i]), float(row[i+1]), float(row[i+2])) for i in range(2, len(row), 3)] if with_tuples else [float(row[i]) for i in range(2, len(row))]
                df.loc[len(df)] = [frame_number, time] + marker_values
                    

    df.attrs = df.attrs | metadata
    df.to_pickle(parsed_path)
    print(f"Parsed MoCap data saved to {parsed_path}")


if __name__ == "__main__":
    parse(overwrite=True)
    parse(overwrite=True, with_tuples=True)