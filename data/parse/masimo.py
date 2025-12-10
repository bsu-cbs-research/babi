import os
import pandas as pd
from typing import cast
from data import constants

def parse(overwrite: bool = False):
    """Parse the raw Masimo data into a structured format that can be exported as a pickle file."""

    if not overwrite and os.path.exists(constants.parsed_data_paths["masimo"]):
        print(f"Parsed Masimo data already exists at {constants.parsed_data_paths['masimo']}")
        return
    
    raw_masimo_df = pd.read_excel(constants.raw_data_paths["masimo"], sheet_name="All Data")

    metadata: dict[str, str | list[str]] = {"Date": cast(str, raw_masimo_df.iloc[4, 0])}
    metadata["trial_names"] = []

    df = pd.DataFrame(columns=[cast(str, c).strip() for c in raw_masimo_df.iloc[1, :].values.tolist()][1:] + ["Trial Number"])
    data_start_row = 2  # data starts at row 3 (0-indexed), after the metadata rows

    current_trial_number = -1  # start with -1 to account for the first row being trial 0
    for i, row in raw_masimo_df.iloc[data_start_row:, :].iterrows():
        if all(pd.isna(row)):
            continue

        if type(row.iloc[0]) == str and all(pd.isna(row.iloc[1:])):
            current_trial_number += 1
            metadata["trial_names"].append(row.iloc[0])
            continue

        # ensure all values exist -> add row only if no NaN values are present
        if not any(pd.isna(row.values)):
            df.loc[len(df)] = row.values[1:].tolist() + [current_trial_number]

    df.attrs = df.attrs | metadata  # append metadata to the DataFrame attributes

    df.to_pickle(constants.parsed_data_paths["masimo"])
    print(f"Parsed Masimo data saved to {constants.parsed_data_paths['masimo']}")