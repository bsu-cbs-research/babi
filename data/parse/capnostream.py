from typing import cast
import os
import pandas as pd
import json
from data import constants

def _compute_similar_group(trial_name: str):
    """Compute the most similar trial group to the given trial name based on substring matches."""
    scores: dict[str, int] = {}

    # score each group based on number of substring matches
    for group, substrings in constants.capnostream_group_similarity_mappings.items():
        scores[group] = sum(1 for match in substrings if match in trial_name)

    # get the group with the highest score
    return max(scores, key=scores.get) # type: ignore

def parse_preprocessed_local(file: str, overwrite: bool = False, export: bool = False):
    """Parse a single raw capnostream file into predictable pkl format"""
    identifier = file.split(" Capnostream Data")[0] # extract the unique identifier from the filename
    file_path = os.path.normpath(os.path.join(constants.raw_data_paths["capnostream"], file))
    parsed_data_path = os.path.join(constants.parsed_data_dir, "capnostream", f"{identifier}.pkl")
    print(f"[{identifier}] Parsing Capnostream data from {file_path}")

    if not overwrite and os.path.exists(parsed_data_path):
        print(f"Parsed Capnostream data for {identifier} already exists at {parsed_data_path}")
        df = pd.read_pickle(parsed_data_path)
        return identifier, df
    
    raw_capnostream_df = pd.read_excel(file_path) # default to reading the first sheet
    metadata: dict[str, str | list[str]] = {r[0]: r[1] for r in raw_capnostream_df.iloc[:3, :2].values} | {"Date": raw_capnostream_df.iloc[8, 0]}

    df = pd.DataFrame(columns=["time", "co2_wave", "et_co2", "rr", "group_index"])
    data_start_row = 6 # data starts at row 6 (0-indexed), after the metadata rows

    current_group_index: int | None = None
    for _, row in raw_capnostream_df.iloc[data_start_row:, :].iterrows():
        if type(row.iloc[0]) == str and all(pd.isna(row.iloc[1:])): # all other rows are empty -> must have trial start
            trial_name = cast(str, row.iloc[0]).split("-")[-1].strip().lower()
            similar_group = _compute_similar_group(trial_name) # find the most similar group to the current trial name
            current_group_index = constants.capnostream_groups.index(similar_group)
            continue

        # only add row if all values (except the first column) are present (not NaN)
        if not any(pd.isna(row.values)):
            data = row.values[1:].tolist()  # append the current trial number to the data

            # check if the data contains "--" which indicates a missing value
            if any([d == "--" or d == 0 for d in data[:]]): # todo: handle data with zeros
                continue

            df.loc[len(df)] = data + [current_group_index]  # add the data to the DataFrame

    # convert necessary columns to appropriate types
    df["group_index"] = df["group_index"].astype(int)
    df["et_co2"] = df["et_co2"].astype(int)
    df["rr"] = df["rr"].astype(int)

    df.attrs = df.attrs | metadata # append metadata to the DataFrame attributes

    if export: # only export if specified
        # create directory if it doesn't exist
        if not os.path.exists(os.path.dirname(parsed_data_path)):
            os.makedirs(os.path.dirname(parsed_data_path))
        df.to_pickle(parsed_data_path)
        print(f"[{identifier}] Parsed Capnostream data saved to {parsed_data_path}")

    print(f"[{identifier}] Parsed {len(df)} rows of Capnostream data")

    return identifier, df

def parse_preprocessed_locals(overwrite: bool = False, export: bool = False):
    """Combine all individual parsed capnostream files into a single pkl file."""
    
    if not overwrite and os.path.exists(constants.parsed_data_paths["capnostream"]):
        print(f"Combined parsed Capnostream data already exists at {constants.parsed_data_paths['capnostream']}")
        return
    
    print("Parsing all Capnostream files...")

    combined_df = pd.DataFrame(columns=["baby_index", "time", "co2_wave", "et_co2", "rr", "group_index"])
    combined_metadata: dict[str, dict[str, str | list[str]]] = { } # ids associated with entry metadata
    current_baby_identifier_index = 0
    for path in [f for f in os.listdir(os.path.join(constants.raw_data_dir, "capnostream")) if f.endswith(".xlsx")]:
        identifier, df = parse_preprocessed_local(path, overwrite=overwrite, export=export) # ensure the individual file is parsed
        df.insert(0, "baby_index", current_baby_identifier_index)
        combined_metadata[identifier] = cast(dict, df.attrs) | {"index": current_baby_identifier_index}
        combined_df = pd.concat([combined_df, df], ignore_index=True)
        current_baby_identifier_index += 1

    combined_df.attrs = {} | combined_metadata
    combined_df.to_pickle(constants.parsed_data_paths["capnostream"])

    print(f"Combined parsed Capnostream data saved to {constants.parsed_data_paths['capnostream']}")

def export_label_studio(overwrite: bool = False):
    """Prepare the parsed capnostream data for import into Label Studio."""

    if not overwrite and os.path.exists(constants.parsed_data_paths["capnostream-labelstudio"]):
        print(f"Label Studio Capnostream data already exists at {constants.parsed_data_paths['capnostream-labelstudio']}")
        return
    
    df: pd.DataFrame = pd.read_pickle(constants.parsed_data_paths["capnostream"])
    groups = df.groupby(["baby_index", "group_index"])

    tasks: list[dict] = []
    for (baby_index, group_index), group in groups:
        co2_wave = group["co2_wave"].values.tolist()
        task = {
            "id": f"baby-{baby_index}-group-{group_index}",
            "data": {"ts":{"time": [round(i * 0.05, 2) for i in range(len(co2_wave))], "co2": co2_wave}}, 
            "meta": {"baby_index": int(baby_index), "group_index": int(group_index)}
        }
        tasks.append(task)

    with open(constants.parsed_data_paths["capnostream-labelstudio"], "w") as f:
        json.dump(tasks, f)

    print(f"Label Studio Capnostream data saved to {constants.parsed_data_paths['capnostream-labelstudio']}")
        
if __name__ == "__main__":
    parse_preprocessed_locals(overwrite=False)
    export_label_studio(overwrite=False)