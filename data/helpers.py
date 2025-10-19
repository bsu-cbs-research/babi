import json
from data import constants

def load_capnostream_annotated_data() -> list[list[int]]:
    """Loads the annotated capnostream data in a practical format"""
    with open(constants.preprocessed_data_paths["capnostream-labelstudio-annotated"], "r") as f:
        raw = json.load(f)

    data: list[list[int]] = []

    for entry in raw:
        waveform = entry["data"]["ts"]["co2"]
        for annotation in entry["annotations"]:
            for result in annotation["result"]:
                value = result["value"]
                start, end = (int(i * constants.capnostream_sampling_rate) for i in (value["start"], value["end"]))
                data.append(waveform[start:end])

    return data

if __name__ == "__main__":
    annotated_data = load_capnostream_annotated_data()
    print(f"Loaded {len(annotated_data)} annotated segments from capnostream data")