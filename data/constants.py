import os

raw_data_dir = "data/raw/"

raw_data_paths = {k: os.path.join(raw_data_dir, v) for k, v in {
    "mocap": "45_Head 1 MoCap Data.tsv",
    "capnostream": "capnostream",
    "masimo": "BB-003-2 Masimo Data.xlsx",
    "program": "program",
}.items()}

parsed_data_dir = "data/out/parsed/"

parsed_data_paths = {k: os.path.join(parsed_data_dir, v) for k, v in {
    "mocap": "mocap.pkl",
    "mocap-tuples": "mocap-tuples.pkl",
    "capnostream": "capnostream.pkl",
    "capnostream-labelstudio": "labelstudio/capnostream.json",
    "capnostream-labelstudio-annotated": "labelstudio/capnostream-annotated.json",
    "waveform": "waveform.pkl",
    "masimo": "masimo.pkl"
}.items()}

capnostream_group_similarity_mappings: dict[str, list[str]] = {k: v + k.split() for k,v in {
    "flat supine": ["fs"],
    "head neck flexion supine": ["hn", "head/neck"],
    "head pelvis flexion supine": ["hp"],
    "thoracic flexion supine": ["t"],
    "cs": [""],
}.items()} # data categories to group disparate trial names: (category name, substrings)
capnostream_groups = list(capnostream_group_similarity_mappings.keys())