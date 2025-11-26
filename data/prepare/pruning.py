from data.prepare import extraction
import os
import numpy as np

base_path = os.path.dirname(os.path.abspath(__file__))

def apply_pruning_filter(units: np.ndarray, constraints_hash: str) -> np.ndarray:
    path = os.path.join(base_path, "filters", f"{constraints_hash}.txt")

    if not os.path.exists(path):
        print(f"No pruning filter found for constraints key '{constraints_hash}'. Using all units.")
        return units
    
    included_indices = np.loadtxt(path, dtype=int)
    mask = np.zeros(len(units), dtype=bool)
    mask[included_indices] = True
    return units[mask]

def build_pruning_filter_file(files: str, constraints: extraction.ExtractionConstraints):
        key = constraints.hash()
        included_indices = np.array(sorted([int(i.replace(".png", "")) - 1 for i in os.listdir(files) if i.endswith(".png")]))
        np.savetxt(os.path.join(os.path.join(base_path, "filters"), f"{key}.txt"), included_indices, fmt="%d")

if __name__ == "__main__":
    build_pruning_filter_file(files=os.path.join(base_path, "out"),  constraints=extraction.ExtractionConstraints())