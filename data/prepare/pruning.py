import os
import hashlib
import numpy as np
import matplotlib.pyplot as plt

base_path = os.path.dirname(os.path.abspath(__file__))
raw_plots_dir = os.path.join(base_path, "pruning", "raw")
filters_dir = os.path.join(base_path, "pruning", "filters")

def hash_array(arr):
    """Generates a SHA-256 hash for a numpy array, considering both its data and shape."""
    arr_contiguous = np.ascontiguousarray(arr)
    data_bytes = arr_contiguous.tobytes()
    shape_bytes = str(arr.shape).encode('utf-8')
    hasher = hashlib.sha256()
    hasher.update(data_bytes)
    hasher.update(shape_bytes)
    return hasher.hexdigest()

def apply_mask(signals: np.ndarray, key: str) -> np.ndarray:
    """Applies a pruning filter based on signal characteristics."""
    path = os.path.join(filters_dir, f"{key}.txt")
    if not mask_exists(key):
        print(f"No pruning filter found for signal key '{key}'. Using all signals.")
        return signals
    
    included_indices = np.loadtxt(path, dtype=int)
    mask = np.zeros(len(signals), dtype=bool)
    mask[included_indices] = True
    return signals[mask]

def build_mask(key: str):
    """Creates a index mask based on existing plots and saves it."""
    if not os.path.exists(os.path.join(raw_plots_dir, key)):
        raise FileNotFoundError(f"No raw plots found for key '{key}' at '{os.path.join(raw_plots_dir, key)}'")
    
    if not os.path.exists(filters_dir): os.makedirs(filters_dir, exist_ok=True)

    files = os.listdir(os.path.join(raw_plots_dir, key))
    included_indices = np.array(sorted([int(i.replace(".png", "")) - 1 for i in files if i.endswith(".png")]))
    np.savetxt(os.path.join(filters_dir, f"{key}.txt"), included_indices, fmt="%d")

def export_plots(signals: np.ndarray, key: str, max: int | None = None):
    """Exports signals as plots for manual pruning"""
    output_dir = os.path.join(raw_plots_dir, key)

    if os.path.exists(output_dir):
        print(f"Plots for key '{key}' already exist at '{output_dir}'. Skipping export.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for i, signal in enumerate(signals):
        if max is not None and i >= max:
            break

        plt.figure(figsize=(12, 4))
        plt.plot(signal, color='blue')
        plt.title(f"Signal {i+1}")
        plt.xlabel("Time (samples)")
        plt.ylabel("Normalized CO2 Level")
        plt.grid()
        plt.savefig(os.path.join(output_dir, f"{i+1}.png"))
        plt.close()

def mask_exists(key: str) -> bool:
    """Checks if a pruning filter file exists for the given key."""
    path = os.path.join(filters_dir, f"{key}.txt")
    return os.path.exists(path)