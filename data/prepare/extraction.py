from dataclasses import dataclass
import numpy as np
from data import constants
import hashlib

@dataclass
class ExtractionConstraints:
    """
    min_length: minimum length of unit (also distance between plateaus) in samples\n
    slope_window: windowed samples to calculate slope\n
    plateau_slope_tolerance: tolerance for slope to be considered a plateau (close to zero)\n
    min_amplitude: minimum amplitude between initial marker and unit max value (peak)\n
    plateau_vertical_tolerance: allowed vertical difference between consecutive plateaus\n
    max_units_from_zero: how far the bases can be from zero\n
    mean_max_amplitude_error_percentage: allowed % error between mean amplitude and unit max amplitude\n
    max_search_before_discard: if no plateau found within this length, discard unit and find the next one (samples)
    """
    min_length: int = int(0.4 * constants.capnostream_sampling_rate)
    slope_window: int = 4
    plateau_slope_tolerance: float = 0.01
    min_amplitude: int = 16
    plateau_vertical_tolerance: int = 2
    max_units_from_zero: int = 5
    mean_max_amplitude_error_percentage: float = 0.7
    max_search_before_discard: int = int(1.5 * constants.capnostream_sampling_rate)

    def hash(self) -> str:
        formatted_string = f"{self.min_length}_{self.slope_window}_{self.plateau_slope_tolerance}_{self.min_amplitude}_{self.plateau_vertical_tolerance}_{self.max_units_from_zero}_{self.mean_max_amplitude_error_percentage}_{self.max_search_before_discard}"
        return hashlib.sha256(formatted_string.encode()).hexdigest()[:10]

def extract_unit_markers(signal: np.ndarray, constraints = ExtractionConstraints()) -> list[tuple[int, int]]:
    """Extract 'breathing' units by capturing segments between low plateaus"""
    position = constraints.slope_window - 1
    units: list[tuple[int, int]] = []
    constraint_failure_count = {
        "min_amplitude": 0,
        "plateau_slope_tolerance": 0,
        "plateau_vertical_tolerance": 0,
        "min_length": 0,
        "max_units_from_zero": 0,
        "unit_max_amplitude": 0,
        "mean_max_amplitude_error_percentage": 0,
    }

    marker = (-1, 0) # (index, mean amplitude)
    while position < len(signal):
        position += 1
        sample = signal[position-constraints.slope_window:position]
        mean_amplitude = np.mean(sample)

        # amplitude of this sample is too high
        if mean_amplitude > constraints.min_amplitude: 
            constraint_failure_count["min_amplitude"] += 1
            continue
        
        slope = np.mean(np.diff(sample))
        
        # slope not close enough to zero
        if abs(slope) > constraints.plateau_slope_tolerance:
            constraint_failure_count["plateau_slope_tolerance"] += 1
            continue

        # marker doesn't exist yet or max search distance reached -> create without other checks
        if marker[0] == -1 or position - marker[0] > constraints.max_search_before_discard:
            marker = (position, mean_amplitude)
            continue
        
        # not within vertical tolerance of previous plateau
        if abs(mean_amplitude - marker[1]) > constraints.plateau_vertical_tolerance: 
            constraint_failure_count["plateau_vertical_tolerance"] += 1
            continue

        # not enough distance from previous marker
        if position - marker[0] < constraints.min_length: 
            constraint_failure_count["min_length"] += 1
            continue 

        unit_sample = signal[marker[0]:position]
        unit_max_amplitude, unit_mean_amplitude = np.max(unit_sample), np.mean(unit_sample)

        # max amplitude within unit not high enough
        if unit_max_amplitude < constraints.min_amplitude: 
            constraint_failure_count["unit_max_amplitude"] += 1
            continue 
        
        # mean and max amplitude too far
        if abs(unit_mean_amplitude - unit_max_amplitude) > unit_max_amplitude * constraints.mean_max_amplitude_error_percentage: 
            constraint_failure_count["mean_max_amplitude_error_percentage"] += 1
            continue

        if marker[1] > constraints.max_units_from_zero or mean_amplitude > constraints.max_units_from_zero:
            constraint_failure_count["max_units_from_zero"] += 1
            continue

        units.append((marker[0], position)) # store valid unit
        marker = (position, mean_amplitude) # update marker to current position
        # print(f"Position: {position}, Mean Amplitude: {mean_amplitude:.2f}, Slope: {slope:.4f}")

    for constraint, count in constraint_failure_count.items():
        print(f"Constraint '{constraint}' failures: {count}")


    mean_unit_length = np.mean([end-start for start, end in units])
    max_unit_length = np.max([end-start for start, end in units])
    min_unit_length = np.min([end-start for start, end in units])
    total_unit_length = np.sum([end-start for start, end in units])

    print(f"Min unit length: {min_unit_length/constants.capnostream_sampling_rate:.2f} seconds.")
    print(f"Max unit length: {max_unit_length/constants.capnostream_sampling_rate:.2f} seconds.")
    print(f"Mean unit length: {mean_unit_length/constants.capnostream_sampling_rate:.2f} seconds.")
    print(f"Total unit length: {total_unit_length/constants.capnostream_sampling_rate/60:.2f} minutes.")

    return units

    