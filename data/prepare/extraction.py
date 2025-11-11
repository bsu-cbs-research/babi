from data import constants
import numpy as np

min_length = 0.4 * constants.capnostream_sampling_rate # minimum length of unit (also distance between plateaus) (s)
slope_window = 4 # windowed samples to calculate slope
plateau_slope_tolerance = 0.01 # within x unit(s) of zero-slope
min_amplitude = 16 # minimum amplitude between initial marker and unit max value (peak)
plateau_vertical_tolerance = 2 # within x unit(s) of previous plateau
max_units_from_zero = 5 # how far the bases can be from zero 
mean_max_amplitude_error_percentage = 0.7 # % error allowed between mean amplitude and unit max amplitude (ideally ~40%)
max_search_before_discard = 1.5 * constants.capnostream_sampling_rate # if no plateau found within this length, discard unit and find the next one (s)

def extract_unit_markers(signal: np.ndarray) -> list[tuple[int, int]]:
    """Extract 'breathing' units by capturing segments between low plateaus"""
    position = slope_window - 1
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
        sample = signal[position-slope_window:position]
        mean_amplitude = np.mean(sample)

        # amplitude of this sample is too high
        if mean_amplitude > min_amplitude: 
            constraint_failure_count["min_amplitude"] += 1
            continue
        
        slope = np.mean(np.diff(sample))
        
        # slope not close enough to zero
        if abs(slope) > plateau_slope_tolerance:
            constraint_failure_count["plateau_slope_tolerance"] += 1
            continue

        # marker doesn't exist yet or max search distance reached -> create without other checks
        if marker[0] == -1 or position - marker[0] > max_search_before_discard:
            marker = (position, mean_amplitude)
            continue
        
        # not within vertical tolerance of previous plateau
        if abs(mean_amplitude - marker[1]) > plateau_vertical_tolerance: 
            constraint_failure_count["plateau_vertical_tolerance"] += 1
            continue

        # not enough distance from previous marker
        if position - marker[0] < min_length: 
            constraint_failure_count["min_length"] += 1
            continue 

        unit_sample = signal[marker[0]:position]
        unit_max_amplitude, unit_mean_amplitude = np.max(unit_sample), np.mean(unit_sample)

        # max amplitude within unit not high enough
        if unit_max_amplitude < min_amplitude: 
            constraint_failure_count["unit_max_amplitude"] += 1
            continue 
        
        # mean and max amplitude too far
        if abs(unit_mean_amplitude - unit_max_amplitude) > unit_max_amplitude * mean_max_amplitude_error_percentage: 
            constraint_failure_count["mean_max_amplitude_error_percentage"] += 1
            continue

        if marker[1] > max_units_from_zero or mean_amplitude > max_units_from_zero:
            constraint_failure_count["max_units_from_zero"] += 1
            continue

        units.append((marker[0], position)) # store valid unit
        marker = (position, mean_amplitude) # update marker to current position
        # print(f"Position: {position}, Mean Amplitude: {mean_amplitude:.2f}, Slope: {slope:.4f}")

    for constraint, count in constraint_failure_count.items():
        print(f"Constraint '{constraint}' failures: {count}")

    return units