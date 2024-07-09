from typing import Tuple, List
import numpy as np


def find_value_indices(signal: np.ndarray, target_values) -> List[Tuple[int, int]]:
    """
    Get the couple of indices of the signal having as value the target value

    :param signal: signal to analyze
    :param target_value: value we want to find in the signal
    :return: list of couples of indices pointing to the start and the end of a continuous signal with the value
    """
    indices = np.where(signal == target_values)[0]

    nearby_indices = []

    start = 0

    for i in range(len(indices) - 1):
        if indices[i + 1] - indices[i] > 1:
            nearby_indices.append((indices[start], indices[i] + 1))
            start = i + 1

    return nearby_indices


def crop_events(events, sfreq, tmin=None, tmax=None):
    if tmax is not None:
        if len(np.where(events[:, 0] > tmax * sfreq)[0]) != 0:
            idx = np.where(events[:, 0] > tmax * sfreq)[0][0]
            events = events[:idx + 1, :]
            events[-1, 0] = tmax * sfreq
    if tmin is not None:
        if len(np.where(events[:, 0] <= tmin*sfreq)[0]) != 0:
            idx = np.where(events[:, 0] <= tmin*sfreq)[0][-1]
            events = events[idx:, :]
            events[:, 0] = events[:, 0] - (tmin * sfreq)
            events[0, 0] = 0
    
    return events