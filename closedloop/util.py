from typing import Tuple, List

import numpy as np


def find_value_indices(signal: np.ndarray, target_value) -> List[Tuple[int, int]]:
    """
    Get the couple of indices of the signal having as value the target value

    :param signal: signal to analyze
    :param target_value: value we want to find in the signal
    :return: list of couples of indices pointing to the start and the end of a continuous signal with the value
    """
    indices = np.where(signal == target_value)[0]

    nearby_indices = []

    start = 0

    for i in range(len(indices) - 1):
        if indices[i + 1] - indices[i] > 1:
            nearby_indices.append((indices[start], indices[i] + 1))
            start = i + 1

    return nearby_indices
