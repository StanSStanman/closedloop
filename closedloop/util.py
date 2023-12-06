import numpy as np

def find_value_indices(arr, target_value):
    indices = np.where(arr == target_value)[0]

    nearby_indices = []

    start = 0

    for i in range(len(indices)-1):
        if indices[i+1] - indices[i] > 1:
            nearby_indices.append((indices[start], indices[i] + 1))
            start = i + 1

    return nearby_indices