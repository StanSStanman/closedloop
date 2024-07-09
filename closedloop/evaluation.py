import math
import numpy as np


def get_online_detection_evaluation(offline_phase_signal: np.ndarray, input_online_times: np.ndarray):
    """
    Get evaluation of online detection with respect to an offline detection

    :param offline_phase_signal: signal of slow waves phases from offline detection
    :param input_online_times: input times detected from the online algorithm
    :return: accuracy and normalized frequencies
    """

    phases_times = offline_phase_signal[input_online_times]

    not_nan_phases_times = phases_times[~np.isnan(phases_times)]
    accuracy = len(not_nan_phases_times) / len(input_online_times)

    bins = 16
    bin_width = 2 * math.pi / bins

    frequencies = np.full(bins, 0)

    for index in range(bins):
        bin = not_nan_phases_times[(not_nan_phases_times >= index * bin_width)
                                   & (not_nan_phases_times < (index + 1) * bin_width)]
        frequencies[index] = len(bin)

    normalized_frequencies = frequencies / len(not_nan_phases_times)
    return accuracy, normalized_frequencies
