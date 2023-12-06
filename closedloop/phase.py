import math

import numpy as np
import pandas as pd


def convert_sec_to_timestamp(sec: float, sf: int) -> int:
    """
    Converts seconds to sample timestamp

    :param sec: seconds
    :param sf: sampling frequency
    :return: corresponding timestamp
    """
    return int(sec * sf)


def linear_interpolation(length: int, start_value: float, end_value: float) -> np.ndarray:
    """
    Creates a signal that linearly interpolates from a start to an end value

    :param length: length of the signal
    :param start_value: starting value
    :param end_value: ending value
    :return: signal of linear interpolation
    """
    return np.linspace(start_value, end_value, length)


def phase_signal(length: int, sf: int, events: pd.DataFrame) -> np.ndarray:
    """
    Creates a signal representing the phases of the detected slow waves

    :param length: length of the signal
    :param sf: sampling frequency of the signal
    :param events: dataframe of detected slow waves information
    :return: signal representing the phase of the detected slow waves
    """

    signal = np.full(length, np.nan)

    for row in range(events.shape[0]):
        start_timestamp = convert_sec_to_timestamp(events['Start'][row], sf)
        neg_peak_timestamp = convert_sec_to_timestamp(events['NegPeak'][row], sf)
        mid_timestamp = convert_sec_to_timestamp(events['MidCrossing'][row], sf)
        pos_peak_timestamp = convert_sec_to_timestamp(events['PosPeak'][row], sf)
        end_timestamp = convert_sec_to_timestamp(events['End'][row], sf)
        signal[start_timestamp:neg_peak_timestamp] = \
            linear_interpolation(neg_peak_timestamp - start_timestamp, 0, math.pi / 2)
        signal[neg_peak_timestamp:mid_timestamp] = \
            linear_interpolation(mid_timestamp - neg_peak_timestamp, math.pi / 2, math.pi)
        signal[mid_timestamp:pos_peak_timestamp] = \
            linear_interpolation(pos_peak_timestamp - mid_timestamp, math.pi, 3 / 2 * math.pi)
        signal[pos_peak_timestamp:end_timestamp] = \
            linear_interpolation(end_timestamp - pos_peak_timestamp, 3 / 2 * math.pi, 2 * math.pi)

    return signal


# Example
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import seaborn as sns

    START = 8445000
    END = 8445000 + 15000
    SF = 500
    # Define the data
    data = [
        [6.72600, 6.99200, 7.26200, 7.45800, 8.27800, 1.55200, -79.71397, 35.53945, 115.25342, 426.86452, 0.64433, 3,
         'CHAN000', 0],
        [8.27800, 8.62200, 9.07400, 9.28000, 10.15600, 1.87800, -106.93591, 34.94352, 141.87943, 313.89254, 0.53248, 3,
         'CHAN000', 0],
        [10.15600, 10.40400, 11.10800, 11.39600, 12.18400, 2.02800, -108.60267, 31.98732, 140.58999, 199.70169, 0.49310,
         3, 'CHAN000', 0]
    ]

    # Define column names
    columns = ['Start', 'NegPeak', 'MidCrossing', 'PosPeak', 'End', 'Duration', 'ValNegPeak', 'ValPosPeak', 'PTP',
               'Slope', 'Frequency', 'Stage', 'Channel', 'IdxChannel']

    # Create a DataFrame
    events = pd.DataFrame(data, columns=columns)

    signal = phase_signal(END - START, SF, events)

    times = np.arange(signal.size) / SF

    # Plot the signal
    fig, ax = plt.subplots(1, 1, figsize=(16, 4))
    plt.plot(times, signal, lw=1.5, color='k')
    plt.xlim([times.min(), times.max()])
    sns.despine()
    plt.show()
