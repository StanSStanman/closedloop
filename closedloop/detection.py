import math
import re

import mne
import pandas as pd
import yasa
from streamer_class import data_streamer
from yasa import sw_detect
import numpy as np

from phase import convert_sec_to_timestamp


def correct_slow_waves(filtered_signal: np.ndarray, events: pd.DataFrame, sf: int) -> pd.DataFrame:
    """
    This function is used to convert the detection of slow waves by yasa in order to point towards the real
    positive and negative peaks in the wave

    :param filtered_signal: the signal filtered
    :param events: the dataframe containing the information about slow waves detected by yasa
    :param sf: sampling frequency
    :return: corrected dataframe of slow waves info
    """
    results: pd.DataFrame = events.copy()
    for row in range(events.shape[0]):
        start_timestamp = convert_sec_to_timestamp(events['Start'][row], sf)
        end_timestamp = convert_sec_to_timestamp(events['End'][row], sf)

        interest_signal = filtered_signal[start_timestamp:end_timestamp]
        maximum = interest_signal.max()
        maximum_index = interest_signal.argmax()
        minimum = interest_signal.min()
        minimum_index = interest_signal.argmin()

        results['NegPeak'][row] = (minimum_index + start_timestamp) / sf
        results['ValNegPeak'][row] = minimum
        results['PosPeak'][row] = (maximum_index + start_timestamp) / sf
        results['ValPosPeak'][row] = maximum
        results['PTP'][row] = maximum - minimum

    return results


def get_slow_waves(signal: np.ndarray, hypno: np.ndarray, sf: int) -> yasa.SWResults:
    """
    Get the slow waves detection from yasa with parameters predefined
    :param signal: the raw eeg signal
    :param hypno: the signal of the hypnogram
    :param sf: sampling frequency
    :return: structure containing yasa slow wave detection information
    """
    sw = sw_detect(signal, sf, hypno=hypno, include=(2, 3), freq_sw=(0.3, 1.5),
                   dur_neg=(0.3, 1.5), dur_pos=(0.1, 1.5), amp_neg=(40, 200),
                   amp_pos=(10, 150), amp_ptp=(75, 350), coupling=False,
                   remove_outliers=False, verbose=False)

    return sw


# Example
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    RAW_DATA_FILE = '../resources/n1_raw.fif'
    RAW_EVE_FILE = '../resources/n1_eve.fif'
    SF = 500

    START = 0
    END = 17310000

    CHANNELS = ['F4-C4', 'C4-A1']

    raw = mne.io.read_raw_fif(RAW_DATA_FILE)
    events = mne.read_events(RAW_EVE_FILE)

    streamer: data_streamer = data_streamer(raw, events)
    streamer.chans_sel(CHANNELS)
    stages = streamer.staging()

    TIMESLICE = (START, END)
    hypno = stages[TIMESLICE[0]:TIMESLICE[1]]
    data_1 = raw.get_data(start=TIMESLICE[0], stop=TIMESLICE[1], picks=CHANNELS)[0] * 1e6
    data_2 = raw.get_data(start=TIMESLICE[0], stop=TIMESLICE[1], picks=CHANNELS)[1] * 1e6
    data = data_1 + data_2

    sw: yasa.SWResults = get_slow_waves(data, hypno, SF)
    waves = sw.summary()
    filtered = sw._data_filt[0]

    corrected = correct_slow_waves(filtered, waves, SF)

    sf = 500.
    times = np.arange(data.size) / sf

    mask = sw.get_mask()
    sw_highlight = data * mask
    sw_highlight[sw_highlight == 0] = np.nan

    plt.figure(figsize=(16, 4.5))

    plt.plot(times, filtered, 'k')
    plt.plot(corrected['NegPeak'], filtered[(corrected['NegPeak'] * sf).astype(int)], 'bo', label='Negative peaks')
    plt.plot(corrected['PosPeak'], filtered[(corrected['PosPeak'] * sf).astype(int)], 'go', label='Positive peaks')
    plt.plot(corrected['Start'], filtered[(corrected['Start'] * sf).astype(int)], 'ro', label='Start')

    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude (uV)')
    plt.xlim([0, times[-1]])
    plt.title('N3 sleep EEG data')
    plt.legend()
    sns.despine()

    plt.show()

    from phase import phase_signal

    phase = phase_signal(END - START, 500, corrected)

    times = np.arange(phase.size) / SF

    # Plot the signal
    fig, ax = plt.subplots(1, 1, figsize=(16, 4))
    plt.plot(times, phase, lw=1.5, color='k')
    plt.xlim([times.min(), times.max()])
    sns.despine()
    plt.show()

    times = []
    pattern = r'\[\s*1\s+(\d+)\]'
    with open('../resources/n1_sw.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            match = re.search(pattern, line)
            if match:
                second_number = int(match.group(1))
                times.append(second_number)
            else:
                print("No match found.")

    print(times)

    from evaluation import get_online_detection_evaluation

    accuracy, frequencies = get_online_detection_evaluation(phase, times)

    # Compute pie slices
    N = len(frequencies)
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    radii = frequencies
    width = 2 * math.pi / N
    colors = plt.cm.viridis(radii * 3)

    ax = plt.subplot(projection='polar')
    ax.bar(theta, radii, width=width, bottom=0.0, color=colors, alpha=0.5)

    plt.show()
