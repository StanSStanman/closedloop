import matplotlib.pyplot as plt
import mne
import pandas as pd
from streamer_class import data_streamer
from yasa import sw_detect
import numpy as np
import pywt
from streamer_class import data_streamer
import seaborn as sns
import pywt

def get_slow_waves(signal: np.ndarray, hypno: np.ndarray, sf: int) -> pd.DataFrame:
    """
    Get a Dataframe containing the information about the slow waves

    :param signal: signal to analyse
    :param hypno: signal of sleep phases
    :param sf: sampling frequency
    :return: dataframe containing slow waves information
    """
    sw = sw_detect(signal, sf, hypno=hypno, include=(2, 3), freq_sw=(0.3, 1.5),
                   dur_neg=(0.3, 1.5), dur_pos=(0.1, 1.5), amp_neg=(40, 200),
                   amp_pos=(10, 150), amp_ptp=(75, 350), coupling=False,
                   remove_outliers=False, verbose=False)

    return sw.summary()

def get_slow_waves_mask(signal: np.ndarray, hypno: np.ndarray, sf: int):
    """
    Get a Dataframe containing the information about the slow waves

    :param signal: signal to analyse
    :param hypno: signal of sleep phases
    :param sf: sampling frequency
    :return: dataframe containing slow waves mask information
    """
    sw = sw_detect(signal, sf, hypno=hypno, include=(2, 3), freq_sw=(0.3, 1.5),
                   dur_neg=(0.3, 1.5), dur_pos=(0.1, 1.5), amp_neg=(40, 200),
                   amp_pos=(10, 150), amp_ptp=(75, 350), coupling=False,
                   remove_outliers=False, verbose=False)

    return sw.get_mask()

def get_tuple_wavedec_coeffs (data, waveletType, mode, levelDec):
    """
    Performs discrete wavelet transform (DWT) on a signal.
    It was created to try to avoid the constraint of having
    to manually declare as many variables as the decomposition
    levels chosen when deciding to use it than suggested by the
    relevant documentation.

    Args:
        data: The signal to be decomposed.
        waveletType: The type of wavelet to be used.
        mode: The signal extension mode.
        levelDec: The level of decomposition.

    Returns:
        A tuple containing the approximation and detail coefficients.

    N.B. Feel free to apply modification for params or whatever.
    """

    coeffs = pywt.wavedec(data, waveletType, mode, levelDec)

    cA = coeffs[0]
    cD = []
    for i in range(1, levelDec + 1):
        cD.append(coeffs[i])

    return cA, cD

# Example Of Usage
if __name__ == '__main__':

    RAW_DATA_FILE = './resources/n1_raw.fif'
    RAW_EVE_FILE = './resources/n1_eve.fif'
    SF = 500

    CHANNELS = ['F4-C4', 'C4-A1']

    raw = mne.io.read_raw_fif(RAW_DATA_FILE)
    events = mne.read_events(RAW_EVE_FILE)

    streamer: data_streamer = data_streamer(raw, events)
    streamer.chans_sel(CHANNELS)
    stages = streamer.staging()

    TIMESLICE = (8445000, 8445000 + 15000)
    hypno = stages[TIMESLICE[0]:TIMESLICE[1]]
    data_1 = raw.get_data(start=TIMESLICE[0], stop=TIMESLICE[1], picks=CHANNELS)[0] * 1e6
    data_2 = raw.get_data(start=TIMESLICE[0], stop=TIMESLICE[1], picks=CHANNELS)[1] * 1e6
    data = data_1 + data_2

    # Return slow wave summary infos
    slow_waves_eve = get_slow_waves(data, hypno, SF)

    # Highlight slow waves
    mask = get_slow_waves_mask(data, hypno, SF)
    sw_highlight = data * mask
    sw_highlight[sw_highlight == 0] = np.nan

    # Create a time array
    times = np.arange(data.size) / SF


    # Wavelet Parameters
    waveletType = 'haar'
    mode = 'symmetric'
    levelDec = 8

    # --- Applying Wavelet Decomposition ---
    # (not "get_tuple_wavedec_coeffs" version. Check its definition above)
    coeffs = pywt.wavedec(data, waveletType, mode, levelDec)
    cA, cD1, cD2, cD3, cD4, cD5, cD6, cD7, cD8 = coeffs

    fig, ax = plt.subplots(1, 1, figsize=(16, 4))
    # Shows coefficients obtained by wavelet function
    plt.title("Global Apprximation")
    plt.plot(cA, label='Coefficient A')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (μV)")
    fig, ax = plt.subplots(1, 1, figsize=(16, 4))
    plt.title("Detail 1")
    plt.plot(cD1, label='Coefficient D1')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (μV)")
    fig, ax = plt.subplots(1, 1, figsize=(16, 4))
    plt.title("Detail 8")
    plt.plot(cD8, label='Coefficient D8')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (μV)")
    plt.xlim([times.min(), times.max()])
    sns.despine()
    plt.legend()

    fig, ax = plt.subplots(1, 1, figsize=(16, 4))
    # Plot the signal
    plt.plot(times, data, 'k')
    plt.plot(times, sw_highlight, 'indianred')
    plt.title("Raw Signal")
    # Plot the slow waves peaks
    plt.plot(slow_waves_eve['NegPeak'], sw_highlight[(slow_waves_eve['NegPeak'] * SF).astype(int)], 'bo', label='Negative peaks')
    plt.plot(slow_waves_eve['PosPeak'], sw_highlight[(slow_waves_eve['PosPeak'] * SF).astype(int)], 'go', label='Positive peaks')
    plt.plot(slow_waves_eve['Start'], data[(slow_waves_eve['Start'] * SF).astype(int)], 'ro', label='Start')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (μV)")
    plt.xlim([times.min(), times.max()])
    sns.despine()
    plt.legend()
    plt.show()