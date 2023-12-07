import mne
import pandas as pd
from streamer_class import data_streamer
from yasa import sw_detect
import numpy as np


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


# Example
if __name__ == '__main__':
    RAW_DATA_FILE = '../resources/n1_raw.fif'
    RAW_EVE_FILE = '../resources/n1_eve.fif'
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

    waves = get_slow_waves(data, hypno, SF)
