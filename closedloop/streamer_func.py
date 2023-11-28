import os.path as op
import math
import numpy as np
import mne
import xarray as xr
import time
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


def data_streamer(raw_data, events, channels=None, stages=[2, 3], buffer_len=50.):
    # Check function arguments are correct:
    assert isinstance(raw_data, mne.io.fiff.raw.Raw), 'raw shoul be a mne.Raw \
        object'

    if channels is None:
        channels = raw_data.ch_names
    elif isinstance(channels, list):
        channels = [raw_data.ch_names[c] if isinstance(c, int) else c for c in channels]
        assert all(c in raw_data.ch_names for c in channels), "Invalid channel(s)"
    else:
        raise ValueError("Channels should be a list of strings, list of int, or None")
    
    if isinstance(events, np.ndarray):
        if not (events.ndim == 2 and events.shape[1] == 3):
            raise ValueError('events should be a [n, 3] dimensional array')
    else:
        raise ValueError('events should be a [n, 3] dimensional array')

    # Creating staging vector
    staging = raw_data.copy().pick(["staging"]).get_data().squeeze()
    for i, e in enumerate(events):
        ev_start, ev_end = e[0], len(staging) if i == len(events) - 1 else events[i + 1, 0]
        staging[ev_start:ev_end] = e[-1]
            
    # Selecting desired channels (get rid of staging)
    raw_data = raw_data.pick_channels(channels, ordered=True, verbose=False)
    print('Keeping channels:', raw_data.ch_names)

    # Computing buffer sample length 
    sfreq = raw_data.info['sfreq']
    n_sample = int((sfreq / 1000) * buffer_len) # samples each ms * buffer ms
    print('Number of samples in a buffer:', n_sample)

    raw_data = xr.DataArray(raw_data.get_data(),
                            coords=[raw_data.ch_names, staging], 
                            dims=['channel', 'stage'])
    
    s_start, s_end = 0, n_sample
    s_steps = math.ceil(raw_data.shape[-1] / n_sample)

    for _ in range(s_steps):
        chunk = raw_data[:, s_start:s_end]
        if chunk.stage[0] not in stages:
            chunk = xr.full_like(chunk, np.nan)

        s_start += n_sample
        s_end += n_sample

        yield chunk

    return


def returner(x):
    return x


if __name__ == '__main__':
    subjects = ['n1', 'n2', 'n3', 'n4', 'n5', 'n10']
    path = '/home/jerry/python_projects/space/closedloop/test_data'
    raw_fname = op.join(path, '{0}_raw.fif')
    eve_fname = op.join(path, '{0}_eve.fif')

    raw = mne.io.read_raw_fif(raw_fname.format(subjects[0]))
    events = mne.read_events(eve_fname.format(subjects[0]))

    t_start = time.time()
    n_chunks = int(raw.times[-1] / 0.05)
    n_chunks = 50000
    print(n_chunks)

    chunk = data_streamer(raw, events, channels=['C4-A1'], stages=[2, 3], buffer_len=50.)

    data = []
    for n in range(n_chunks):
        c = next(chunk)
        data.append(c.as_numpy())
    # data = Parallel(n_jobs=-1, backend='sequential')(delayed(next)(chunk) for i in range(n_chunks))
    t_end = time.time()
    print('Total time:', t_end - t_start)
    print('Average time:', (t_end - t_start) / n_chunks)
    plt.plot(np.hstack(data).T)
    plt.show()