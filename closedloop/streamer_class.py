import os.path as op
import math
import numpy as np
import mne
import xarray as xr
import time
# from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from visualize import online_vis


class data_streamer:

    def __init__(self, raw, events):
        assert isinstance(raw, mne.io.fiff.raw.Raw), 'raw shoul be a mne.Raw \
        object'

        if isinstance(events, np.ndarray):
            if not (events.ndim == 2 and events.shape[1] == 3):
                raise ValueError('events should be a [n, 3] dimensional array')
        else:
            raise ValueError('events should be a [n, 3] dimensional array')

        self.raw_data = raw
        self.ev_data = events
        self.channels = raw.ch_names
        self.stages = [2, 3]
        self.buffer_len = 50.
        self.stage_vec = self.staging()

    def chans_sel(self, channels):
        if isinstance(channels, list):
            channels = [self.raw_data.ch_names[c] if isinstance(c, int) else c for c in channels]
            assert all(c in self.raw_data.ch_names for c in channels), "Invalid channel(s)"
        else:
            raise ValueError("Channels should be a list of strings, list of int, or None")
        self.channels = channels
    
    def stages_sel(self, stages):
        self.stages = stages

    def buffer_ms(self, buffer_len):
        self.buffer_len = buffer_len

    def staging(self):
        # Creating staging vector
        stage_vec = self.raw_data.copy().pick(["staging"]).get_data().squeeze()
        for i, e in enumerate(self.ev_data):
            ev_start, ev_end = e[0], len(stage_vec) if i == len(self.ev_data) - 1 else self.ev_data[i + 1, 0]
            stage_vec[ev_start:ev_end] = e[-1]
        return stage_vec
    
    def prepare(self):
         # Selecting desired channels (get rid of staging)
        self.raw_data = self.raw_data.pick_channels(self.channels, ordered=True, verbose=False)
        print('Keeping channels:', self.raw_data.ch_names)

        # Computing buffer sample length 
        sfreq = self.raw_data.info['sfreq']
        self.n_sample = int((sfreq / 1000) * self.buffer_len) # samples each ms * buffer ms
        print('Number of samples in a buffer:', self.n_sample)

        self.raw_data = xr.DataArray(self.raw_data.get_data(),
                                    coords=[self.raw_data.ch_names, self.stage_vec], 
                                    dims=['channel', 'stage'])
        
        self.s_start, self.s_end = 0, self.n_sample
        self.s_steps = math.ceil(self.raw_data.shape[-1] / self.n_sample)

    def generator(self):
        for _ in range(self.s_steps):
            self.chunk = self.raw_data[:, self.s_start:self.s_end]
            if self.chunk.stage[0] not in self.stages:
                self.chunk = xr.full_like(self.chunk, np.nan)

            self.s_start += self.n_sample
            self.s_end += self.n_sample

            yield self.chunk

    def stream(self):
        return next(self.generator())


if __name__ == '__main__':
    subjects = ['n1', 'n2', 'n3', 'n4', 'n5', 'n10']
    path = '/home/jerry/python_projects/space/closedloop/test_data'
    raw_fname = op.join(path, '{0}_raw.fif')
    eve_fname = op.join(path, '{0}_eve.fif')

    raw = mne.io.read_raw_fif(raw_fname.format(subjects[0]))
    events = mne.read_events(eve_fname.format(subjects[0]))

    stream = data_streamer(raw, events)
    stream.chans_sel(['F1-F3','C4-A1'])
    # stream.chans_sel(['C4-A1'])
    stream.stages = [0, 1, 2, 3]
    stream.buffer_len = 50.
    stream.prepare()

    figure = online_vis()
    figure.n_sample = 5000
    figure.figsize = (15, 9)
    figure.stages = True

    t_start = time.time()
    # n_chunks = int(raw.times[-1] / (stream.buffer_len / 1000))
    n_chunks = 50000
    print(n_chunks)

    data = []
    for n in range(n_chunks):
        signal = stream.stream()
        data.append(signal)
        # figure.update(signal)
    # data = Parallel(n_jobs=-1, backend='sequential')(delayed(stream.stream)() for i in range(n_chunks))
    t_end = time.time()
    print('Total time:', t_end - t_start)
    print('Average time:', (t_end - t_start) / n_chunks)
    plt.plot(np.hstack(data).T)
    plt.show(block=True)