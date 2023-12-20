import xarray as xr
import numpy as np
import threading
# import multiprocessing
from mne.filter import filter_data 

class thr_detect:

    def __init__(self, sfreq=500, threshold=-80e-6, twin_len=5., 
                 fil_low=.5, fil_high=4., 
                 delays=[.5, 1.075], refractory=2.5):
        
        self.sfreq = sfreq
        self.thrs = threshold
        self.twin_len = twin_len
        self.fil_low = fil_low
        self.fil_high = fil_high
        self.delays = delays
        self.refractory = refractory
        
        self.chunk = None
        self.n_samples = int(sfreq * twin_len)
        self.listening = True
        # Events:
        # - 0 = threshold pass
        # - 1 = first trigger
        # - 2 = second trigger
        self.events = []
        # current 'time'
        self.ct = 0
        

    def d_filter(self):
        f_data = filter_data(self.chunk.data, self.sfreq, 
                            l_freq=self.fil_low, h_freq=self.fil_high, 
                            method='iir', pad='reflect_limited', 
                            verbose=False, n_jobs=1)
        
        f_chunk = self.chunk.copy()
        f_chunk.data = f_data

        return f_chunk
    

    def trigger(self):
        # developed on simulated time, should be adapted on real time
        self.listening = False
        
        all_delays = self.delays + [self.refractory]
        for i, d in enumerate(all_delays):
            t_start = self.ct
            d -= self.t_diff
            d_tp = d * self.sfreq
            while self.ct < t_start + d_tp:
                continue
            # deliver the trigger
            if i != len(all_delays) -1:
                self.events.append([i + 1, self.ct])

                print(self.events[-1])

        self.listening = True
        return
        
    
    def read_buffer(self, d_buffer):
        self.ct += d_buffer.shape[1]

        if self.chunk is None:
            self.chunk = d_buffer
        else: 
            self.chunk = xr.concat((self.chunk, d_buffer), dim='stage')
        
        if self.chunk.shape[-1] > self.n_samples:
            self.chunk = self.chunk[:, -self.n_samples:]


        # refr_tp = self.refractory * self.sfreq

        if self.listening and self.chunk.shape[-1] >= self.n_samples:

            f_chunk = self.d_filter()

            low_idx = np.where(f_chunk[:, -d_buffer.shape[1]:] <= self.thrs)
            if len(low_idx[0]) > 0:
                thr_pass = low_idx[1][0]
                self.t_diff = (d_buffer.shape[1] - thr_pass) / self.sfreq
                self.events.append([0, self.ct - (d_buffer.shape[1] - thr_pass)])

                self.trig_thr = threading.Thread(target=self.trigger, args=())
                self.trig_thr.start()

        return

if __name__ == '__main__':
    import os.path as op
    import mne
    import time
    from streamer_class import data_streamer
    import os 

    subjects = ['n1', 'n2', 'n3', 'n4', 'n5', 'n10']
    path = '/home/jerry/python_projects/space/closedloop/test_data'
    raw_fname = op.join(path, '{0}_raw.fif')
    eve_fname = op.join(path, '{0}_eve.fif')

    raw = mne.io.read_raw_fif(raw_fname.format(subjects[0]), preload=True)
    events = mne.read_events(eve_fname.format(subjects[0]))

    new_chan_data = raw.copy().pick_channels(['F4-C4']).get_data() + raw.copy().pick_channels(['C4-A1']).get_data()
    info = mne.create_info(['F4-A1'], 500., ch_types=['eeg'])
    new_chan_data = mne.io.RawArray(data=new_chan_data, info=info, first_samp=0)
    raw.add_channels([new_chan_data], force_update_info=True)

    stream = data_streamer(raw, events)
    # stream.chans_sel(['F1-F3','C4-A1'])
    stream.chans_sel(['F4-A1'])
    # stream.stages = [0, 1, 2, 3]
    stream.stages = [2, 3]
    stream.buffer_len = 50.
    stream.prepare()

    # figure = online_vis()
    # figure.n_sample = 5000
    # figure.figsize = (15,9)
    # figure.stages = True
    listener = thr_detect()
    listener.thrs = -40e-6
    listener.twin_len = 5.
    listener.fil_low = .3
    listener.fil_high = 1.5


    t_start = time.time()
    n_chunks = int(raw.times[-1] / (stream.buffer_len / 1000))
    # n_chunks = 50000
    print(n_chunks)

    data = []
    for n in range(n_chunks):
        signal = stream.stream()
        listener.read_buffer(signal)
        data.append(signal)
    listener.trig_thr.join()
    t_end = time.time()
    print('Total time:', t_end - t_start)
    print('Average time:', (t_end - t_start) / n_chunks)
    triggers = listener.events
    triggers = np.array(triggers)
    print(triggers)


