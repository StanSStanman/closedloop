import xarray as xr
import numpy as np
import threading
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
        # self.n_samples = int(sfreq * twin_len)
        self.listening = True
        # Events:
        # - 0 = threshold pass
        # - 1 = first trigger
        # - 2 = second trigger
        self.events = []
        # current 'time'
        self.ct = 0
        # self.check_time = int(sfreq * delays[0])

        self.prepared = False


    def prepare(self):
        self.n_samples = int(self.sfreq * self.twin_len)
        self.check_time = int(self.sfreq * self.delays[0])
        self.prepared = False
        

    def d_filter(self):
        f_data = filter_data(self.chunk.data, self.sfreq, 
                            l_freq=self.fil_low, h_freq=self.fil_high, 
                            method='fir', pad='reflect_limited', 
                            verbose=False, n_jobs=1)
        
        f_chunk = self.chunk.copy()
        f_chunk.data = f_data

        return f_chunk
    

    def trigger(self):
        # developed on simulated time, should be adapted on real time
        self.listening = False
        
        # deliver a trigger instantly
        self.events.append([1, self.ct])
        print(self.events[-1])

        all_delays = self.delays[1:] + [self.refractory]
        for i, d in enumerate(all_delays):
            t_start = self.ct
            d -= self.t_diff
            d_tp = d * self.sfreq
            while self.ct < t_start + d_tp:
                continue
            # deliver the trigger
            if i != len(all_delays) - 1:
                self.events.append([i + 2, self.ct])

                print(self.events[-1])

        self.listening = True
        return
        
    
    def read_buffer(self, d_buffer):
        if self.prepared is False:
            self.prepare()

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

            low_idx = np.where(f_chunk[:, -self.check_time:-(self.check_time - d_buffer.shape[1])] <= self.thrs)
            if len(low_idx[0]) > 0:
                thr_pass = low_idx[1][0]
                self.t_diff = (d_buffer.shape[1] - thr_pass) / self.sfreq
                self.events.append([0, self.ct - (d_buffer.shape[1] - thr_pass)])

                self.trig_thr = threading.Thread(target=self.trigger, args=())
                self.trig_thr.start()

        return
    

    def stop(self):
        self.ct += 1000
        self.trig_thr.join()
        return
