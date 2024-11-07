import mne_lsl
from mne_lsl.lsl import resolve_streams

import numpy as np
import pandas as pd
import xarray as xr

import time
import warnings
from typing import Optional, List
# from numba import njit

import threading
import multiprocessing
import queue

import parallel

from closedloop.lsl.utils import high_precision_sleep

import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


class stimulation_protocol:
    
    def __init__(self, sfreq) -> None:
        self.sfreq = sfreq
        self.devices = []
        self.streams = []
        self.connected = multiprocessing.Value('b', False)
        self.all_connected = multiprocessing.Value('b', False)
        self.aquiring_data = multiprocessing.Value('b', False)
        self.data = queue.Queue()
        self.timestamps = queue.Queue()
        return
    
    def search_streams(self, snames: list, stypes: list) -> None:
        timeout = 10.
        print('Looking for available devices...')
        avail_devices = resolve_streams()
        
        self.devices = []
        tstart = time.time()
        while len(snames) != len(self.devices):
            for sn, st in zip(snames, stypes):
                for ad in avail_devices:
                    if ad.name == sn and ad.stype == st:
                        self.devices.append((ad.name, ad.stype, ad.uid))
                        
            if len(snames) != len(self.devices):
                self.devices = []
                print(len(self.devices), 'out of', len(snames), 
                          'devices found, trying again...')
                time.sleep(1.)
            else:
                print('\tFound', len(self.devices), 'correspondig streams.')
                
            if time.time() - tstart >= timeout:
                warnings.warn('Unable to find all the devices, timeout passed')
                return
        
        return
        
    def open_streams(self, bufsize: float = 5.) -> None:
        self.bufsize = bufsize
        self.buflen = int(self.sfreq * bufsize)
        
        timeout = 10.
        print('Opening streams...')
        
        self.streams = []
        tstart = time.time()
        while len(self.devices) != len(self.streams):
            for d in self.devices:
                sname, stype, suid = d
                try:
                    self.streams.append(mne_lsl.stream.StreamLSL(bufsize, 
                                                                 name=sname, 
                                                                 stype=stype))
                        # mne_lsl.stream.StreamLSL(bufsize, 
                        #                          name=sname, 
                        #                          stype=stype, 
                        #                          source_id=suid)
                except Exception:
                    print('Unable to open stream', sname)
                    
            if len(self.devices) != len(self.streams):
                self.streams = []
                print('Unable to open all the streams, trying again...')
                time.sleep(1.)
            else:
                print('\tOpened', len(self.streams), 'correspondig streams.')
                
            if time.time() - tstart >= timeout:
                warnings.warn('Unable to open all the streams, timeout passed')
                return
            
        return
    
    # def connect_streams(self) -> None:
    #     timeout = len(self.streams) * 20.
    #     print('Connecting to the streams...')
        
    #     connected = 0
    #     tstart = time.time()
    #     while len(self.streams) != connected:
    #         for s in self.streams:
    #             try:
    #                 s.connect(acquisition_delay=0, 
    #                           processing_flags='all', 
    #                           timeout=10.)
    #                 connected += 1
    #             except Exception:
    #                 print('Unable to connect to stream', s._name)
            
    #         if len(self.streams) != connected:
    #             connected = 0
    #             for s in self.streams:
    #                 if s.connected:
    #                     s.disconnect()
    #             print('Unable to connect to all the streams, trying again...')
    #             time.sleep(1.)
    #         else:
    #             time.sleep(10)
    #             print('\tStreams connected.')
                
    #         if time.time() - tstart >= timeout:
    #             warnings.warn('Unable to open all the streams, timeout passed')
    #             return

    #     return
    
    # def apply_filter(self, low_freq: float = .5, 
    #                  high_freq: float = 4.) -> None:
    #     for s in self.streams:
    #         s.filter(low_freq, high_freq, iir_params=None)
            
    #     return
    
    
    def connect_and_acquire(self, stream, queue, interval, filter, sync_start, 
                            acquiring, connected, all_connected):
        
        if filter == [None, None]:
            filter = None
        
        print('Connecting to stream', stream.name)
        
        sync_connect = sync_start - time.perf_counter()
        print('Waiting', sync_connect, 'seconds to connect to stream', stream.name)
        high_precision_sleep(sync_connect)
        
        connected.value = stream.connected
        while not connected.value:
            try:
                stream.connect(acquisition_delay=0, 
                               processing_flags='all', 
                               timeout=5.)
                
                if filter is not None:
                    low_freq, high_freq = filter[0], filter[-1]
                    # stream.filter(low_freq, high_freq, iir_params=None)
                    stream.filter(low_freq, high_freq, iir_params=None)
                    
                connected.value = stream.connected
            except Exception:
                print('Unable to connect to stream', stream.name,
                      ', trying again...')
                
        queue.put(stream.connected)
                
        while not all_connected.value:
            pass
        
        high_precision_sleep(.5)
        print('Starting acquisition for stream', stream.name, 'at', time.perf_counter())
        
        t_start = time.perf_counter()
        t_next = t_start
        
        n_samp = 0
        while acquiring.value:
            
            t_next = t_next + interval
            delay = t_next - time.perf_counter()
            if delay > 0:
                high_precision_sleep(delay)
            
            # print(stream.connected)
            
            stream.acquire()
            # print(stream.n_new_samples, stream.name)
            
            # new_samp = stream.n_new_samples
            # tot_samp = n_samp + new_samp
            # if tot_samp == 33:
            #     data, timestamps = stream.get_data()
            #     queue.put((data, timestamps))
            #     n_samp = 0
            # elif tot_samp < 33:
            #     n_samp = new_samp
            # else:
            #     data, timestamps = stream.get_data()    
            # print(tot_samp, stream.name)
            
            data, timestamps = stream.get_data()
            queue.put((data, timestamps))
            
            # print(timestamps[-1])
            
        print('Stopping acquisition for stream', stream.name, 'at', time.perf_counter())
            
        stream.disconnect()
        connected.value = stream.connected
        print('Stream', stream.name, 'disconnected with value', connected.value)
        return
    
    
    def stop_acquisition(self):
        self.aquiring_data.value = False
        return
    
    
    def new_protocol(self, interval: float = 0.011, filter: Optional[list] = [.5, 5.]) -> None:
        print('Setting up new protocol...')
        # self.protocol_thread = threading.Thread(target=self._new_protocol, args=(interval, filter))
        self.protocol_thread = multiprocessing.Process(target=self._new_protocol, args=(interval, filter))
        print('Starting new protocol...')
        self.protocol_thread.start()
        return
    
    
    def _new_protocol(self, interval: float = 0.011, filter: Optional[list] = [.5, 5.]) -> None:
        
        # plt.ion()
        # fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        # ax.set_ylim([-20e-5, 20e-5])
        # ln, = ax.plot([], [])
        
        self.aquiring_data.value = True
        
        queues = [multiprocessing.Queue() for _ in range(len(self.streams))]
        
        sync_start = time.perf_counter() + 1.
        # sync_start = 5.
        
        acquire_proc = []
        for s, q in zip(self.streams, queues):
                proc = multiprocessing.Process(target=self.connect_and_acquire,
                                               args=(s, q, interval, filter,
                                                     sync_start,
                                                     self.aquiring_data,
                                                     self.connected,
                                                     self.all_connected)
                                               )
                acquire_proc.append(proc)
                # high_precision_sleep(.011)
                
        for p in acquire_proc:
            p.daemon = True
            p.start()
        
        while not self.all_connected.value:
            alcon = []
            for q in queues:
                alcon.append(q.get())
            if all(alcon):
                self.all_connected.value = True
                print('Ready to stream, synching...')

        print('All streams connected')
        
        past_time = [0, 0, 0, 0]
        change_time_start = time.perf_counter()
        all_amp_new = 0
        while self.aquiring_data.value:
            
            _data, _timestamps = [], []
            
            for q in queues:
                dt, ts = q.get()
                _data.append(dt.astype('float32'))
                _timestamps.append(np.round(ts.astype('float32'), 3))
                # _data.append(dt)
                # _timestamps.append(np.round(ts, 3))
                # _timestamps.append(ts.astype('float32'))
                # _timestamps.append(ts)
                # self.data.put(dt)
                # self.timestamps.put(ts)
                
            # These lines are for putting the data in the thread, they are super RAM consuming
            # UNLESS you don't retrieve continuously the data from the queue
            # TODO find a way to stop the thread and flush the queue when stopping acquisition
            # self.data.put(_data)
            # self.timestamps.put(_timestamps)
            
            # print(self.timestamps[-1][-1], time.perf_counter())
            # past_time = last_time
            last_time = [lts[-1] for lts in _timestamps]
            print(last_time)
            # print([_d.shape for _d in _data])
            
            # def no_memory_explosion(queues):
            #     while True:
            #         # print('Emptying queues...')
            #         for q in queues:
            #             q.get()
            #     pass
            # explode = multiprocessing.Process(target=no_memory_explosion, args=(queues,)).start()
            
            self.sync_stack_data(_data, _timestamps, n_chans=64)
            # print([len(lts) for lts in _timestamps])
            
            # if past_time != last_time:
            if all(pt != lt for pt, lt in zip(past_time, last_time)):
                # func_start = time.perf_counter()
                # self.sync_stack_data(_data, _timestamps, n_chans=64)
                # sync = threading.Thread(target=self.sync_stack_data, args=(_data, _timestamps, 64))
                # sync.start()
                # sync.join()
                # print(time.perf_counter() - func_start)
                # print([len(lts) for lts in _timestamps])
                
                past_time = last_time
                
                change_time_end = time.perf_counter()
                print('All ampli updated after', change_time_end - change_time_start, 'seconds')
                change_time_start = change_time_end
                
            # if all(pt == lt for pt, lt in zip(past_time, last_time)):
            #     self.sync_stack_data(_data, _timestamps, n_chans=64)
                
            

            
            # ln.set_xdata(self.timestamps[0].squeeze())
            # ln.set_ydata(self.data[0][1, :])
            
            # ax.relim()
            # ax.autoscale_view()
            
            # fig.canvas.draw()
            # fig.canvas.flush_events()
            
            # print('\n New acquisition:', [s.n_new_samples for s in self.streams])
            
        print('Out of acquiring loop')
        # time.sleep(2)
        for p in acquire_proc:
            p.terminate()
            p.join()
        print('Acquisition stopped')
        
        return
    
        
    # def start_protocol(self, interval: float = 0.011) -> None:
        
    #     self.aquiring_data.value = True
    #     self.data = {}
    #     self.timestamps = {}
        
    #     def _aquire_data(stream, queue, interval, sync_start, acquiring):
    #         # Make sure that all the acquisition starts at the same time
    #         wait = sync_start - time.perf_counter()
    #         high_precision_sleep(wait)
            
    #         t_start = time.perf_counter()
    #         t_next = t_start
            
    #         while acquiring.value:
                                
    #             t_next = t_next + interval
    #             delay = t_next - time.perf_counter()
    #             if delay > 0:
    #                 high_precision_sleep(delay)
                
    #             print(stream.connected)
    #             stream.acquire()
    #             print(stream.n_new_samples)
                
    #             data, timestamps = stream.get_data()
    #             print(timestamps[-1])
                
    #             queue.put((data, timestamps))
        
    #         return
        
    #     queues = [multiprocessing.Queue() for _ in range(len(self.streams))]
        
    #     sync_start = time.perf_counter() + 0.05
        
    #     acquire_proc = []
    #     for s, q in zip(self.streams, queues):
    #             proc = multiprocessing.Process(target=_aquire_data,
    #                                            args=(s, q, interval,
    #                                                  sync_start,
    #                                                  self.aquiring_data)
    #                                            )
    #             acquire_proc.append(proc)
                
    #     for p in acquire_proc:
    #             p.start()

    #     while self.aquiring_data.value:
            
    #         self.data, self.timestamps = [], []
            
    #         for q in queues:
    #             dt, ts = q.get()
    #             self.data.append(dt)
    #             self.timestamps.append(ts)
                
    #         print('\n New acquisition:', [s.n_new_samples for s in self.streams])
                
    #         # stacked_data = np.vstack(self.data)
    #         # stacked_timestamps = np.vstack(self.timestamps)
            
    #         # print(stacked_timestamps)

    # @njit
    def sync_stack_data(self, data: Optional[List[np.ndarray]], timestamps: Optional[List[np.ndarray]], n_chans: int) -> None:
        
        # data = data if data is not None else self.data
        # timestamps = timestamps if timestamps is not None else self.timestamps
        
        # print([len(ts) for ts in timestamps])
        # print(timestamps[0][-1])
        # data = np.vstack([d[:nc] for d, nc in zip(self.data, n_chans)])
        
        # Picking a certain number of channels from each amplifier
        ch_data = [d[:n_chans, :] for d in data]
        # Aligning data according to the timestamps
        ts = time.perf_counter()
        xarr = []
        for i, (_d, _t) in enumerate(zip(ch_data, timestamps)):
            da = xr.DataArray(_d, coords={
                'channels': np.arange(n_chans) + n_chans * i, 
                'times': _t}, dims=('channels', 'times'))
            da = da.drop_duplicates(dim='times', keep=False)
            print(da.times)
            xarr.append(da)
        # xarr = [
        # xr.DataArray(_d, 
        #              coords={"channels": np.arange(n_chans) + n_chans * i, 
        #                      "times": _t}, 
        #              dims=("channels", "times")).drop_duplicates(dim='times', 
        #                                                          keep='last')
        # for i, (_d, _t) in enumerate(zip(ch_data, timestamps))]
        # xarr = [
        # xr.DataArray(_d, 
        #              coords={"channels": np.arange(n_chans) + n_chans * i, 
        #                      "times": _t}, 
        #              dims=("channels", "times"))
        # for i, (_d, _t) in enumerate(zip(ch_data, timestamps))]
        
        print([len(ts.times) for ts in xarr])
        
        print('Creating xarrays:', time.perf_counter() - ts)
        
        ts = time.perf_counter()
        aligned_data = xr.combine_by_coords(xarr)
        # try:
        #     aligned_data = xr.combine_by_coords(xarr)
        # except Exception:
        #     for t in timestamps:
        #         for _t in t:
        #             print(_t)
        #         print('\n\n\n\n\n\n')
        print('Combining data:', time.perf_counter() - ts)
                
        # Check if there are missing values in the aligned data, 
        # that would probably due to a difference in timestamps
        if aligned_data.isnull().any():
            # # Take the last time point available of aligned data
            # ts = time.perf_counter()
            # max_time = aligned_data.times[-1].values
            # print('Max time:', time.perf_counter() - ts)
            # # Upsample the data to have continuous time series
            # ts = time.perf_counter()
            # upsample = aligned_data.interpolate_na(dim='times', method='linear')
            # print('Upsampling:', time.perf_counter() - ts)
            # # Drop the NaN values (first or last time points)
            # ts = time.perf_counter()
            # cleaned = upsample.dropna('times')
            # print('Cleaning:', time.perf_counter() - ts)
            # # Computing the new timestamps for the buffer
            # bufend = cleaned.times[-1].values
            # bufstart = bufend - self.bufsize
            # buftimes = np.round(np.linspace(bufstart, bufend,
            #                                 self.buflen), 3)
            # # Downsample and align the data to the new buffer timestamps
            # ts = time.perf_counter()
            # aligned_data = cleaned.interp(times=buftimes, method='linear')
            # print('Downsampling:', time.perf_counter() - ts)
            # # Compute the time between the last time point available and the new buffer end
            # elapsed_time = max_time - aligned_data.times[-1].values
            # # print('Elapsed time:', elapsed_time)
            
            # Take the last time point available of aligned data
            ts = time.perf_counter()
            max_time = aligned_data.times[-1].values
            print('Max time:', time.perf_counter() - ts)
            # Upsample the data to have continuous time series
            ts = time.perf_counter()
            aligned_data = aligned_data.interpolate_na(dim='times', method='linear').dropna('times')
            print('Upsampling:', time.perf_counter() - ts)
            # Drop the NaN values (first or last time points)
            ts = time.perf_counter()
            aligned_data = aligned_data.dropna('times')
            print('Cleaning:', time.perf_counter() - ts)
            # Computing the new timestamps for the buffer
            # bufend = aligned_data.times[-1].values
            # bufstart = bufend - self.bufsize
            buftimes = np.round(np.linspace(aligned_data.times[-1].values - self.bufsize,
                                            aligned_data.times[-1].values,
                                            self.buflen), 3)
            # Downsample and align the data to the new buffer timestamps
            ts = time.perf_counter()
            aligned_data = aligned_data.interp(times=buftimes, method='linear')
            print('Downsampling:', time.perf_counter() - ts)
            # Compute the time between the last time point available and the new buffer end
            elapsed_time = max_time - aligned_data.times[-1].values
            # print('Elapsed time:', elapsed_time)
            
            
        else:
            elapsed_time = 0.
            
        return aligned_data, elapsed_time
            
            # align the timestamps, align the data, cut for the latest time point available for the four amplifiers
            # keep how many final points are not the same (to have the time to add to the detection)
            
    # def online_plot(self):
    #     threading.Thread(target=self._online_plot).start()
    #     pass
    
    
    # def _online_plot(self):
    #     plt.ion()
    #     fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    #     ax.set_ylim([-20e-5, 20e-5])
    #     ln, = ax.plot([], [])
        
    #     background = fig.canvas.copy_from_bbox(ax.bbox)
    #     ax.draw_artist(ln)
    #     fig.canvas.blit(ax.bbox)
        
    #     while self.aquiring_data.value:
    #         # print(self.timestamps)
            
    #         ln.set_xdata(self.timestamps[0].squeeze())
    #         ln.set_ydata(self.data[0][41, :])
            
    #         ax.relim()
    #         ax.autoscale_view()
            
    #         fig.canvas.restore_region(background)
    #         ax.draw_artist(ln)
    #         fig.canvas.blit(ax.bbox)
    #         fig.canvas.flush_events()
            
        return
            
if __name__ == '__main__':
    # import xarray as xr
    
    # d = [np.random.uniform(-1, 1, (2500, 64)) for i in range(4)]
    
    # t = [list(range(0, 2500)), list(range(500, 3000)), list(range(1000, 3500)),
    #      list(range(1500, 4000))]
    
    # t = [np.linspace(0, 2499, 2500), np.linspace(1.1, 2500.1, 2500), np.linspace(3.3, 2502.3, 2500),
    #      np.linspace(5.4, 2504.4, 2500)]
    
    # xarr = []
    # for _d, _t in zip(d, t):
    #     xarr.append(xr.DataArray(_d, coords=[_t, list(range(64))], dims=('times', 'channels')))
        
    # comb = xr.combine_nested(xarr, concat_dim='ampli')
    # comb.dropna('times')
        
    # print(xarr)

    streams_name = ['EE225-000000-000625',
                    'EE225-000000-000626',
                    'EE225-000000-000627',
                    'EE225-000000-000628']
    streams_type = ['EEG', 'EEG', 'EEG', 'EEG']
    # streams_type = ['eeg', 'eeg', 'eeg', 'eeg']
    
    # streams_name = ['EE225-000000-000625']
    # streams_name = ['EE225-000000-000630']
    # streams_type = ['EEG']
    
    task = stimulation_protocol(sfreq=500.)
    
    task.search_streams(snames=streams_name, stypes=streams_type)
    
    task.open_streams(bufsize=5.)
    
    # task.connect_streams()
    
    # task.apply_filter()
    
    # task.start_protocol()
    
    task.new_protocol(interval=0.011, filter=[0.5, 15.])
    time.sleep(5)

    print('Acquiring data...')
    t0 = time.time()
    t1 = t0 + 5
    while time.time() < t1:
        # print(task.data.get())
        # print(task.timestamps.get()[0][-1], time.perf_counter())
        pass
    print('time passed')
       
    task.stop_acquisition()
    print('closing threads...')

    task.protocol_thread.join()
    print('Done')
    