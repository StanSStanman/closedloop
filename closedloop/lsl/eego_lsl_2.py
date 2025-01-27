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
# matplotlib.use('TkAgg')
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


class ClosedLoopLSL:
    
    def __init__(self, sfreq) -> None:
        self.sfreq = sfreq
        self.devices = []
        self.streams = []
        # self.connected = multiprocessing.Value('b', False)
        self.all_connected = False # multiprocessing.Value('b', False)
        self.aquiring_data = False # multiprocessing.Value('b', False)
        self._aquire = False
        self.data = [] # queue.Queue()
        self.timestamps = [] # queue.Queue()
        return
    
    def search_streams(self, snames: list, stypes: list)->bool:
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
                print('\tFound', len(self.devices), 'correspondig streams.\n')
                
            if time.time() - tstart >= timeout:
                warnings.warn('Unable to find all the devices, timeout passed')
                return False
        
        time.sleep(1.)
        return True
        
    def open_streams(self, bufsize: float = 5.) -> None:
        
        timeout = 10.
        print('Opening streams...')
        
        if len(self.devices) == 0:
            print('\tNo devices to open streams from.\n')
            return False
        
        self.bufsize = bufsize
        self.buflen = int(self.sfreq * bufsize)
        
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
                print('\tOpened', len(self.streams), 'correspondig streams.\n')
                
            if time.time() - tstart >= timeout:
                warnings.warn('Unable to open all the streams, timeout passed')
                return
        
        time.sleep(1.)    
        return False
    
    def connect_streams(self) -> bool:
        
        timeout = len(self.streams) * 20.
        print('Connecting to the streams...')
        
        if len(self.streams) == 0:
            print('\tNo streams to connect to.\n')
            return False
        
        connected = 0
        tstart = time.time()
        while len(self.streams) != connected:
            for s in self.streams:
                try:
                    s.connect(acquisition_delay=0, 
                              processing_flags='all', 
                              timeout=10.)
                    connected += 1
                except Exception:
                    print('Unable to connect to stream', s._name)
            
            if len(self.streams) != connected:
                connected = 0
                for s in self.streams:
                    if s.connected:
                        s.disconnect()
                print('Unable to connect to all the streams, trying again...')
                time.sleep(1.)
            else:
                time.sleep(1.)
                self.all_connected = True
                print('\tStreams connected.\n')
                
            if time.time() - tstart >= timeout:
                warnings.warn('Unable to open all the streams, timeout passed')
                return False

        time.sleep(1.)
        return True
    
    
    def disconnect_streams(self) -> None:
        print('Disconnecting from the streams...')
        for s in self.streams:
            if s.connected:
                s.disconnect()
        self.all_connected = False
        print('\tStreams disconnected.\n')
        return True
    
    
    def apply_filter(self, low_freq: float = .5, 
                     high_freq: float = 4.) -> None:
        for s in self.streams:
            s.filter(low_freq, high_freq, iir_params=None)
            print('Filter applied to stream', s._name, 'range:', 
                  low_freq, '-', high_freq, 'Hz')            
        return
    
    
    def start_acquisition(self, interval: float=.011) -> None:
        
        if not self.all_connected:
            print('No streams are connected for aquisition.')
            return
        
        print('Starting acquisition...')
        self.aquiring_data = True
        
        def _acquire_data(self, interval: float) -> None:
            self._aquire = True
            t_start = time.perf_counter()
            t_next = t_start
            
            while self.aquiring_data:
                # start_time = time.perf_counter()
                data, timestamps = [], []
                
                t_next = t_next + interval
                delay = t_next - time.perf_counter()
                if delay > 0:
                    high_precision_sleep(delay)
                
                for stream in self.streams:
                    stream.acquire()
                    
                    _dt, _ts = stream.get_data()
                    data.append(_dt)
                    timestamps.append(_ts)
                
                aligned_data, _ = self.sync_stack_data(data, timestamps, 
                                                       n_chans=64)
                # print(aligned_data)
                self.data = (aligned_data)
                # print('Acquisition time:', time.perf_counter() - start_time)
            
            self._aquire = False
            return
        
        threading.Thread(target=_acquire_data, args=(self, interval)).start()
        
        print('\tAcquisition thread started.\n')
        high_precision_sleep(1.)
        
        return
    
                    
    def stop_acquisition(self):
        start = time.perf_counter()
        if not self.aquiring_data:
            print('Acquisition is not running.')
            return
        else:
            self.aquiring_data = False
            print('Stopping acquisition...')
            while self._aquire:
                pass
            print('\tAcquisition stopped.\n')
            print('Stop acquisition time:', time.perf_counter() - start)
            return
    
    
    def sync_stack_data(self, data: Optional[List[np.ndarray]], 
                        timestamps: Optional[List[np.ndarray]], 
                        n_chans: int) -> None:
        # start_time = time.perf_counter()
        # Picking a certain number of channels from each amplifier
        ch_data = [d[:n_chans, :] for d in data]
        # Aligning data according to the timestamps
        xarr = []
        for i, (_d, _t) in enumerate(zip(ch_data, timestamps)):
            da = xr.DataArray(_d, coords={
                'channels': np.arange(n_chans) + n_chans * i, 
                'times': _t}, dims=('channels', 'times'))
            da = da.drop_duplicates(dim='times', keep=False)
            # print(da.times)
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
        # for i, (_d, _t) in enumerate(zip(ch_data, timestamps))
        aligned_data = xr.combine_by_coords(xarr)
                
        # Check if there are missing values in the aligned data, 
        # that would probably due to a difference in timestamps
        if aligned_data.isnull().any():            
            # Take the last time point available of aligned data
            max_time = aligned_data.times[-1].values
            # Upsample the data to have continuous time series
            aligned_data = aligned_data.interpolate_na(dim='times', method='linear').dropna('times')
            # Drop the NaN values (first or last time points)
            aligned_data = aligned_data.dropna('times')
            # Computing the new timestamps for the buffer
            # bufend = aligned_data.times[-1].values
            # bufstart = bufend - self.bufsize
            buftimes = np.round(np.linspace(aligned_data.times[-1].values - self.bufsize,
                                            aligned_data.times[-1].values,
                                            self.buflen), 3)
            # Downsample and align the data to the new buffer timestamps
            aligned_data = aligned_data.interp(times=buftimes, method='linear')
            # Compute the time between the last time point available and the new buffer end
            elapsed_time = max_time - aligned_data.times[-1].values            
        else:
            elapsed_time = 0.
            
        # print(elapsed_time)
        # print('Sync and stack time:', time.perf_counter() - start_time)
        return aligned_data, elapsed_time
    
            
if __name__ == '__main__':

    # streams_name = ['EE225-000000-000625',
    #                 'EE225-000000-000626',
    #                 'EE225-000000-000627',
    #                 'EE225-000000-000628']
    # streams_type = ['EEG', 'EEG', 'EEG', 'EEG']
    # streams_type = ['eeg', 'eeg', 'eeg', 'eeg']
    
    streams_name = ['EE225-000000-000625']
    # streams_name = ['EE225-000000-000630']
    # streams_type = ['EEG']
    streams_type = ['eeg']
    
    task = ClosedLoopLSL(sfreq=500.)
    
    task.search_streams(snames=streams_name, stypes=streams_type)
    
    task.open_streams(bufsize=5.)
    
    task.connect_streams()
    
    task.apply_filter()
    
    task.start_acquisition()
    
    # task.new_protocol(interval=0.011, filter=[0.5, 15.])
    # time.sleep(5)

    # print('Acquiring data...')
    t0 = time.time()
    t1 = t0 + 10.
    data = []
    while time.time() < t1:
        # print(task.data)
        time.sleep(0.05)
        data.append(task.data)
        # print(task.data.get())
        # print(task.timestamps.get()[0][-1], time.perf_counter())
        pass
    print('time passed')
       
    task.stop_acquisition()
    # print('closing threads...')
    task.disconnect_streams()
    
    for d in data:
        plt.plot(d.times, d.values[0,:])
    plt.show()
    # plt.close()

    # task.protocol_thread.join()
    print('Done')
    