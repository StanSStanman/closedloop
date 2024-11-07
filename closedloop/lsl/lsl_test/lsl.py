import mne_lsl
from mne_lsl.lsl import resolve_streams
import time
import multiprocessing
from closedloop.lsl.utils import high_precision_sleep

import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

def base_test():
    devices = resolve_streams()
    print(devices)
    while devices == []:
        devices = resolve_streams()
        print(devices)
        time.sleep(1)
    for d in devices:
        # if d.name == 'EE225-000000-000625' and (d.stype == 'EEG' or d.stype == 'eeg'):
        if d.name == 'EE225-000000-000630' and d.stype == 'EEG':
            device = d
            print(device)

    # device = devices[0]

    sname, stype, suid = device.name, device.stype, device.uid

    stream = mne_lsl.stream.StreamLSL(bufsize=5, name=sname, stype=stype)
    stream.connect(acquisition_delay=0, processing_flags='all', timeout=5.)
    stream.filter(.5, 15.)

    time.sleep(10)

    dt_chunks, ts_chunks = [], []

    t0 = time.time()
    t1 = t0 + 10
    interval = 0.01
    tnext = t0

    # stream.acquire()
    while time.time() < t1:
        tnext = tnext + interval
        delay = tnext - time.time()
        if delay > 0:
            time.sleep(delay)
        
        stream.acquire()
        print(f"New samples acquired: {stream.n_new_samples}")
        data, timestamps = stream.get_data()
        
        dt_chunks.append(data)
        ts_chunks.append(timestamps)
        
    stream.disconnect()

    0 == 0

    for c in range(len(dt_chunks)):
        plt.plot(ts_chunks[c].squeeze(), dt_chunks[c][1, :])
        # plt.plot(dt_chunks[c][4, :])
        
    plt.show(block=True)
    
    
def parallel_test():
    devices = resolve_streams()
    print(devices)
    while devices == []:
        devices = resolve_streams()
        print(devices)
        time.sleep(1)
    for d in devices:
        # if d.name == 'EE225-000000-000625' and (d.stype == 'EEG' or d.stype == 'eeg'):
        if d.name == 'EE225-000000-000630' and d.stype == 'EEG':
            device = d
            print(device)

    # device = devices[0]

    sname, stype, suid = device.name, device.stype, device.uid

    stream = mne_lsl.stream.StreamLSL(bufsize=5, name=sname, stype=stype)
    # stream.connect(acquisition_delay=0, processing_flags='all', timeout=5.)
    # stream.filter(.5, 15.)
    
    streams = [stream]

    time.sleep(10)

    def _aquire_data(stream, queue, interval, sync_start, acquiring):
        stream.connect(acquisition_delay=0, processing_flags='all', timeout=5.)
        stream.filter(.5, 15.)        
        
        # Make sure that all the acquisition starts at the same time
        wait = sync_start - time.perf_counter()
        high_precision_sleep(wait)
        
        t_start = time.perf_counter()
        t_next = t_start
        
        while acquiring:
                            
            t_next = t_next + interval
            delay = t_next - time.perf_counter()
            if delay > 0:
                high_precision_sleep(delay)
            
            # print(stream.connected)
            stream.acquire()
            new_samples = stream.n_new_samples
            # print(stream.n_new_samples)
            
            data, timestamps = stream.get_data()
            print(timestamps[-1])
            
            if new_samples != 0:
                queue.put((data, timestamps))
    
        return
    
    queues = [multiprocessing.Queue() for _ in range(len(streams))]
    
    sync_start = time.perf_counter() + 0.5
    
    acquire_proc = []
    for s, q in zip(streams, queues):
            proc = multiprocessing.Process(target=_aquire_data,
                                            args=(s, q, 0.011,
                                                  sync_start,
                                                  True)
                                            )
            acquire_proc.append(proc)
            
    for p in acquire_proc:
            p.start()
            
    data, timestamps = [], []

    t0 = time.time()
    t1 = t0 + 10
    while time.time() <= t1:
        
        # data, timestamps = [], []
        
        for q in queues:
            dt, ts = q.get()
            data.append(dt)
            timestamps.append(ts)
            
        # print('\n New acquisition:', [s.n_new_samples for s in streams])
    
    for c in range(len(data)):
        plt.plot(timestamps[c].squeeze(), data[c][1, :])
        # plt.plot(dt_chunks[c][4, :])
            
    plt.show(block=True)
    
if __name__ == '__main__':
    # base_test()
    parallel_test()