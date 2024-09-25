import mne_lsl
from mne_lsl.lsl import resolve_streams
import time


import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


devices = resolve_streams()
print(devices)
# for d in devices:
#     if d.name == 'EE225-000000-000625' and d.stype == 'EEG':
#         device = d
# print(device)
device = devices[0]

sname, stype, suid = device.name, device.stype, device.uid

stream = mne_lsl.stream.StreamLSL(bufsize=5, name=sname, stype=stype)
stream.connect(acquisition_delay=0, processing_flags='all', timeout=5.)

time.sleep(10)

dt_chunks, ts_chunks = [], []

t0 = time.time()
t1 = t0 + 30
interval = 0.02
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
    plt.plot(ts_chunks[c].squeeze(), dt_chunks[c][4, :])
    # plt.plot(dt_chunks[c][4, :])
    
plt.show(block=True)