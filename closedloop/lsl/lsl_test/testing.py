import mne_lsl
from mne_lsl.lsl import resolve_streams
import numpy as np
import time
import threading
import pandas as pd

import parallel

def send_trigger(p, code):
    """
        Invia un codice di trigger e lo converte in binario tramite porta parallela.

        Args:
            p : porta parallela
            code (int): Il codice numerico da inviare.

        Returns:
            None
    """
    
    # Invia il valore alla porta parallela, attende 10ms e poi resetta la porta
    p.setData(code)
    time.sleep(0.01)
    p.setData(0)
    return


class streamer_test:
    
    def __init__(self, t_lenght, interval, ch_trig, triggers):
        self.t_lenght = t_lenght
        self.interval = interval
        self.ch_trig = ch_trig
        self.trig_code = triggers['ids']
        self.trig_time = triggers['delays']
        self.trig_unique = np.unique(triggers['ids'])
        self.p = parallel.Parallel()
        self.timestamps = []
        self.delays_pos = []
        
    def stream_search(self, sname, stype, uid=None, bufsize=2):
        devices = resolve_streams()
        # print(devices)
        for d in devices:
            if d.name == sname and d.stype == stype:
                device = d
        print(device)
        # device = devices[0]

        sname, stype, suid = device.name, device.stype, device.uid
        
        self.stream = mne_lsl.stream.StreamLSL(bufsize=bufsize, 
                                               name=sname, stype=stype)
        
    def start_stream(self):
        print('Streaming...')
        t_start = time.time()
        t_end = t_start + self.t_lenght
        tnext = t_start

        while time.time() < t_end:
            tnext = tnext + self.interval
            delay = tnext - time.time()
            if delay > 0:
                time.sleep(delay)
            
            self.stream.acquire()
            # print(f"New samples acquired: {self.stream.n_new_samples}")
            
            threading.Thread(target=self.trig_search).start()
            # trs.start()
            
            data, timestamps = self.stream.get_data()
            
        self.stream.disconnect()
        print('Disconnected.')
        return data, timestamps
    
    def stream_connect(self):
        self.stream.connect(acquisition_delay=0, 
                            processing_flags='all', 
                            timeout=5.)
        print('Connetting to the stream...')
        time.sleep(10)
    
    def trig_search(self):
        n_samp = self.stream.n_new_samples
        if n_samp != 0:
            samp_data = self.stream.get_data()[0][self.ch_trig][-n_samp:]
            for t in self.trig_unique:
                if t in samp_data:
                    # print(samp_data)
                    self.timestamps.append([t, 'in', time.time()])
                    self.delays_pos.append([n_samp, np.where(samp_data==t)[0]])
                
    def start_trigger(self):
        for tc, tt in zip(self.trig_code, self.trig_time):
            time.sleep(tt)
            # print('Sending trigger', tc)
            strig = threading.Thread(target=self._start_trigger, args=(tc,))
            strig.start()
            self.timestamps.append([tc, 'out', time.time()])
            
    def _start_trigger(self, tc):
        # Invia il valore alla porta parallela, attende 10ms e poi resetta la porta
        self.p.setData(tc)
        time.sleep(0.01)
        self.p.setData(0)
            
    def start_test(self):
        stream = threading.Thread(target=self.start_stream)
        trigger = threading.Thread(target=self.start_trigger)
        stream.start()
        trigger.start()
        
        stream.join()
        trigger.join()
        
    def get_delays(self):
        return self.timestamps
    
    
if __name__ == "__main__":
    
    # Inizializzazione della porta parallela
    address = 0x378
    # p = parallel.Parallel()
    # triggers = {'ids': [22, 22, 24, 24, 26, 26], 
    #             'delays': [15, 1, 1, 2, 2, 3]}
    
    tidx = np.full(1000, 22)
    tdel = np.full(1000, 0.5)
    tdel[0] = 10
    
    triggers = {'ids': list(tidx), 
                'delays': list(tdel)}
    
    test = streamer_test(t_lenght=550, interval=0.01, 
                         ch_trig=67, triggers=triggers)
    test.stream_search(sname='EE225-000000-000625', stype='EEG')
    # test.stream_search(sname='MNE-LSL-Player', stype='')
    test.stream_connect()
    test.start_test()
    
    delays = test.get_delays()
    d_pos = test.delays_pos
    
    delays_df = pd.DataFrame(delays, columns=['trig_id', 'inout', 'timestamp'])
    delays_df.to_csv('/home/phantasos/Scrivania/lsl_tests/test_4.csv')
    d_pos = np.hstack(d_pos)
    d_pos_df = pd.DataFrame(d_pos, columns=['n_samp', 'trig_idx'])
    d_pos_df.to_csv('/home/phantasos/Scrivania/lsl_tests/test_idx_4.csv')
    
    print(test.get_delays())
    print(test.delays_pos)
    
    #############

    # devices = resolve_streams()
    # print(devices)
    # # for d in devices:
    # #     if d.name == 'EE225-000000-000625' and d.stype == 'EEG':
    # #         device = d
    # # print(device)
    # device = devices[0]

    # sname, stype, suid = device.name, device.stype, device.uid

    # stream = mne_lsl.stream.StreamLSL(bufsize=2, name=sname, stype=stype)
    # stream.connect(acquisition_delay=0, processing_flags='all', timeout=5.)

    # t0 = time.time()
    # t1 = t0 + 30
    # interval = 0.01
    # tnext = t0

    # # stream.acquire()
    # while time.time() < t1:
    #     tnext = tnext + interval
    #     delay = tnext - time.time()
    #     if delay > 0:
    #         time.sleep(delay)
        
    #     stream.acquire()
    #     print(f"New samples acquired: {stream.n_new_samples}")
    #     data, timestamps = stream.get_data()
        
    # stream.disconnect()
