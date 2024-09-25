import mne_lsl
from mne_lsl.lsl import resolve_streams
import time
import threading

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
    def __init__(self, t_lenght, interval):
        self.t_lenght = t_lenght
        self.interval = interval
        
    def stream_search(self, sname, stype, uid=None, bufsize=2):
        devices = resolve_streams()
        print(devices)
        # for d in devices:
        #     if d.name == 'EE225-000000-000625' and d.stype == 'EEG':
        #         device = d
        # print(device)
        device = devices[0]

        sname, stype, suid = device.name, device.stype, device.uid
        
        self.stream = mne_lsl.stream.StreamLSL(bufsize=bufsize, 
                                               name=sname, stype=stype)
        
    def start_stream(self):
        self.t_start = time.time()
        self.t_end = self.t_start + self.t_lenght
        tnext = self.t_start

        while time.time() < self.t_end:
            tnext = tnext + self.interval
            delay = tnext - time.time()
            if delay > 0:
                time.sleep(delay)
            
            self.stream.acquire()
            # print(f"New samples acquired: {stream.n_new_samples}")
            data, timestamps = self.stream.get_data()
            
            trs = threading.Thread(target=self.trig_search)
            trs.start()
            
        return data, timestamps
    
    def stream_connect(self):
        self.stream.connect(acquisition_delay=0, 
                            processing_flags='all', 
                            timeout=5.)
        print('Connetting to the stream...')
        time.sleep(10)
    
    def trig_search(self):
        n_samp = self.stream.n_new_samples
        samp_data = self.stream.get_data()[0][self.ch_trig][-n_samp:]
        for t in self.trigger_list:
            if t in samp_data:
                self.timestamps.append({t: time.time()})
            

    
if __name__ == "__main__":
    
    # Inizializzazione della porta parallela
    address = 0x378
    p = parallel.Parallel()
    
    test = streamer_test(t_lenght=60, interval=0.01)
    test.stream_search(sname='EE225-000000-000625', stype='EEG')
    test.stream_connect()
    
    
    #############

    devices = resolve_streams()
    print(devices)
    # for d in devices:
    #     if d.name == 'EE225-000000-000625' and d.stype == 'EEG':
    #         device = d
    # print(device)
    device = devices[0]

    sname, stype, suid = device.name, device.stype, device.uid

    stream = mne_lsl.stream.StreamLSL(bufsize=2, name=sname, stype=stype)
    stream.connect(acquisition_delay=0, processing_flags='all', timeout=5.)

    t0 = time.time()
    t1 = t0 + 30
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
        
    stream.disconnect()

