import math
import numpy as np
import mne

def simulate_trigger(sfreq, triggers_info, lenght=10.):
    signals = []
    for ch in triggers_info.keys():
        trig_ids = triggers_info[ch]['ids']
        delays = triggers_info[ch]['delays']
        n_tp = math.ceil(sfreq * lenght)
        signal = np.zeros(n_tp)
        for t, d in zip(trig_ids, delays):
            _tp = math.ceil(sfreq * d)
            signal[_tp] = t
        
        signal = np.expand_dims(signal, 0)
        signals.append(signal)
    signals = np.vstack(signals)
    
    info = mne.create_info(ch_names=list(triggers_info.keys()), 
                           sfreq=sfreq, ch_types='misc')
    triggers = mne.io.RawArray(signals, info)
    
    return triggers


if __name__ == '__main__':
    trigs = {'TRIG0': 
        {'ids': [22, 22, 24, 24, 26, 26], 
         'delays': [15, 16, 17, 19, 21, 24]}}
    
    data = simulate_trigger(500, trigs, 40)
