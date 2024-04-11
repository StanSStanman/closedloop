from ast import literal_eval
import pandas as pd
import numpy as np
import mne
import json

def make_sw_events(sw_fname, eve_fname, idx='maxnegpk', save_id=False):
    
    sw = pd.read_csv(sw_fname, sep=',', header=0, index_col=0)
    
    eve_id = {}
    tps, evc = [], []
    for n, c in enumerate(sw):
        ch = sw[c]
        eve_tp = np.array(literal_eval(ch[idx]), dtype=int)
        
        if c == 'envelope':
            eve_code = 512
        else:
            eve_code = n
        eve_cd = np.full_like(eve_tp, fill_value=eve_code)
        
        eve_id[c] = eve_code
        tps.append(eve_tp)
        evc.append(eve_cd)
        
    tps = np.hstack(tps)
    zrs = np.zeros_like(tps)
    evc = np.hstack(evc)
    
    events = np.stack((tps, zrs, evc), axis=-1)
    
    if save_id:
        id_fname = eve_fname.replace('.fif', '_id.json')
        with open(id_fname, 'w') as f:
            json.dump(eve_id, f)
        print('Events id written in:', id_fname)
    
    mne.write_events(eve_fname, events=events, overwrite=True)
    print('Events written in:', eve_fname)
    
    return events, eve_id

    
if __name__ == '__main__':
    sw_fname = '/home/jerry/python_projects/space/closedloop/closedloop/sw_detect/envelope_sw.csv'
    eve_fname = '/home/jerry/python_projects/space/closedloop/closedloop/sw_detect/sw-eve.fif'
    
    make_sw_events(sw_fname, eve_fname, save_id=True)
