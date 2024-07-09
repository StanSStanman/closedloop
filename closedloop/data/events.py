import os
import os.path as op  
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


def erase_evk_eve(prep_fname, eve_fname, eve_id, sfreq=100., t_dist=5.):
    tp_dist = int(sfreq * t_dist)
    
    prep = mne.io.read_raw_fif(prep_fname, preload=True)
    prep = prep.resample(sfreq)
    
    try:
        events = mne.read_events(eve_fname)
    except Exception:
        print('No events in', eve_fname)
        return
    
    prp_eve, prp_ev_id = mne.events_from_annotations(prep, event_id=eve_id)
    prp_eve[:, 0] -= prep.first_samp
    
    rej_eve = []
    for i, e in enumerate(events[:, 0]):
        diff = e - prp_eve[:, 0]
        # print(diff)
        if any(np.logical_and(0 <= diff, diff <= tp_dist)):
            rej_eve.append(i)
            
    events = np.delete(events, rej_eve, axis=0)
    print('Rejected events:', rej_eve)
    
    mne.write_events(eve_fname, events, overwrite=True)
    print('Events written in:', eve_fname, '\n')
    return


if __name__ == '__main__': 
    
    prj_data = '/home/ruggero.basanisi/data/tweakdreams'

    data_dir = prj_data
    subjects = ['TD001']
    nights = ['N3']
    
    ev_id = {'Stimulus/s20': 20,
             'Stimulus/s30': 30,
             'Stimulus/s40': 40,
             'Stimulus/s22': 22,
             'Stimulus/s24': 24,
             'Stimulus/s26': 26,
             'Stimulus/s28': 28}
    
    for sbj in subjects:
        for n in nights:
            
            prep_dir = op.join(prj_data, 'mne', '{0}', '{1}', 
                               'prep').format(sbj, n)
            eve_dir = op.join(prj_data, 'mne', '{0}', '{1}', 
                             'eve').format(sbj, n)
            
            aw = [a for a in os.listdir(eve_dir) if a.startswith('aw_')]
            aw.sort()
            # aw = ['aw_2']
            
            for _aw in aw:
                
                prep_fname = op.join(prep_dir, _aw, f'{sbj}_{n}_prep-raw.fif')
                sws_fname = op.join(eve_dir, _aw, 'envelope_sw.csv')
                eve_fname = op.join(eve_dir, _aw, 'envelope_sw-eve.fif')
                
                make_sw_events(sws_fname, eve_fname, save_id=True)
                
                erase_evk_eve(prep_fname, eve_fname, ev_id, 
                              sfreq=100., t_dist=1.)
