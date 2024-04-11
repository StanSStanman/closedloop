import os.path as op
import numpy as np
import mne
import json
import warnings

from read_mne import read_raw, read_eve, resample_raw
# from sleep import sleep_staging, sw_detection


def sw_epochs(raw_fname, eve_fname, epo_fname, sfreq=100., picks=None,
              eve_id=None, tmin=-2., tmax=3., baseline=None):
    
    raw = mne.io.read_raw_fif(raw_fname)
    eve = mne.read_events(eve_fname)
    
    raw.resample(sfreq, n_jobs=8)
    
    if eve_id is None:
        id_fname = eve_fname.replace('.fif', '_id.json')
        try:
            with open(id_fname, 'r') as f:
                eve_id = json.load(f)
        except Warning:
            warnings.warn('No events id specified')
    
    epochs = mne.Epochs(raw, eve, event_id=eve_id, tmin=tmin, tmax=tmax, 
                        baseline=baseline, picks=picks, preload=False, 
                        reject=None, flat=None, proj=True, decim=1, 
                        reject_tmin=None, reject_tmax=None, detrend=None, 
                        on_missing='raise', reject_by_annotation=True, 
                        metadata=None, event_repeated='error', verbose=None)
    
    epochs.save(epo_fname, overwrite=True, split_naming='neuromag')
    
    return epochs
    

if __name__ == '__main__':
    raw_fname = '/home/jerry/python_projects/space/closedloop/test_data/n1_raw.fif'
    eve_fname = '/home/jerry/python_projects/space/closedloop/closedloop/sw_detect/sw-eve.fif'
    epo_fname = '/home/jerry/python_projects/space/closedloop/test_data/n1_epo.fif'
    
    sw_epochs(raw_fname, eve_fname, epo_fname, sfreq=100., picks=None,
              eve_id=None, tmin=-2., tmax=3., baseline=None)