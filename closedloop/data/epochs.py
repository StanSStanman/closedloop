import os
import os.path as op
import numpy as np
import mne
import json
import warnings
import scipy as sp
import autoreject


def detrend(data, type='linear', bp=0):
    # data should be (nchans, ntimes)            
    if isinstance(data, (mne.io.Raw, mne.io.BaseRaw)):
        data._data = sp.signal.detrend(data.get_data(), axis=-1, type=type, 
                                       bp=bp)
    elif isinstance(data, (np.ndarray)):
        data = sp.signal.detrend(data, axis=-1, type=type, bp=bp)
    else:
        raise TypeError('data should be a mne Raw or numpy array type object')
    
    return data


def sw_epochs(raw_fname, eve_fname, epo_fname, sfreq=100., picks=None,
              eve_id=None, tmin=-2., tmax=3., fmin=.5, fmax=40., 
              reject={'eeg': 500e-6}, baseline=None, add_first_samp=True):
    
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    # raw.set_eeg_reference(ref_channels=['L4H', 'R4H'])
    
    # raw.load_data()
    # raw = detrend(raw)
    raw.filter(fmin, fmax, n_jobs='cuda')
    
    try:
        eve = mne.read_events(eve_fname)
    except Exception:
        print('No events, passing to the next one...')
        return
    
    raw.resample(sfreq, n_jobs=8)
    
    if eve_id is None:
        id_fname = eve_fname.replace('.fif', '_id.json')
        try:
            with open(id_fname, 'r') as f:
                eve_id = json.load(f)
        except Exception:
            warnings.warn('No events id specified')
    
    # Add first_samp to events time points to retrieve original time points
    if add_first_samp:
        eve[:, 0] += raw.first_samp
        
    if reject == 'autoreject':
        epochs = mne.Epochs(raw, eve, event_id=eve_id, tmin=tmin, tmax=tmax, 
                            baseline=baseline, picks=picks, preload=False, 
                            reject=None, flat=None, proj=True, decim=1, 
                            reject_tmin=None, reject_tmax=None, detrend=None, 
                            on_missing='raise', reject_by_annotation=True, 
                            metadata=None, event_repeated='error', 
                            verbose=None)
        epochs.load_data()
        ar = autoreject.AutoReject(picks='eeg', n_jobs=16)
        epochs = ar.fit_transform(epochs)
    else:
        # reject = {'eeg': 500e-6} # Maybe 500?
        # reject = None
        epochs = mne.Epochs(raw, eve, event_id=eve_id, tmin=tmin, tmax=tmax, 
                            baseline=baseline, picks=picks, preload=False, 
                            reject=reject, flat=None, proj=True, decim=1, 
                            reject_tmin=None, reject_tmax=None, detrend=None, 
                            on_missing='raise', reject_by_annotation=True, 
                            metadata=None, event_repeated='error', 
                            verbose=None)    
    
    # TODO
    # Check if there are bad epochs excluded from epochs but not from events
    
    # epochs.load_data().filter(.5, 4., n_jobs=32)
    
    epochs.save(epo_fname, overwrite=True, split_naming='neuromag')
    print('Epochs saved at:', epo_fname)
    
    return epochs
    

if __name__ == '__main__':
    
    prj_data = '/home/ruggero.basanisi/data/tweakdreams'

    data_dir = prj_data
    subjects = ['TD001']
    nights = ['N3']
    
    for sbj in subjects:
        for n in nights:
            
            prep_dir = op.join(prj_data, 'mne', '{0}', '{1}', 
                             'prep').format(sbj, n)
            eve_dir = op.join(prj_data, 'mne', '{0}', '{1}', 
                             'eve').format(sbj, n)
            epo_dir = op.join(prj_data, 'mne', '{0}', '{1}', 
                             'epo').format(sbj, n)

            aw = [a for a in os.listdir(eve_dir) if a.startswith('aw_')]
            aw.sort()
            # aw = ['aw_2']
            
            for _aw in aw:
                
                _prep_dir = op.join(prep_dir, _aw)
                _eve_dir = op.join(eve_dir, _aw)
                _epo_dir = op.join(epo_dir, _aw)
                
                os.makedirs(_epo_dir, exist_ok=True)
                
                prep_fname = op.join(_prep_dir, f'{sbj}_{n}_prep-raw.fif')
                eve_fname = op.join(_eve_dir, 'envelope_sw-eve.fif')
                epo_fname = op.join(_epo_dir, 'envelope_sw-epo.fif')
                
                # Making epochs the first time (not clean)
                sw_epochs(prep_fname, eve_fname, epo_fname, sfreq=100., 
                          picks=None, eve_id=None, tmin=-2., tmax=2., 
                          fmin=.3, fmax=40., reject={'eeg': 500e-6},
                          baseline=None)

                # Making clean sw events epochs
                # cl_eve_fname = op.join(_eve_dir, 'envelope_sw_clean-eve.fif')
                # cl_epo_fname = op.join(_epo_dir, 'envelope_sw_clean-epo.fif')
                # if op.exists(cl_eve_fname):
                #     sw_epochs(prep_fname, cl_eve_fname, cl_epo_fname, 
                #               sfreq=100., picks=None, eve_id=None, tmin=-8., 
                #               tmax=2., fmin=.5, fmax=4., reject=None, 
                #               baseline=None, add_first_samp=False)
