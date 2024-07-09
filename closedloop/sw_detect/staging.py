import os 
import os.path as op
import mne 
import yasa
import numpy as np


def compute_staging(raw_fname, stg_fname):
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    raw.set_eeg_reference(ref_channels=['L4H', 'R4H'])
    sls = yasa.SleepStaging(raw, eeg_name='Z6Z', eog_name='HEOG', 
                            emg_name='EMG', metadata=None)
    hypno = sls.predict()
    
    np.save(stg_fname, hypno)
    
    return hypno


def crop_hypno(stg_fname, sfreq, min_samp, max_samp, ret_type='int'):
    hypno = np.load(stg_fname, allow_pickle=True)
    hypno = yasa.hypno_upsample_to_sf(hypno, sf_hypno=1/30, sf_data=sfreq)
    hypno = hypno[min_samp:max_samp]
    if ret_type == 'int':
        hypno = yasa.hypno_str_to_int(hypno)
    return hypno


if __name__ == '__main__':
    
    prj_data = '/home/ruggero.basanisi/data/tweakdreams'

    data_dir = prj_data
    subjects = ['TD001']
    nights = ['N1']
    
    for sbj in subjects:
        for n in nights:
            raw_dir = op.join(prj_data, 'mne', '{0}', '{1}', 
                                'raw').format(sbj, n)
            eve_dir = op.join(prj_data, 'mne', '{0}', '{1}', 
                                'eve').format(sbj, n)
            
            os.makedirs(eve_dir, exist_ok=True)
            
            raw_fname = op.join(raw_dir, f'{sbj}_{n}-raw.fif')
            stg_fname = op.join(eve_dir, 'hypnogram.npy')
            
            compute_staging(raw_fname, stg_fname)
