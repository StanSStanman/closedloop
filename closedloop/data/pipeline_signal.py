# import sys
# sys.path.append("/home/ruggero.basanisi/python_projects/space/closedloop/closedloop/data")
# sys.path.append("/home/ruggero.basanisi/python_projects/space/closedloop/closedloop/sw_detect")

import os
import os.path as op
# import numpy as np
import pickle

from closedloop.data.utils.utils import fname_finder
from closedloop.data.utils.read_brainvision import brainvision_to_mne
from closedloop.data.sleep.staging import compute_staging
from closedloop.data.signal.preprocessing import do_preprocessing
from closedloop.data.sleep.sw_detection import (detect_sw, compute_stage_envelope)
from closedloop.data.signal.events import (make_sw_events, erase_evk_eve)
from closedloop.data.signal.epochs import sw_epochs
from closedloop.data.sleep.sw_correct import realign_sw_epo


def set_mne_dataset(prj_data, subject, night):
    _vhdr = op.join(data_dir, f'{subject}', f'{subject}_{night}', 'eeg', 
                    '*.vhdr')
    vhdr_fnames = fname_finder(_vhdr)
    vhdr_fnames.sort()
    
    _elc = op.join(data_dir, f'{subject}', f'{subject}_{night}', 'eeg', 
                   f'{subject}_{night}*.elc')
    elc_fname = fname_finder(_elc)[0]
    
    events_id = {'Stimulus/s20': 20,
                'Stimulus/s30': 30,
                'Stimulus/s40': 40,
                'Stimulus/s22': 22,
                'Stimulus/s24': 24,
                'Stimulus/s26': 26,
                'Stimulus/s28': 28}
    
    raw_dir = op.join(prj_data, 'mne', f'{subject}', f'{night}', 'raw')
    eve_dir = op.join(prj_data, 'mne', f'{subject}', f'{night}', 'eve')
    # prep_dir = op.join(prj_data, 'mne', f'{subject}', f'{night}', 'prep')
    # epo_dir = op.join(prj_data, 'mne', f'{subject}', f'{night}', 'epo')
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(eve_dir, exist_ok=True)
    # os.makedirs(prep_dir, exist_ok=True)
    # os.makedirs(epo_dir, exist_ok=True)
    
    fif_fname = f'{subject}_{night}'
        
    brainvision_to_mne(vhdr_fnames, elc_fname, events_id, raw_dir, fif_fname, 
                       divide_by='nights')
    brainvision_to_mne(vhdr_fnames, elc_fname, events_id, raw_dir, fif_fname, 
                       divide_by='awakenings')
    
    # wholenight_raw_fname = op.join(raw_dir, f'{subject}_{night}-raw.fif')
    # stg_fname = op.join(eve_dir, 'hypnogram.npy')
    # compute_staging(wholenight_raw_fname, stg_fname)
    
    return


def signal_processing_pipeline(prj_data, subject, night, aw=None):
    
    events_id = {'Stimulus/s20': 20,
                'Stimulus/s30': 30,
                'Stimulus/s40': 40,
                'Stimulus/s22': 22,
                'Stimulus/s24': 24,
                'Stimulus/s26': 26,
                'Stimulus/s28': 28}
    
    raw_dir = op.join(prj_data, 'mne', f'{subject}', f'{night}', 'raw')
    eve_dir = op.join(prj_data, 'mne', f'{subject}', f'{night}', 'eve')
    prep_dir = op.join(prj_data, 'mne', f'{subject}', f'{night}', 'prep')
    epo_dir = op.join(prj_data, 'mne', f'{subject}', f'{night}', 'epo')
    
    # os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(eve_dir, exist_ok=True)
    os.makedirs(prep_dir, exist_ok=True)
    os.makedirs(epo_dir, exist_ok=True)
    
    wholenight_raw_fname = op.join(raw_dir, f'{subject}_{night}-raw.fif')
    stg_fname = op.join(eve_dir, 'hypnogram.npy')
    compute_staging(wholenight_raw_fname, stg_fname)
    
    if aw is None:
        aw = [a for a in os.listdir(raw_dir) if a.startswith('aw_')]
        aw.sort()
    # aw = ['aw_3'] # Debugging purpose only
        
    for _aw in aw:
        
        _raw_dir = op.join(raw_dir, _aw)
        _eve_dir = op.join(eve_dir, _aw)
        _prep_dir = op.join(prep_dir, _aw)
        _epo_dir = op.join(epo_dir, _aw)
        
        eve_fname = op.join(_eve_dir, 'envelope_sw-eve.fif')
        prep_fname = op.join(_prep_dir, f'{subject}_{night}_prep-raw.fif')
        epo_fname = op.join(_epo_dir, 'envelope_sw-epo.fif')
        chs_fname = op.join(_eve_dir, 'maxneg_channels.txt')
        swt_fanme = op.join(_eve_dir, 'sw_types.txt')
        cl_eve_fname = op.join(_eve_dir, 'envelope_sw_clean-eve.fif')
        cl_epo_fname = op.join(_epo_dir, 'envelope_sw_clean-epo.fif')
        
        do_preprocessing(_raw_dir, resample=100., components=.99, 
                         bad_ch_exist=False, restrict=True, save=True)
        
        if not op.exists(_raw_dir):
            continue
        
        stg_fname = op.join(eve_dir, 'hypnogram.npy')
        envp, ch_names, sfreq, hypno = compute_stage_envelope(prep_fname, 
                                                              stg_fname)
              
        df = detect_sw(envp, sfreq=sfreq, hypno=hypno, stgs=[2, 3], 
                       ch_names=ch_names, half_wlen=(0.12, 1.05),
                       neg_amp=(5.e-6, 200.e-6), 
                       pos_amp=(0, 100.e-6),
                       n_jobs='cuda')
        
        sws_fname = op.join(_eve_dir, 'envelope_sw.csv')
        df.to_csv(sws_fname)
        print('Saved to:', sws_fname)
        
        make_sw_events(sws_fname, eve_fname, save_id=True)
        
        erase_evk_eve(prep_fname, eve_fname, events_id, sfreq=100., t_dist=1.)
        
        os.makedirs(_epo_dir, exist_ok=True)
        
        sw_epochs(prep_fname, eve_fname, epo_fname, sfreq=100., 
                  picks=None, eve_id=None, tmin=-2., tmax=2., 
                  fmin=.3, fmax=40., baseline=None)
        
        # The only n_jobs that does not use cuda
        rsw = realign_sw_epo(epo_fname, eve_fname, n_jobs=8)
        
        if rsw is not None:
            gt, bt, mnch, swt = rsw
        
            with open(chs_fname, 'wb') as f:
                pickle.dump(mnch, f)
                
            with open(swt_fanme, 'wb') as f:
                pickle.dump(swt, f)
                
        if op.exists(cl_eve_fname):
            sw_epochs(prep_fname, cl_eve_fname, cl_epo_fname, 
                      sfreq=100., picks=None, eve_id=None, tmin=-8., 
                      tmax=2., fmin=.5, fmax=4., reject=None, 
                      baseline=None, add_first_samp=False)
        
    return
        
        
    

if __name__ == '__main__':
    
    prj_data = '/home/ruggero.basanisi/data/tweakdreams'

    data_dir = prj_data
    # subjects = ['TD001', 'TD005', 'TD009', 'TD010', 'TD011', 
    #             'TD022', 'TD026', 'TD028', 'TD029', 'TD034']
    subjects = ['TD022', 'TD026', 'TD028', 'TD029', 'TD034']
    subjects = ['TD034']
    nights = ['N1', 'N2', 'N3', 'N4']
    nights = ['N3', 'N4']

    for sbj in subjects:
        for n in nights:
            set_mne_dataset(prj_data, sbj, n)
            signal_processing_pipeline(prj_data, sbj, n)
