import os
import os.path as op
import mne
import pickle

from data.signal_se import (compute_inverse_sources, labeling)
from data.sw_origins import find_sw_origin

def sw_sources_pipeline(prj_data, subject, night):
    
    fs_dir = op.join(prj_data, 'freesurfer')
    epo_dir = op.join(prj_data, 'mne', subject, night, 'epo')
    bem_dir = op.join(prj_data, 'mne', subject, night, 'bem')
    fwd_dir = op.join(prj_data, 'mne', subject, night, 'fwd')
    src_dir = op.join(prj_data, 'mne', subject, night, 'src')

    ltc_dir = op.join(prj_data, 'mne', subject, night, 'ltc')
    
    fwd_fname = op.join(fwd_dir, f'{subject}_{night}-fwd.fif')
    src_fname = op.join(src_dir, f'{subject}_{night}-src.fif')
    sph_fname = op.join(ltc_dir, 'spherical_origins.txt')
    geo_fname = op.join(ltc_dir, 'geodesic_origins.txt')
    
    aw = [a for a in os.listdir(epo_dir) if a.startswith('aw_')]
    aw.sort()

    for _aw in aw:
        epo_fname = op.join(epo_dir, _aw, 'envelope_sw_clean-epo.fif')
        bln_fname = epo_fname

        ltc_fname = op.join(ltc_dir, _aw, 'sws_labels_tc.nc')
        
        if op.exists(epo_fname):
            os.makedirs(op.join(ltc_dir, _aw), exist_ok=True)
            
            stc = compute_inverse_sources(epo_fname, bln_fname, 
                                            fwd_fname)
            
            events = mne.read_epochs(epo_fname).events[:, -1]
            labeling(subject, fs_dir, stc, src_fname, ltc_fname, events)
            
            sph_orig, geo_orig = find_sw_origin(subject, fs_dir, stc, 
                                                src_fname, radius=0.02, 
                                                t_dist=.05, value='abs', 
                                                n_jobs=64)
            with open(sph_fname, 'wb') as f:
                pickle.dump(sph_orig, f)
            
            with open(geo_fname, 'wb') as f:
                pickle.dump(geo_orig, f)
                
    return


if __name__ == '__main__':
    
    prj_data = '/home/ruggero.basanisi/data/tweakdreams'
    
    # subjects = ['TD001', 'TD005', 'TD009', 'TD010', 'TD011']
    subjects = ['TD001', 'TD005']
    # nights = ['N1', 'N2', 'N3', 'N4']
    nights = ['N1']

    for sbj in subjects:
        for n in nights:
            sw_sources_pipeline(prj_data, sbj, n)
