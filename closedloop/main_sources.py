import sys
sys.path.append("/home/ruggero.basanisi/python_projects/space/closedloop/closedloop/data")
sys.path.append("/home/ruggero.basanisi/python_projects/space/closedloop/closedloop/sw_detect")

import os
import os.path as op
import mne
import pickle

from data.signal_se import (compute_inverse_sources, labeling)
from data.sw_origins import (find_sw_origin, compute_sources_neighbors, 
                             compute_sw_origin)

def sw_sources_pipeline(prj_data, subject, night):
    
    fs_dir = op.join(prj_data, 'freesurfer')
    epo_dir = op.join(prj_data, 'mne', subject, night, 'epo')
    bem_dir = op.join(prj_data, 'mne', subject, night, 'bem')
    fwd_dir = op.join(prj_data, 'mne', subject, night, 'fwd')
    src_dir = op.join(prj_data, 'mne', subject, night, 'src')

    ltc_dir = op.join(prj_data, 'mne', subject, night, 'ltc')
    
    fwd_fname = op.join(fwd_dir, f'{subject}_{night}-fwd.fif')
    src_fname = op.join(src_dir, f'{subject}_{night}-src.fif')
    
    aw = [a for a in os.listdir(epo_dir) if a.startswith('aw_')]
    aw.sort()
    # aw = ['aw_3'] # Debugging purpose only

    src_neighbors = ()
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
            
            stc = compute_inverse_sources(epo_fname, bln_fname,
                                          fwd_fname)
            
            if len(src_neighbors) == 0:
                src_neighbors = (
                    compute_sources_neighbors(subject, fs_dir, src_fname,
                                              radius=0.02, n_jobs=15))
                
            pnt_orig, sph_orig, geo_orig = (
                compute_sw_origin(subject, fs_dir, stc, src_fname, 
                                  src_neighbors[0], src_neighbors[1], 
                                  t_dist=.05, value='abs'))
            
            # pnt_orig, sph_orig, geo_orig = (
            #     find_sw_origin(subject, fs_dir, stc, src_fname, radius=0.02, 
            #                    t_dist=.05, value='abs', n_jobs=20)              # PUT BACK N_JOBS=80
            #     )
            
            pnt_fname = op.join(ltc_dir, _aw, 'surf_point_origins.txt')
            sph_fname = op.join(ltc_dir, _aw, 'spherical_origins.txt')
            geo_fname = op.join(ltc_dir, _aw, 'geodesic_origins.txt')
            
            with open(pnt_fname, 'wb') as f:
                pickle.dump(pnt_orig, f)
                
            with open(sph_fname, 'wb') as f:
                pickle.dump(sph_orig, f)
            
            with open(geo_fname, 'wb') as f:
                pickle.dump(geo_orig, f)
                
    print('Done!')
                
    return


if __name__ == '__main__':
    
    prj_data = '/home/ruggero.basanisi/data/tweakdreams'
    
    # subjects = ['TD001', 'TD005', 'TD009', 'TD010', 'TD011']
    subjects = ['TD022']
    # nights = ['N1', 'N2', 'N3', 'N4']
    nights = ['N3', 'N4']

    for sbj in subjects:
        for n in nights:
            sw_sources_pipeline(prj_data, sbj, n)
