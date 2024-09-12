import os
import os.path as op

from closedloop.data.anat.freesurfer_meshes import (compute_freesurfer,
                                    compute_watershed_bem,
                                    compute_scalp_meshes)
from closedloop.data.anat.bem import compute_bem
from closedloop.data.sources.source_space import compute_source_space
from closedloop.data.sources.forward_solution import compute_forward_model


def freesurfer_pipeline(fs_home, prj_data, subject):
    
    fs_sbj_dir = op.join(prj_data, 'freesurfer')
    
    os.makedirs(fs_sbj_dir, exist_ok=True)

    mri_fname = op.join(prj_data, 'mri', 
                        f'sub-{subject.lower()}_ses-d01_mri.nii')
    
    compute_freesurfer(fs_home, fs_sbj_dir, subject, mri_fname, n_jobs=16)
    compute_watershed_bem(fs_home, fs_sbj_dir, subject)
    compute_scalp_meshes(fs_home, fs_sbj_dir, subject)
    
    return
    
    
def anatomy_pipeline(prj_data, subject, night):
    
    fs_sbj_dir = op.join(prj_data, 'freesurfer')
    bem_dir = op.join(prj_data, 'mne', subject, night, 'bem')
    src_dir = op.join(prj_data, 'mne', subject, night, 'src')
    info_dir = op.join(prj_data, 'mne', subject, night, 'raw', 'aw_0')
    trans_dir = op.join(prj_data, 'mne', subject, night, 'trans')
    fwd_dir = op.join(prj_data, 'mne', subject, night, 'fwd')
    
    raw_dir = op.join(prj_data, 'mne', subject, night, 'raw')
    if op.exists(op.join(raw_dir, 'aw_0')):
        info_dir = op.join(raw_dir, 'aw_0')
    else:
        info_dir = op.join(raw_dir, 'aw_1')
    
    os.makedirs(bem_dir, exist_ok=True)
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(fwd_dir, exist_ok=True)
    
    bem_fname = op.join(bem_dir, f'{subject}_{night}-bem-sol.fif')
    src_fname = op.join(src_dir, f'{subject}_{night}-src.fif')
    info_fname = op.join(info_dir, f'{subject}_{night}-raw.fif')
    trans_fname = op.join(trans_dir, f'{subject}_{night}-trans.fif')
    fwd_fname = op.join(fwd_dir, f'{subject}_{night}-fwd.fif')
    
    # MNE anatomy pipeline
    compute_bem(subject, fs_sbj_dir, bem_fname)
    compute_source_space(subject, fs_sbj_dir, src_fname, spacing=6, n_jobs=32)
    compute_forward_model(info_fname, trans_fname, src_fname, 
                          bem_fname, fwd_fname, n_jobs=32)

    return

if __name__ == '__main__':
    
    prj_data = '/home/ruggero.basanisi/data/tweakdreams'

    data_dir = prj_data
    # subjects = ['TD001', 'TD005', 'TD009', 'TD010', 'TD011']
    subjects = ['TD028']
    nights = ['N1', 'N2', 'N3', 'N4']
    # nights = ['N4']

    fs_home = '/home/programmi/freesurfer'
    
    for sbj in subjects:
        for n in nights:
            anatomy_pipeline(prj_data, sbj, n)
