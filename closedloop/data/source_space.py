import mne


def compute_source_space(subject, sbj_dir, src_fname, spacing='oct6', 
                         n_jobs=32):
    """Compute the source space

    Args:
        subject (str): name of the subject
        sbj_dir (str): freesurfer subjects folder
        src_fname (str): name of the path were the source space will be saved
        spacing (str, optional): verices spacing, the lower the number, 
            the less sources will be computed. Defaults to 'oct6'.
    """
    src = mne.setup_source_space(subject, spacing=spacing, add_dist=True,
                                 subjects_dir=sbj_dir, n_jobs=n_jobs)

    src.save(src_fname, overwrite=True)

    return src


def compute_vol_source_space(subject, sbj_dir, bem_fname, src_fname, 
                             spacing=5):
    """Compute the source space

    Args:
        subject (str): name of the subject
        sbj_dir (str): freesurfer subjects folder
        src_fname (str): name of the path were the source space will be saved
        spacing (str, optional): verices spacing, the lower the number, 
            the less sources will be computed. Defaults to 'oct6'.
    """
    bem = mne.read_bem_solution(bem_fname)
    
    src = mne.setup_volume_source_space(subject, pos=spacing, mri=None, 
                                        sphere=None, bem=bem, surface=None,
                                        mindist=5.0, exclude=0.0,
                                        subjects_dir=sbj_dir, n_jobs=-1)

    src.save(src_fname, overwrite=True)

    return src


def compute_vol_source_space_srf(subject, sbj_dir, srf_fname, src_fname, 
                                 spacing=5):
    """Compute the source space

    Args:
        subject (str): name of the subject
        sbj_dir (str): freesurfer subjects folder
        src_fname (str): name of the path were the source space will be saved
        spacing (str, optional): verices spacing, the lower the number, 
            the less sources will be computed. Defaults to 'oct6'.
    """
    
    src_h = []
    for srf in srf_fname:
        
        src_h.append(mne.setup_volume_source_space(subject, pos=spacing, mri=None, 
                                            sphere=None, bem=None, surface=srf,
                                            mindist=5.0, exclude=0.0,
                                            subjects_dir=sbj_dir, n_jobs=-1))
        
    src = src_h[0] + src_h[1]

    src.save(src_fname, overwrite=True)

    return src


if __name__ == '__main__':
    
    import os
    import os.path as op
    
    prj_data = '/home/ruggero.basanisi/data/tweakdreams'
    
    subjects = ['TD005']
    nights = ['N1']
    
    fs_dir = op.join(prj_data, 'freesurfer')
    
    for sbj in subjects:
        for n in nights:
            
            bem_dir = op.join(prj_data, 'mne', sbj, n, 'bem')
            srf_dir = op.join(fs_dir, sbj, 'surf')
            src_dir = op.join(prj_data, 'mne', sbj, n, 'src')
            
            os.makedirs(src_dir, exist_ok=True)

            bem_fname = op.join(bem_dir, f'{sbj}_{n}-bem-sol.fif')
            srf_lh = op.join(srf_dir, 'lh.pial')
            srf_rh = op.join(srf_dir, 'rh.pial')
            src_fname = op.join(src_dir, f'{sbj}_{n}-src.fif')
    
            compute_source_space(sbj, fs_dir, src_fname, spacing=10)
            # compute_vol_source_space(sbj, fs_dir, bem_fname, src_fname, 
            #                          spacing=5)
            # compute_vol_source_space_srf(sbj, fs_dir, [srf_lh, srf_rh], 
            #                              src_fname, spacing=5)
