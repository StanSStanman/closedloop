import mne


def compute_bem(subject, sbj_dir, bem_fname):
    """Compute and save BEM model and BEM solution

    Args:
        subject (path_like): name of the subject
        sbj_dir (path_like): freesurfer subjects folder
        bem_fname (path_like): name of the path were the BEM solution will 
            be saved
    """
    bem_model = mne.make_bem_model(subject=subject, subjects_dir=sbj_dir,
                                   conductivity=[0.3, 0.006, 0.3])
    bem_solution = mne.make_bem_solution(bem_model)
    mne.write_bem_solution(bem_fname, bem_solution, overwrite=True)

    return


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
            
            os.makedirs(bem_dir, exist_ok=True)
    
            bem_fname = op.join(bem_dir, f'{sbj}_{n}-bem-sol.fif')
    
            compute_bem(sbj, fs_dir, bem_fname)
