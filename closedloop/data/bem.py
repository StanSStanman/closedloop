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
    
    subject = 'TD001'
    subjects_dir = '/home/jerry/freesurfer/TweakDreams'
    bem_fname = ('/media/jerry/ruggero/dataset_td02/mne/TD001/n1/bem/'
                 'TD001-bem-sol.fif')
    compute_bem(subject, subjects_dir, bem_fname)
