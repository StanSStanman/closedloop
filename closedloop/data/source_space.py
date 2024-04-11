import mne


def compute_source_space(subject, sbj_dir, src_fname, spacing='oct6'):
    """Compute the source space

    Args:
        subject (str): name of the subject
        sbj_dir (str): freesurfer subjects folder
        src_fname (str): name of the path were the source space will be saved
        spacing (str, optional): verices spacing, the lower the number, 
            the less sources will be computed. Defaults to 'oct6'.
    """
    src = mne.setup_source_space(subject, spacing=spacing, add_dist=True,
                                 subjects_dir=sbj_dir, n_jobs=-1)

    src.save(src_fname, overwrite=True)

    return


if __name__ == '__main__':
    subject = 'TD001'
    subjects_dir = '/home/jerry/freesurfer/TweakDreams'
    src_fname = '/media/jerry/ruggero/dataset_td02/mne/TD001/n1/src/TD001-src.fif'
    
    compute_source_space(subject, subjects_dir, src_fname, spacing='oct5')
