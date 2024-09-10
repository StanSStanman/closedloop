import mne


def compute_forward_model(info_fname, trans_fname, src_fname,
                          bem_fname, fwd_fname, n_jobs=32):
    """_summary_

    Args:
        info_fname (str): path of the file containg info, 
            can be a mne raw or epochs file
        trans_fname (str): path to the transformation matrix file
        src_fname (str): path to the source space file
        bem_fname (str): path to the BEM model file
        fwd_fname (str): name of the path were the forward solution 
            will be saved
    """
    if '-raw' in info_fname:
        info = mne.io.read_raw_fif(info_fname).info
    elif '-epo' in info_fname:
        info = mne.read_epochs(info_fname).info
    trans = mne.read_trans(trans_fname)
    src = mne.read_source_spaces(src_fname)
    bem = mne.read_bem_solution(bem_fname)

    fwd = mne.make_forward_solution(info=info, trans=trans,
                                    src=src, bem=bem,
                                    meg=False, eeg=True,
                                    mindist=0.0, n_jobs=n_jobs, verbose=False)

    mne.write_forward_solution(fwd_fname, fwd, overwrite=True)

    return


if __name__ == '__main__':
    
    import os
    import os.path as op
    
    prj_data = '/home/ruggero.basanisi/data/tweakdreams'
    
    subjects = ['TD005']
    nights = ['N1']
    
    for sbj in subjects:
        for n in nights:
            
            info_dir = op.join(prj_data, 'mne', sbj, n, 'raw', 'aw_0')
            trans_dir = op.join(prj_data, 'mne', sbj, n, 'trans')
            bem_dir = op.join(prj_data, 'mne', sbj, n, 'bem')
            src_dir = op.join(prj_data, 'mne', sbj, n, 'src')
            fwd_dir = op.join(prj_data, 'mne', sbj, n, 'fwd')
            
            os.makedirs(fwd_dir, exist_ok=True)

            # TODO correct all the raws filenames
            info_fname = op.join(info_dir, f'{sbj}_{n}-raw.fif')
            trans_fname = op.join(trans_dir, f'{sbj}_{n}-trans.fif')
            bem_fname = op.join(bem_dir, f'{sbj}_{n}-bem-sol.fif')
            src_fname = op.join(src_dir, f'{sbj}_{n}-src.fif')
            fwd_fname = op.join(fwd_dir, f'{sbj}_{n}-fwd.fif')
            
            compute_forward_model(info_fname, trans_fname, src_fname,
                                  bem_fname, fwd_fname)
