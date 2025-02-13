import numpy as np
from mne.channels.montage import make_dig_montage
import h5py
import glob
import mne
import os
import os.path as op
from scipy.io import loadmat


def open_matfile(mat_fname):
    try:
        mat = loadmat(mat_fname)
        print('Loaded mat file <= v7.3...\n')
    except:
        mat = h5py.File(mat_fname)
        print('Loaded mat file >= v7.3...\n')
    return mat


def fname_finder(pathname):
    return glob.glob(pathname)


def brainvision_reader(vhdr_fname):
    scale = 1.
    # scale = 1e-2
    try:
        raw = mne.io.read_raw_brainvision(vhdr_fname, scale=scale, 
                                          preload=False)
    except FileNotFoundError:
        print('Changing temporarely vmrk and eeg file name:')
        vmrk_fname = vhdr_fname.replace('.vhdr', '.vmrk')
        print('From... ', vmrk_fname)
        vmrk_split = vmrk_fname.split('/')
        vmrk_split[-1] = vmrk_split[-1].replace(vmrk_split[-4], '')
        vmrk_split[0] = '/'
        new_vmrk_fname = op.join(*vmrk_split)
        os.rename(vmrk_fname, new_vmrk_fname)
        print('To... ', new_vmrk_fname)
        eeg_fname = vhdr_fname.replace('.vhdr', '.eeg')
        print('From... ', eeg_fname)
        eeg_split = eeg_fname.split('/')
        eeg_split[-1] = eeg_split[-1].replace(eeg_split[-4], '')
        eeg_split[0] = '/'
        new_eeg_fname = op.join(*eeg_split)
        os.rename(eeg_fname, new_eeg_fname)
        print('To... ', new_eeg_fname)
        raw = mne.io.read_raw_brainvision(vhdr_fname, scale=scale, 
                                          preload=False, verbose=False)
        print('Reversing vmrk and eeg file names...')
        os.rename(new_vmrk_fname, vmrk_fname)
        os.rename(new_eeg_fname, eeg_fname)
    finally:
        return raw
    

def brainvision_loader(raw):
    try:
        raw.load_data()
    except FileNotFoundError:
        new_eeg_fname = raw.filenames[0]
        eeg_split = new_eeg_fname.split('/')
        eeg_split[-1] = eeg_split[-4] + eeg_split[-1]
        eeg_split[0] = '/'
        eeg_fname = op.join(*eeg_split)
        os.rename(eeg_fname, new_eeg_fname)
        raw.load_data(verbose=False)
        os.rename(new_eeg_fname, eeg_fname)
    finally:
        return


def read_elc(fname, head_size=None):
    """Read .elc files.

    Parameters
    ----------
    fname : str
        File extension is expected to be '.elc'.
    head_size : float | None
        The size of the head in [m]. If none, returns the values read from the
        file with no modification.

    Returns
    -------
    montage : instance of DigMontage
        The montage in [m].
    """
    fid_names = ('Nz', 'LPA', 'RPA')

    # ch_names_, pos = [], []
    with open(fname) as fid:
        # _read_elc does require to detect the units. (see _mgh_or_standard)
        for line in fid:
            if 'UnitPosition' in line:
                units = line.split()[-1]
                scale = dict(m=1., mm=1e-3)[units]
                break
        else:
            raise RuntimeError('Could not detect units in file %s' % fname)
        for line in fid:
            if 'Positions\n' in line:
                break
        pos = []
        for line in fid:
            if 'Labels\n' in line:
                break
            pos.append(list(map(float, line.split()[-3:])))
        for line in fid:
            # if not line or not set(line) - {' '}:
            #     break
            if 'NumberHeadShapePoints' in line:
                break
            # ch_names_.append(line.strip('\n').split())
            ch_names_ = line.strip('\n').split()
        for line in fid:
            if 'UnitHeadShapePoints' in line:
                hs_units = line.split()[-1]
                hs_scale = dict(m=1., mm=1e-3)[hs_units]
                break
        hs_pos = []
        for line in fid:
            if 'HeadShapePoints' in line:
                continue
            hs_pos.append(list(map(float, line.split())))
            
    # Check for blank lines at the end of the file
    hs_pos = [hsp for hsp in hs_pos if len(hsp) != 0]            

    pos = np.array(pos) * scale
    # only include head shape points if they are present
    if len(hs_pos) != 0:
        hs_pos = np.array(hs_pos) * hs_scale
        if head_size is not None:
            pos *= head_size / np.median(np.linalg.norm(pos, axis=1))

        # ch_pos = _check_dupes_odict(ch_names_, pos)
        # nasion, lpa, rpa = [hs_pos.pop(n, None) for n in fid_names]
        nasion, lpa, rpa = tuple(hs_pos[-3:])
        hs_pos = hs_pos[:-3, :]
    else:
        nasion = [1, 1, 1]
        
    # Check for reference points
    if nasion[1] != 0 or nasion[2] != 0:
        nasion, lpa, rpa = None, None, None

    ch_pos = {ch_names_[i]: pos[i] for i in range(len(ch_names_))}
    
    if nasion is None:
        return make_dig_montage(ch_pos=ch_pos, nasion=nasion, lpa=lpa, rpa=rpa,
                                hsp=hs_pos, coord_frame='head')
    else:
        return make_dig_montage(ch_pos=ch_pos, nasion=nasion, lpa=lpa, rpa=rpa,
                                hsp=hs_pos, coord_frame='unknown')
