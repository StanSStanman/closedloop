
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
