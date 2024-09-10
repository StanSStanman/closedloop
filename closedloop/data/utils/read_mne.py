import mne
import os
import os.path as op
from closedloop.data.utils.utils import fname_finder


def read_raw(raw_dir, extension='', preload=False, return_fnames=False):
    raw_file = fname_finder(op.join(raw_dir,
                                    '*{0}-raw.fif'.format(extension)))[0]
    eve_file = fname_finder(op.join(raw_dir,
                                    '*{0}-eve.fif'.format(extension)))[0]

    raw = mne.io.read_raw_fif(raw_file, allow_maxshield=False, preload=preload)
    try:
        eve = mne.read_events(eve_file)
    except Exception:
        eve = None

    if return_fnames:
        return raw, eve, (raw_file, eve_file)
    else:
        return raw, eve
    
    
def read_eve(eve_dir, extension='', return_fnames=False):
    eve_file = fname_finder(op.join(eve_dir,
                                    '*{0}-eve.fif'.format(extension)))[0]
    eve = mne.read_events(eve_file)
    
    if return_fnames:
        return eve, eve_file
    else:
        return eve
    

def resample_raw(raw, eve, freq=100.):
    
    print('Loading data in memory...')
    raw.load_data(verbose=False)
    
    print('Resampling data from {0}Hz to {1}Hz...'.format(raw.info['sfreq'], 
                                                          freq))
    raw, eve = raw.resample(sfreq=freq, npad='auto', window='boxcar', 
                            pad='reflect_limited', n_jobs='cuda', 
                            events=eve, verbose=False)
    return raw, eve


if __name__ == "__main__":
    import time
    
    prj_data = ''

    data_dir = prj_data
    subjects = ['TD001']
    nights = ['N1']

    for sbj in subjects:
        for n in nights:
            raw_dir = op.join(prj_data, 'mne', '{0}', '{1}', 
                                'raw').format(sbj, n)
            
            raw, eve = read_raw(raw_dir)
            start = time.time()
            raw, eve = resample_raw(raw, eve)
            end = time.time()
            print('Done in ', end-start, ' seconds.')
