import os.path as op
import numpy as np
import mne
from mne.preprocessing import ICA
import pyprep

from read_mne import read_raw, resample_raw


def do_preprocessing(raw_dir, resample=None, components=.99, save=False):
    
    raw, eve, fnames = read_raw(raw_dir, return_fnames=True)
    # np.sum(np.abs(data[5:10, :]))/data[5:10, :].size
    
    raw.load_data()
    
    raw = raw.filter(l_freq=.01, h_freq=None, picks='eeg', n_jobs='cuda')
    raw = raw.filter(l_freq=2.5, h_freq=20., picks='ecg', n_jobs='cuda')
    raw = raw.filter(l_freq=.5, h_freq=None, picks='eog', n_jobs='cuda')
    
    # raw.apply_function(lambda x: x * 1e-3)
    
    # if resample is not None:
    #     raw, eve = resample_raw(raw, eve, freq=resample)
    print('\nComputing bad channels')
    nc = pyprep.NoisyChannels(raw=raw.copy().pick_types(eeg=True).
                              resample(100.), 
                              do_detrend=True, random_state=23, 
                              matlab_strict=False)
    nc.find_all_bads()
    nc.find_bad_by_nan_flat()
    bads = nc.get_bads(as_dict=True)
    
    # bads['bad_all'].remove('Z12Z')
    raw.info['bads'] = bads['bad_all']
    print('Channels marked as bad:', raw.info['bads'])
    
    if components == 'all': # Perform a PCA
        components = (len(raw.get_channel_types(picks='eeg')) 
                      - len(raw.info['bads']))
    
    # raw = raw.set_eeg_reference(ref_channels='average', projection=False, 
    #                             ch_type='eeg', forward=None, joint=False, 
    #                             verbose=None)
    
    filt_raw = raw.copy().filter(1., 45., n_jobs='cuda')
    
    print('\nComputing ICA')
    ica = ICA(n_components=components, noise_cov=None, random_state=23, 
              method='picard', fit_params=dict(ortho=True, extended=True), 
              max_iter=1000, allow_ref_meg=False, 
              )
    
    # reject_ptp = dict(eeg=40e-6, eog=250e-6)
    reject_ptp = None
    
    ica.fit(inst=filt_raw, decim=5, reject=reject_ptp)
    
    bad_ecg, ecg_scores = ica.find_bads_ecg(filt_raw, ch_name='ECG', 
                                            method='correlation',
                                            measure='zscore', threshold=2.5)
    bad_eog, eog_scores = ica.find_bads_eog(filt_raw, ch_name=['VEOG', 'HEOG'])
    bad_emg, emg_scores = ica.find_bads_muscle(filt_raw, sphere='auto')
    
    bads_ica = list(np.unique(bad_ecg + bad_eog + bad_emg))
    
    ica.exclude = bads_ica
    
    prep_raw = raw.copy()
    
    prep_raw = ica.apply(prep_raw, exclude=bads_ica)
    prep_raw = prep_raw.interpolate_bads(reset_bads=True, 
                                         mode='accurate', 
                                         method='spline')
    
    prep_raw = prep_raw.set_eeg_reference(ref_channels='average', 
                                          projection=False, ch_type='eeg', 
                                          forward=None, joint=False, 
                                          verbose=None)
    
    raw._data = mne.baseline.rescale(raw.get_data(), raw.times, 
                                     baseline=(None, None), mode='mean')
    
    if resample is not None:
        prep_raw, eve = resample_raw(prep_raw, eve, freq=resample)
    
    if save:
        # Saving the raw fif file
        prep_raw.save(fnames[0].replace('-raw', '_prep-raw'),
                      split_naming='neuromag', overwrite=True)
        # Saving events
        mne.write_events(fnames[1].replace('-eve', '_prep-eve'),
                         eve, overwrite=True)
    
    return prep_raw, eve

if __name__ == "__main__":
    # from utils.globals import prj_data
    import time

    prj_data = ''

    data_dir = prj_data
    subjects = ['TD001']
    nights = ['N1']
    # awakenings = ['aw_4','aw_5']
    # awakenings = ['aw_5'] # TODO: change with a dir finder

    for sbj in subjects:
        for n in nights:
            raw_dir = op.join(prj_data, 'mne', '{0}', '{1}', 
                                'raw').format(sbj, n)
            
            # raw, eve = read_raw(raw_dir)
            start = time.time()
            # raw, eve = resample_raw(raw, eve)
            prep_raw = do_preprocessing(raw_dir, resample=100., 
                                        components='all', save=True)
            end = time.time()
            print('Done in ', end-start, ' seconds.')
