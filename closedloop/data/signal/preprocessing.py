import os
import os.path as op
import numpy as np
import mne
from mne.preprocessing import ICA
import pyprep
import scipy as sp
import json
import shutil

from closedloop.data.utils.read_mne import read_raw, resample_raw


def do_preprocessing(raw_dir, resample=None, components=.99, 
                     bad_ch_exist=False, restrict=False, save=False):
    
    # Retrieving raw data specs
    raw, eve, fnames = read_raw(raw_dir, return_fnames=True)
    # np.sum(np.abs(data[5:10, :]))/data[5:10, :].size
    
    if len(raw.times) <= raw.info['sfreq'] * 60:
        print('Raw file containing less than one minute ',
              'of signal, skipping...\n')
        shutil.rmtree(raw_dir)
        print(raw_dir, 'removed.')
        return
    
    # Creating prep and eve directories
    prep_dir = raw_dir.replace('raw', 'prep')
    eve_dir = raw_dir.replace('raw', 'eve')
    os.makedirs(prep_dir, exist_ok=True)
    os.makedirs(eve_dir, exist_ok=True)
    
    # Start preprocessing
    raw.load_data()
    
    # Notch filter
    raw = raw.notch_filter(freqs=np.arange(50, 201, 50), n_jobs='cuda')
    
    raw = raw.filter(l_freq=.2, h_freq=None, picks='eeg', n_jobs='cuda')
    raw = raw.filter(l_freq=.5, h_freq=40., picks='ecg', n_jobs='cuda')
    raw = raw.filter(l_freq=10., h_freq=None, picks='emg', n_jobs='cuda')
    raw = raw.filter(l_freq=.1, h_freq=30., picks='eog', n_jobs='cuda')
    
    bad_fname = op.join(prep_dir, 'bad_channels.json')
    if bad_ch_exist:
        print('Loading bad channels')
        with open(bad_fname, 'r') as f:
            bads = json.load(f)
            
    else:
        print('\nComputing bad channels')
        nc = pyprep.NoisyChannels(raw=raw.copy().pick_types(eeg=True).
                                  resample(100.), 
                                  do_detrend=True, random_state=23, 
                                  matlab_strict=False)
        nc.find_all_bads()
        nc.find_bad_by_nan_flat()
        bads = nc.get_bads(as_dict=True)
        
        with open(bad_fname, 'w') as f:
            json.dump(bads, f)
    
    raw.info['bads'] = bads['bad_all']
    print('Channels marked as bad:', raw.info['bads'])
    
    if components == 'all': # Perform a PCA
        components = (len(raw.get_channel_types(picks='eeg')) 
                      - len(raw.info['bads']))
    
    filt_raw = raw.copy().filter(1., 45., n_jobs='cuda')
    
    # Uncomment to perform ocular artifact rejection via SSP
    # # Create ocular movements epochs
    # heog = mne.preprocessing.create_eog_epochs(filt_raw, ch_name='HEOG', 
    #                                            reject={'eeg': 40e-5})
    # veog = mne.preprocessing.create_eog_epochs(filt_raw, ch_name='VEOG', 
    #                                            reject={'eeg': 40e-5})
    # # Compute evoked 
    # heog = heog.average().apply_baseline((None, None))
    # veog = veog.average().apply_baseline((None, None))
    # # Compute projections (one per direction)
    # h_proj = mne.compute_proj_evoked(heog, n_grad=0, n_mag=0, n_eeg=1)
    # v_proj = mne.compute_proj_evoked(veog, n_grad=0, n_mag=0, n_eeg=1)
    # # Add and apply projections
    # filt_raw.add_proj(h_proj + v_proj)
    # filt_raw.apply_proj()
    
    # Uncomment to perform ocular artifact rejection via regression
    # eog_reg = mne.preprocessing.EOGRegression()
    # eog_reg.fit(filt_raw)
    # filt_raw = eog_reg.apply(filt_raw)
    
    print('\nComputing ICA')
    # ica = ICA(n_components=components, noise_cov=None, random_state=23, 
    #           method='picard', fit_params=dict(ortho=True, extended=True), 
    #           max_iter=1500, allow_ref_meg=False)
    
    # No peak-to-peak rejection, implemented when computing epochs
    reject_ptp = None
    
    # Exclude waking period (if working on awakenings)
    if restrict:
        ev, ev_dict = mne.events_from_annotations(raw)
        # TODO: check if raw.first_sample and the first event (s20) have the same timestamp
        start = int(ev[np.where(ev[:, -1] == ev_dict['Stimulus/s20'])[0], 0][0])
        if 'Stimulus/s30' in ev_dict.keys():
            stop = int(ev[np.where(ev[:, -1] == ev_dict['Stimulus/s30'])[0], 0][-1])
            stop -= start
        else:
            stop = None
        start -= start
    else:
        start, stop = None, None
    
    # Use ICA to find and remove cardiac and muscolar components
    try:
        ica = ICA(n_components=components, noise_cov=None, random_state=23, 
                  method='fastica', max_iter=1500, allow_ref_meg=False)
        # Alternative
        # ica = ICA(n_components=components, noise_cov=None, random_state=23, 
        #           method='picard', fit_params=dict(ortho=True, extended=True), 
        #           max_iter=1500, allow_ref_meg=False)
        ica.fit(inst=filt_raw, decim=1, start=start, stop=stop, 
                reject=reject_ptp)
    except Exception: # To overcome the 'single component problem' 
        print(f'Attempt to compute ICA with n_components={components} failed,', 
              'attempting with number of compontents fixed to 30')
        ica = ICA(n_components=30, noise_cov=None, random_state=23, 
                  method='fastica', max_iter=1500, allow_ref_meg=False)
        ica.fit(inst=filt_raw, decim=1, start=start, stop=stop, 
                reject=reject_ptp)

    bad_ecg, ecg_scores = ica.find_bads_ecg(filt_raw, ch_name='ECG', 
                                            start=start, stop=stop)
    bad_emg, emg_scores = ica.find_bads_muscle(filt_raw, sphere='auto',
                                               threshold=0.1,
                                               start=start, stop=stop)
    
    bads_ica = list(np.unique(bad_ecg + bad_emg))
    
    # Uncomment to perform ocular artifact rejection via ICA
    # bad_eog, eog_scores = ica.find_bads_eog(filt_raw, 
    #                                         ch_name=['VEOG', 'HEOG'],
    #                                         start=start, stop=stop)    
    # In case you want to remove ocular components
    # bads_ica = list(np.unique(bad_ecg + bad_eog + bad_emg))
    
    ica.exclude = bads_ica
    
    # Create a copy and filter
    prep_raw = raw.copy()
    prep_raw = prep_raw.filter(.2, 45., n_jobs='cuda')
    
    # Apply SSP projection (if computed)
    # prep_raw.add_proj(h_proj + v_proj)
    # prep_raw.add_proj(v_proj)
    # prep_raw.apply_proj()
    
    # Re-compute and apply EOG regression (if computed)
    # eog_reg = mne.preprocessing.EOGRegression()
    # eog_reg.fit(prep_raw)
    # prep_raw = eog_reg.apply(prep_raw)
    
    # Apply ICA
    prep_raw = ica.apply(prep_raw, exclude=bads_ica)
    
    # Apply projections
    # prep_raw.add_proj(v_proj)
    # prep_raw.apply_proj()
    
    # Bad channels interpolation
    prep_raw = prep_raw.interpolate_bads(reset_bads=True, 
                                         mode='accurate', 
                                         method='spline')
    
    # Reference channels on the average
    # prep_raw = prep_raw.set_eeg_reference(ref_channels='average', 
    #                                       projection=False, ch_type='eeg', 
    #                                       forward=None, joint=False, 
    #                                       verbose=None)à
    
    # Reference channels on mastoids electrodes
    prep_raw.set_eeg_reference(ref_channels=['L4H', 'R4H'])
    
    # Repeat RANSAC, bad channels detection and interpolation
    nc = pyprep.NoisyChannels(raw=prep_raw.copy().pick_types(eeg=True).
                                  resample(100.), 
                                  do_detrend=True, random_state=23,
                                  matlab_strict=False)
    nc.find_all_bads()
    bads = nc.get_bads(as_dict=True)
    prep_raw.info['bads'] = bads['bad_all']
    print('Channels marked as bad:', prep_raw.info['bads'])
    prep_raw = prep_raw.interpolate_bads(reset_bads=True, 
                                         mode='accurate', 
                                         method='spline')
    
    # Detrending
    # raw._data = mne.baseline.rescale(raw.get_data(), raw.times, 
    #                                  baseline=(None, None), mode='mean')
    # prep_raw._data = sp.signal.detrend(prep_raw.get_data(), axis=-1, 
    #                                    type='linear', bp=0)
    
    if resample is not None:
        prep_raw, eve = resample_raw(prep_raw, ev, freq=resample)
    
    if save:
        # Saving the raw fif file
        prep_fname = fnames[0].split('/')[-1].replace('-raw', '_prep-raw')
        prep_raw.save(op.join(prep_dir, prep_fname),
                      split_naming='neuromag', overwrite=True)
        
        # Saving events
        eve_fname = fnames[1].split('/')[-1].replace('-eve', '_prep-eve')
        mne.write_events(op.join(prep_dir, eve_fname),
                         eve, overwrite=True)
    
    return prep_raw, eve

if __name__ == "__main__":
    import time
    
    os.environ['NUMEXPR_MAX_THREADS'] = '32'

    # prj_data = '/home/ruggero.basanisi/data/tweakdreams'
    prj_data = '/media/jerry/ruggero/data/tweakdreams'

    data_dir = prj_data
    subjects = ['TD001']
    nights = ['N4']
    
    for sbj in subjects:
        for n in nights:
            raw_dir = op.join(prj_data, 'mne', '{0}', '{1}', 
                                'raw').format(sbj, n)
            
            aw = [a for a in os.listdir(raw_dir) if a.startswith('aw_')]
            aw.sort()
            aw = ['aw_5']
            
            for _aw in aw:
                _raw_dir = op.join(raw_dir, _aw)
                

                start = time.time()
                
                prep_filraw = do_preprocessing(_raw_dir, resample=100., 
                                            components=.99, 
                                            bad_ch_exist=False,
                                            restrict=True,
                                            save=True)
                
                end = time.time()
                print('Done in ', end-start, ' seconds.')
