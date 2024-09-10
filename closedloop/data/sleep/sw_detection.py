import os
import os.path as op
import numpy as np
import scipy as sp
import pandas as pd
import mne
import warnings
from ast import literal_eval
from closedloop.data.sleep.staging import crop_hypno


def envelope(data, n_excl=1, n_kept=3):
    ch_name = ['envelope']
    sfreq = None
    if isinstance(data, (mne.io.Raw, mne.io.BaseRaw)):
        sfreq = data.info['sfreq']
        data = data.get_data()       
    # data = np.sort(data, axis=0, kind='quicksort')
    data = np.sort(data, axis=0, kind='stable')
    envp = np.mean(data[n_excl:n_excl+n_kept, :], axis=0, keepdims=True)
    
    return envp, ch_name, sfreq


def noout_detrend(data, perc=99.95):
    # Detrend on single channel, taking out outliers
    if isinstance(data, (mne.io.Raw, mne.io.BaseRaw)):
        perc_val = np.percentile(abs(data.get_data()), perc, axis=-1)
        means = np.nanmean(
            np.where(abs(data.get_data()) < np.expand_dims(perc_val, -1), 
                     data.get_data(), np.nan), 
            axis=-1, keepdims=True)
        data._data = data.get_data() - means
    elif isinstance(data, (np.ndarray)):
        perc_val = np.percentile(abs(data), perc, axis=-1)
        means = np.nanmean(np.where(abs(data) < np.expand_dims(perc_val, -1), 
                                    data, np.nan), 
                           axis=-1, keepdims=True)
        data = data - means
    else:
        raise TypeError('data should be a mne Raw or numpy array type object')
    
    return data


def detrend(data, type='linear', bp=0):
    # data should be (nchans, ntimes)            
    if isinstance(data, (mne.io.Raw, mne.io.BaseRaw)):
        data._data = sp.signal.detrend(data.get_data(), axis=-1, type=type, 
                                       bp=bp)
    elif isinstance(data, (np.ndarray)):
        data = sp.signal.detrend(data, axis=-1, type=type, bp=bp)
    else:
        raise TypeError('data should be a mne Raw or numpy array type object')
    
    return data


def to_uV(data):
    # Remove artifacts and compute a masked array
    max_amp = np.percentile(abs(data), 99.95)
    msk_data = np.ma.masked_greater(abs(data), max_amp)
    _d = data.copy()
    _d[msk_data.mask] = np.nan
    # Good data should have an abs ampitude around 1e-5 V
    while not (1e-6 <= (np.nanmean(abs(_d))) <= 1e-4):
        warnings.warn('Rescaling data to fit the uV scale...')
        if (np.sum(np.abs(data)) / data.size) >= 1e-6:
            warnings.warn('... dividing by 10')
            data *= 1e-1
        elif (np.sum(np.abs(data)) / data.size) <= 1e-4:
            warnings.warn('... multiplying by 10')
            data *= 1e1
            
        max_amp = np.percentile(abs(data), 99.95)
        msk_data = np.ma.masked_greater(abs(data), max_amp)
        _d = data.copy()
        _d[msk_data.mask] = np.nan
            
    return data
        

def sw_filter(data, sfreq=None, copy=False, n_jobs='cuda'):
    
    if sfreq is None and isinstance(data, (np.ndarray)):
        raise ValueError('sfreq should be specified when data is a numpy' / 
                         'array type object')
    elif sfreq is None and isinstance(data, (mne.io.Raw, mne.io.BaseRaw)):
        sfreq = data.info['sfreq']
    
    wp = np.array([.5, 4.]) #/ (sfreq / 2) # pass bands
    ws = np.array([.1, 10.]) #/ (sfreq / 2) # stop bands
    Rp, Rs = 3, 10
    n, Wn = sp.signal.cheb2ord(wp, ws, Rp, Rs, analog=False, fs=sfreq)
    bbp, abp = sp.signal.cheby2(n, Rs, Wn, btype='bandpass', analog=False, 
                                output='ba', fs=sfreq)
    
    # The parameters that we used for the IIR filter can be summarized as
    # fillow (for MNE users)
    # iir_params = {
    #     'ftype': 'cheby2',
    #     'rp': 3.,
    #     'rs': 10.,
    #     'gpass': 3.,
    #     'gstop': 10.,
    #     'output': 'ba'
    # }
    
    if isinstance(data, (np.ndarray)):
        filt_data = sp.signal.filtfilt(bbp, abp, data, axis=-1)
        
    elif isinstance(data, (mne.io.Raw, mne.io.BaseRaw)):
        filt_data = data.copy()
        filt_data._data = sp.signal.filtfilt(bbp, abp, data.get_data(), 
                                             axis=-1)
    
    return filt_data


def swsd(ch_data, sfreq, hypno=None, stgs=[2, 3], half_wlen=(.125, 1.), 
         neg_amp=(40.e-6, 200.e-6), pos_amp=(10e-6, 150e-6)):
    
    ch_data = ch_data.squeeze()
    assert ch_data.ndim == 1, 'too many dimensions for single channel data'
    
    # Smooth data
    ch_data = np.convolve(ch_data, np.ones(5) / 5, mode='same')
    
    # Correct data to uV
    # ch_data = to_uV(ch_data)
    
    negzx = np.where(np.diff(np.sign(ch_data))==-2)[0] # Negative 0 crosses
    poszx = np.where(np.diff(np.sign(ch_data))==2)[0] # Positive 0 crosses
    
    if poszx[0] < negzx[0]:
        poszx = np.delete(poszx, 0)
        
    # negpeaks = sp.signal.find_peaks(-ch_data, height=(40e-5, 200e-5),
    #                                 width=(25, 200))[0]
    negpeaks = sp.signal.find_peaks(-ch_data, height=neg_amp)[0]
    # pospeaks = sp.signal.find_peaks(ch_data, height=(20e-5, 200e-5), 
    # width=(25, 200))[0]
    pospeaks = sp.signal.find_peaks(ch_data, height=pos_amp)[0]
    
    negpeaks = negpeaks[ch_data[negpeaks] < 0]
    pospeaks = pospeaks[ch_data[pospeaks] > 0]
    
    sw_info = {
        'negzx': [],        # first (negative) zero-crossing
        'poszx': [],        # second (positive) zero-crossing
        'wvend': [],        # endpoint of the slow wave
        'negpks': [],       # positions of all the negative peaks
        'maxnegpk': [],     # position of the maximal negative peak
        'negpkamp': [],     # amplitude of all detected negative peaks
        'maxnegpkamp': [],  # amplitude of the maximal negative peak
        'pospks': [],       # positions of all the positive peaks
        'maxpospk': [],     # position of the maximal positive peak
        'pospkamp': [],     # amplitude of all detected positive peaks
        'maxpospkamp': [],  # amplitude of the maximal positive peak
        'mxdnslp': [],      # value of the descending slope
        'mxupslp': []       # value of the ascending slope
    }
        
    for inzx, nzx in enumerate(negzx[:-1]):
        # nzx is the position of the first negative zero-crossing
        if hypno is not None:
            if hypno[nzx] in stgs:
                pass
            else:
                continue
        # w_end is the end of the wave (second negative zero-crossing)
        w_end = negzx[inzx+1]
        # pzx is the positive zero-crossing in between the negative two
        pzx = int(poszx[np.logical_and(poszx>nzx, poszx<w_end)])
        # take all negative and positie peaks in the middle
        npks = negpeaks[np.logical_and(negpeaks>nzx, negpeaks<w_end)]
        ppks = pospeaks[np.logical_and(pospeaks>nzx, pospeaks<w_end)]
        # make sure they are arrays
        npks = np.array(list(npks))
        ppks = np.array(list(ppks))
        # compute the half (negative) wavelength
        hwl = np.abs((pzx - nzx) / sfreq)
        check_hwl = half_wlen[0] <= hwl <= half_wlen[1]
        # Check if there are enough positive and negative peaks and if the 
        # half wavelength is in the correct range. 
        # If not move to the next pair of neg 0-cross
        if (len(npks)>0 and len(ppks)>0) and check_hwl:
            pass
        else:
            continue
        
        # Compute all negative peaks amplitude
        npks_amp = ch_data[npks]
        # Compute the maximal amplitude
        max_npk_amp = np.min(npks_amp)
        # Compute the position of the most negative peak according to amplitude
        max_npk = npks[npks_amp==max_npk_amp]
        
        # Compute all positive peaks amplitude
        ppks_amp = ch_data[ppks]
        # Compute the maximal amplitude
        max_ppk_amp = np.max(ppks_amp)
        # Compute the position of the most positive peak according to amplitude
        max_ppk = ppks[ppks_amp==max_ppk_amp]
        
        # Amplitude check (already implemented in find_peaks)
        # if not amp_thr[0] <= abs(max_npk_amp) <= amp_thr[1]:
        #     continue            
        
        #TODO check what property of the slope we actually want
        dwn_slope = np.abs(
            np.min(
                np.convolve(
                    np.diff(ch_data[nzx:pzx]), 
                    np.ones(5) / 5, mode='same'))) * sfreq
        up_slope = np.max(
            np.convolve(
                np.diff(ch_data[nzx:pzx]), 
                np.ones(5) / 5, mode='same')) * sfreq
        
        sw_info['negzx'].append(int(nzx))
        sw_info['poszx'].append(int(pzx))
        sw_info['wvend'].append(int(w_end))
        sw_info['negpks'].append(list(npks.astype(int)))
        sw_info['maxnegpk'].append(int(max_npk))
        sw_info['negpkamp'].append(list(npks_amp.astype('float32')))
        sw_info['maxnegpkamp'].append(float(max_npk_amp.astype('float32')))
        sw_info['pospks'].append(list(ppks.astype(int)))
        sw_info['maxpospk'].append(int(max_ppk))
        sw_info['pospkamp'].append(list(ppks_amp.astype('float32')))
        sw_info['maxpospkamp'].append(float(max_ppk_amp.astype('float32')))
        sw_info['mxdnslp'].append(float(dwn_slope))
        sw_info['mxupslp'].append(float(up_slope))
        
    import matplotlib.pyplot as plt
    plt.plot(ch_data.squeeze(), color='b')
    if hypno is not None:
        plt.plot(hypno*1e-5)
    plt.scatter(sw_info['maxnegpk'], ch_data.squeeze()[sw_info['maxnegpk']], color='r')
    plt.hlines(-neg_amp[0], 0, len(ch_data.squeeze()), color='r', ls='--')
    plt.hlines(-neg_amp[1], 0, len(ch_data.squeeze()), color='r', ls='--')
    plt.hlines(pos_amp[0], 0, len(ch_data.squeeze()), color='g', ls='--')
    plt.hlines(pos_amp[1], 0, len(ch_data.squeeze()), color='g', ls='--')
    # plt.show(block=False)
    
    return sw_info
    

def detect_sw(data, sfreq=None, hypno=None, stgs=[2, 3], ch_names=None, 
              half_wlen=(0.25, 1.), neg_amp=(40.e-6, 200.e-6), 
              pos_amp=(10e-6, 150e-6), n_jobs='cuda'):
    
    if isinstance(data, (mne.io.Raw, mne.io.BaseRaw)):
        data = data.pick(['eeg'])
    
    if isinstance(data, (np.ndarray)):
        assert data.ndim == 2, ('data must be a two-dimensional array, '/
                                'the last dimension should correspond to time')
    
    if sfreq is None and isinstance(data, (np.ndarray)):
        raise ValueError('sfreq should be specified when data is a numpy' / 
                         'array type object')
    elif sfreq is None and isinstance(data, (mne.io.Raw, mne.io.BaseRaw)):
        sfreq = data.info['sfreq']
        
    if ch_names is None and isinstance(data, (np.ndarray)):
        ch_names = [f'ch_{i+1}' for i in range(data.shape[0])]
    elif ch_names is None and isinstance(data, (mne.io.Raw, mne.io.BaseRaw)):
        ch_names = data.ch_names
    
    if isinstance(data, (np.ndarray)):
        assert len(ch_names) == data.shape[0], ('mismatch in between length '/ 
            'of ch_names and number of channels in data')
    elif isinstance(data, (mne.io.Raw, mne.io.BaseRaw)):
        assert len(ch_names) == data.get_data().shape[0], ('mismatch in between '/
            'length of ch_names and number of channels in data')
    
    # data = detrend(data, type='linear')
    data = detrend(data, type='constant')
    # data = noout_detrend(data)
    
    data = sw_filter(data, sfreq, copy=False, n_jobs=n_jobs)
    
    chans_sw = {}
    
    for i, c in enumerate(ch_names):
        if isinstance(data, (np.ndarray)):
            ch_data = data[i, :]
        elif isinstance(data, (mne.io.Raw, mne.io.BaseRaw)):
            ch_data = data.copy().pick_channels([c]).get_data()
            
        sw_info = swsd(ch_data, sfreq, hypno, stgs, half_wlen=half_wlen, 
                       neg_amp=neg_amp, pos_amp=pos_amp)
        
        chans_sw[c] = sw_info
        
    chans_sw = pd.DataFrame.from_dict(chans_sw)
            
    return chans_sw


def check_envelope_sw(raw, sw_fname):
    raw.load_data()
    raw.pick_types(eeg=True)
    raw.filter(.5, 4., n_jobs='cuda')
    
    sw = pd.read_csv(sw_fname, sep=',', header=0, index_col=0)
    
    envl = sw['envelope']
    tps = np.array(literal_eval(envl['maxnegpk']), dtype=int)
    zrs = np.zeros_like(tps)
    evc = np.full_like(tps, 512)
    
    events = np.stack((tps, zrs, evc), axis=-1)
    
    epochs = mne.Epochs(raw, events, event_id=None, tmin=-5, tmax=5., 
                        baseline=None, picks=None, preload=True, 
                        reject=None, flat=None, proj=True, decim=1, 
                        reject_tmin=None, reject_tmax=None, detrend=None, 
                        on_missing='raise', reject_by_annotation=True, 
                        metadata=None, event_repeated='error', verbose=None)
    epochs.crop(-.04, .04)
    good_sw, bad_sw = [], []
    upb = -30e-6
    lwb = -180e-6
    
    for i, e in enumerate(epochs):
        
        avgs = e.mean(-1).squeeze()
        
        if np.sum(np.logical_and(avgs > lwb, avgs < upb)) >= 2:
            if np.sum(avgs < lwb) >= 3:
                continue
            else:
                good_sw.append(i)
    
    return
            

def inspect_sw(raw, sws, sfreq=None, channels=['envelope'], align='maxnegpk', 
               twin=(-2., 2.), tstep=.5):
    import matplotlib.pyplot as plt
    
    if sfreq is None:
        if isinstance(raw, (mne.io.Raw, mne.io.BaseRaw)):
            sfreq = raw.info['sfreq']
        else:
            raise TypeError('sfreq should be specified unless'/
                            'raw is a mne object')
    
    raw.pick_types(eeg=True)
    envp, _, _ = envelope(raw)
    
    if channels is None:
        channels = sws.keys()
        
    for ch in channels:
        ch_sw = sws[ch]
        tps = np.array(literal_eval(ch_sw[align]), dtype=int)
        for tp in tps:
            negtp = np.arange(twin[0], 1e-3) * sfreq
            postp = np.arange(0., twin[1]+1e-3) * sfreq
            
            
def compute_stage_envelope(prep_fname, stg_fname):
    prep = mne.io.read_raw_fif(prep_fname, preload=True)
    
    eve, ev_dict = mne.events_from_annotations(prep)
    wake_id = [ev_dict[w] for w in ev_dict.keys() 
                if w=='Stimulus/s30']
    if len(wake_id) == 1:
        wake_tp = eve[np.where(eve[:, -1]==wake_id[0]), 0][0]
        wake_tp -= prep._first_samps
        if wake_tp[0] >= len(prep.times):
            tmax = prep.times[-1]
        else:
            tmax = prep.times[wake_tp[0]]
        prep.crop(tmin=0., tmax=tmax)
    
    prep = prep.pick_types(eeg=True)
    prep.filter(.5, 40., n_jobs='cuda')
    
    hypno = crop_hypno(stg_fname, prep.info['sfreq'], 
                        prep.first_samp, prep.last_samp)
    
    envp, ch_names, sfreq = envelope(prep, n_excl=1, n_kept=3)
    
    return envp, ch_names, sfreq, hypno
    
    
            
            
if __name__ == '__main__':
    
    prj_data = '/home/ruggero.basanisi/data/tweakdreams'

    data_dir = prj_data
    subjects = ['TD001']
    nights = ['N3']
    
    for sbj in subjects:
        for n in nights:
            
            prep_dir = op.join(prj_data, 'mne', '{0}', '{1}', 
                                'prep').format(sbj, n)
            eve_dir = op.join(prj_data, 'mne', '{0}', '{1}', 
                                'eve').format(sbj, n)
            
            stg_fname = op.join(eve_dir, 'hypnogram.npy')
            
            aw = [a for a in os.listdir(prep_dir) if a.startswith('aw_')]
            aw.sort()
            # aw = ['aw_2']
            
            for _aw in aw:
                
                prep_fname = op.join(prep_dir, _aw, f'{sbj}_{n}_prep-raw.fif')
                
                raw = mne.io.read_raw_fif(prep_fname, preload=True)
                # Applying reference online
                # raw.set_eeg_reference(ref_channels=['L4H', 'R4H'])
                
                eve, ev_dict = mne.events_from_annotations(raw)
                wake_id = [ev_dict[w] for w in ev_dict.keys() 
                           if w=='Stimulus/s30']
                if len(wake_id) == 1:
                    wake_tp = eve[np.where(eve[:, -1]==wake_id[0]), 0][0]
                    wake_tp -= raw._first_samps
                    if wake_tp[0] >= len(raw.times):
                        tmax = raw.times[-1]
                    else:
                        tmax = raw.times[wake_tp[0]]
                    raw.crop(tmin=0., tmax=tmax)
                
                raw = raw.pick_types(eeg=True)
                # raw.resample(100., n_jobs=8)
                raw.filter(.5, 40., n_jobs='cuda')
                
                hypno = crop_hypno(stg_fname, raw.info['sfreq'], 
                                   raw.first_samp, raw.last_samp)
                
                raw, ch_names, sfreq = envelope(raw, n_excl=1, n_kept=3)
                
                # FIrst attempt, extracting less SW
                # df = detect_sw(raw, sfreq=sfreq, hypno=hypno, stgs=[2, 3], 
                #                ch_names=ch_names, neg_amp=(10.e-6, 100.e-6), 
                #                pos_amp=(None, 50.e-6), #(20.e-6, 150e-6), 
                #                n_jobs=16)
                
                # Trying to extend sw detection range after coding sw_correct, 
                # if it doesn't work return to previous lines
                df = detect_sw(raw, sfreq=sfreq, hypno=hypno, stgs=[2, 3], 
                               ch_names=ch_names, half_wlen=(0.12, 1.05),
                               neg_amp=(5.e-6, 200.e-6), 
                               pos_amp=(0, 100.e-6),
                               n_jobs=16)
                
                eve_dir = op.join(prj_data, 'mne', '{0}', '{1}', 
                                  'eve', _aw).format(sbj, n)
                os.makedirs(eve_dir, exist_ok=True)
                csv_fname = op.join(eve_dir, 'envelope_sw.csv')
                df.to_csv(csv_fname)
                print('Saved to:', csv_fname)
