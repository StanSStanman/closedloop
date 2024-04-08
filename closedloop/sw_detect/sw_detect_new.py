import numpy as np
import scipy as sp
import mne
import warnings


def envelope(data, n_excl=1, n_kept=3):
    ch_name = ['envelope']
    sfreq = None
    if isinstance(data, (mne.io.Raw, mne.io.BaseRaw)):
        sfreq = data.info['sfreq']
        data = data.get_data()       
    data = np.sort(data, axis=0, kind='quicksort')
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


def detrend(data):
    # data should be (nchans, ntimes)            
    if isinstance(data, (mne.io.Raw, mne.io.BaseRaw)):
        data._data = data.get_data() - data.get_data().mean(-1, keepdims=True)
    elif isinstance(data, (np.ndarray)):
        data = data - data.mean(-1, keepdims=True)
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
    
    wp = np.array([.5, 4.]) / (sfreq / 2) # pass bands
    ws = np.array([.1, 10.]) / (sfreq / 2) # stop bands
    iir_params = {
        'ftype': 'cheby2',
        'rp': 3.,
        'rs': 10.,
        'gpass': 3.,
        'gstop': 10.,
        'output': 'sos'
    }
    iir_filt = mne.filter.construct_iir_filter(iir_params=iir_params, 
                                               f_pass=wp, f_stop=ws, 
                                               sfreq=sfreq, btype='bandpass',
                                               return_copy=copy)
    
    if isinstance(data, (np.ndarray)):
        filt_data = mne.filter.filter_data(data, sfreq, l_freq=.5, h_freq=4., 
                                           picks=None, filter_length='auto',
                                           n_jobs=n_jobs, method='iir',
                                           iir_params=iir_filt, copy=copy, 
                                           phase='zero', pad='reflect_limited')
    elif isinstance(data, (mne.io.Raw, mne.io.BaseRaw)):
        # filt_data = data.filter(l_freq=.5, h_freq=4., picks=None, 
        #                         filter_length='auto', n_jobs=n_jobs, 
        #                         method='iir', iir_params=iir_filt, 
        #                         phase='zero', pad='reflect_limited')
        filt_data = data.filter(l_freq=.5, h_freq=4., picks=None, 
                                filter_length='auto', n_jobs=n_jobs, 
                                method='fir', phase='zero', 
                                pad='reflect_limited')
    
    return filt_data
    
    
def zerocross(ch_data):
    
    ch_data = ch_data.squeeze()
    assert ch_data.ndim == 1, 'too many dimensions for single channel data'
    
    pos_idx = np.zeros_like(ch_data)
    pos_idx[ch_data>0] = 1
    difference = np.diff(pos_idx, n=1)
    poscross = np.where(difference==1)[0]
    negcross = np.where(difference==-1)[0]
    # TODO: Check these steps, they can mess easily up with positions
    smoothed = np.convolve(ch_data, np.ones(5) / 5, mode='same')
    pos_idx = np.zeros_like(smoothed)
    pos_idx[smoothed>0] = 1
    difference = np.diff(pos_idx, n=1)
    peaks = np.where(difference==-1)[0] + 1
    troughs = np.where(difference==1)[0] + 1
    # peaks = peaks[ch_data[peaks]<0]
    peaks = peaks[ch_data[peaks]>=0]
    # troughs = troughs[ch_data[troughs>0]]
    troughs = troughs[ch_data[troughs]<=0]
    # Makes negcross and poscross same size to start
    # TODO: check these steps too, maybe can be redefined if the only purpose 
    # is to define a starting point of the array
    if negcross[0] < poscross[0]:
        start = 1
    else:
        start = 2
        
    if start == 2:
        poscross = poscross[1:] 
        
    return negcross, poscross, troughs, peaks, start


def swsd(ch_data, sfreq, half_wlen=(.125, 1.), neg_amp=(40.e-5, 200.e-5),
         pos_amp=(10e-5, 150e-5)):
    
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
    
    return sw_info
    

def detect_sw(data, sfreq=None, ch_names=None, half_wlen=(0.25, 1.), 
              neg_amp=(40.e-5, 200.e-5), pos_amp=(10e-5, 150e-5), 
              n_jobs='cuda'):
    
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
    
    # data = detrend(data)
    data = noout_detrend(data)
    
    data = sw_filter(data, sfreq, copy=False, n_jobs=n_jobs)
    
    chans_sw = {}
    
    for i, c in enumerate(ch_names):
        if isinstance(data, (np.ndarray)):
            ch_data = data[i, :]
        elif isinstance(data, (mne.io.Raw, mne.io.BaseRaw)):
            ch_data = data.copy().pick_channels([c]).get_data()
            
        sw_info = swsd(ch_data, sfreq, half_wlen=half_wlen, neg_amp=neg_amp)
        
        chans_sw[c] = sw_info
            
    return


if __name__ == '__main__':
    raw_fname = 'test_data/n1_raw.fif'
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    raw.resample(100., n_jobs=8)
    raw._data = raw.get_data() * 10
    
    print(np.sum(np.abs(raw.get_data()))/raw.get_data().size)
    
    chans = ['F4-C4','C4-A1']
    raw.pick_channels(chans)
    
    detect_sw(raw, n_jobs=8)
