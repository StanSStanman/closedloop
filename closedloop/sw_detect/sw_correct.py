import os
import os.path as op
import mne 
import numpy as np
import pandas as pd
import pickle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

from closedloop.sw_detect.sw_detect import detect_sw

def realign_sw_epo(epo_fname, eve_fname, t_dist=.05):
    
    try:
        epochs = mne.read_epochs(epo_fname)
    except Exception:
        print('No SWs epochs found for this awakening... skipping. \n')
        return None
    try:
        events = mne.read_events(eve_fname)
    except Exception:
        print('No events found for this awakening... skipping. \n')
        return None
    
    epochs = epochs.pick_types(eeg=True)
    # epochs.set_eeg_reference(ref_channels=['L4H', 'R4H'])
    
    radius = 0.035
    coords = np.array([ch['loc'][:3] for ch in epochs.info['chs']])
    
    t_zero = int(np.where(epochs.times == 0.)[0][0])
    samp_dist = int(epochs.info['sfreq'] * t_dist)
    samp_range = np.arange(t_zero - samp_dist, t_zero + samp_dist + 1)
    
    good_eve, bad_ev_idx,  = [], []
    maxneg_chs, sw_types = [], []
    # df = {}
    for i, e in enumerate(epochs):
        samp_data = e[:, samp_range]
        maxneg_ch = np.where(samp_data == samp_data.min())[0][0]
        
        ch_data = np.expand_dims(e[maxneg_ch, :], 0)
        
        distances = np.linalg.norm(coords - coords[maxneg_ch], axis=1)
        selected_indices = np.where(distances <= radius)[0]
        selected_indices = selected_indices[selected_indices != maxneg_ch]
        selected_data = e[selected_indices, :]
        _ch_data = selected_data.mean(0, keepdims=True)
        # selected_channels = [epochs.ch_names[idx] for idx in selected_indices]
        # epo_subset = epochs.copy().pick_channels(selected_channels)
        
        # plt.plot(e.T, color='grey', alpha=.5, lw=.2)
        
        # Detect slow waves in the maximun negative channel around 
        # the previously detected sw
        # sw_mn = detect_sw(ch_data, sfreq=epochs.info['sfreq'], hypno=None, 
        #                   ch_names=['envelope'], half_wlen=(0.125, 1.), 
        #                   neg_amp=(25.e-6, 250.e-6), pos_amp=(10e-6, 80e-6), 
        #                   n_jobs='cuda')
        # Trying to catch only SWs with an higher amplitude
        sw_mn = detect_sw(ch_data, sfreq=epochs.info['sfreq'], hypno=None, 
                          ch_names=['envelope'], half_wlen=(0.125, 1.), 
                          neg_amp=(40.e-6, 250.e-6), pos_amp=(20e-6, 150e-6), 
                          n_jobs='cuda')
        
        if any(tp in samp_range for tp in sw_mn['envelope']['maxnegpk']):
            # Detect slow waves on the average of the channels surrounding 
            # the maximal negative channel
            # sw_sn = detect_sw(_ch_data, sfreq=epochs.info['sfreq'], hypno=None, 
            #                   ch_names=['envelope'], half_wlen=(0.125, 1.), 
            #                   neg_amp=(20.e-6, 250.e-6), pos_amp=(5e-6, 80e-6),
            #                   n_jobs='cuda')
            # Trying to catch SW type I onlyTrying to catch only SWs with an higher amplitude
            sw_sn = detect_sw(_ch_data, sfreq=epochs.info['sfreq'], hypno=None, 
                              ch_names=['envelope'], half_wlen=(0.125, 1.), 
                              neg_amp=(30.e-6, 250.e-6), 
                              pos_amp=(15e-6, 150e-6),
                              n_jobs='cuda')
            
            
            if any(tp in samp_range for tp in sw_sn['envelope']['maxnegpk']):
                mnpk_tp = sw_mn['envelope']['maxnegpk']
                new_tp = [tp for tp in mnpk_tp if tp in samp_range][0]
                tp_diff = int(new_tp - t_zero)
                new_eve = epochs.events[i, :].copy()
                new_eve[0] += tp_diff
                good_eve.append(new_eve)
                
                maxneg_chs.append(epochs.ch_names[maxneg_ch])
                amp_pos = np.where(np.array(mnpk_tp) == new_tp)[0][0]
                amp_val = sw_mn['envelope']['negpkamp'][amp_pos][0]
                if amp_val <= -7.5e-5:
                    sw_types.append(1)
                else:
                    sw_types.append(2)
                
                plt.title('good sw')
                # plt.plot(ch_data.squeeze(), color='orange')
                # plt.plot(_ch_data.squeeze(), color='yellow')
                
                # TODO finish to implement the function to save correct SWs in
                # a .csv file ->
                
                # Rebuild the dataframe to save as .csv
                # npk_idx = np.where(np.array(sw_mn['envelope']['maxnegpk']) 
                #                    == new_tp)[0][0]
                # if len(df) == 0:
                #     df['envelope'] = {}
                #     for k in sw_mn['envelope'].keys():
                #         df['envelope'][k] = [sw_mn['envelope'][k][npk_idx]]
                # else:
                #     for k in sw_mn['envelope'].keys():
                #         df['envelope'][k].append(sw_mn['envelope'][k][npk_idx])
                
            
            else:
                bad_ev_idx.append(i)
                
        else:
            bad_ev_idx.append(i)
            
        
        plt.plot(ch_data.squeeze(), color='orange')
        plt.plot(_ch_data.squeeze(), color='yellow')
        a = 0
        
    good_eve = np.vstack(good_eve)
    mne.write_events(eve_fname.replace('-eve.fif', '_clean-eve.fif'), 
                     good_eve, overwrite=True)
    print('\nDiscarded {0} out of {1} epochs'.format(len(bad_ev_idx),
                                                        len(epochs)))
    print('Kept {0} epochs... saving \n'.format(good_eve.shape[0]), '\n')
    
    # TODO
    # df = pd.DataFrame.from_dict(df)
    # df.to_csv(eve_fname.replace('-eve.fif', '_clean.csv'))
    
    return good_eve, bad_ev_idx, maxneg_chs, sw_types
    
    
if __name__ == '__main__':
    
    prj_data = '/home/ruggero.basanisi/data/tweakdreams'

    data_dir = prj_data
    subjects = ['TD001']
    nights = ['N1']
    
    for sbj in subjects:
        for n in nights:
            
            # prep_dir = op.join(prj_data, 'mne', '{0}', '{1}', 
            #                  'prep').format(sbj, n)
            epo_dir = op.join(prj_data, 'mne', '{0}', '{1}', 
                             'epo').format(sbj, n)
            eve_dir = op.join(prj_data, 'mne', '{0}', '{1}', 
                             'eve').format(sbj, n)
            

            aw = [a for a in os.listdir(eve_dir) if a.startswith('aw_')]
            aw.sort()
            aw = ['aw_5']
            
            for _aw in aw:
                
                # _prep_dir = op.join(prep_dir, _aw)
                _epo_dir = op.join(epo_dir, _aw)
                _eve_dir = op.join(eve_dir, _aw)
                
                os.makedirs(_epo_dir, exist_ok=True)
                
                # prep_fname = op.join(_prep_dir, 'TD001-N1_prep-raw.fif')
                epo_fname = op.join(_epo_dir, 'envelope_sw-epo.fif')
                eve_fname = op.join(_eve_dir, 'envelope_sw-eve.fif')
                chs_fname = op.join(_eve_dir, 'maxneg_channels.txt')
                swt_fanme = op.join(_eve_dir, 'sw_types.txt')
                
                rsw = realign_sw_epo(epo_fname, eve_fname)
                
                if rsw is not None:
                    gt, bt, mnch, swt = rsw
                
                    with open(chs_fname, 'wb') as f:
                        pickle.dump(mnch, f)
                        
                    with open(swt_fanme, 'wb') as f:
                        pickle.dump(swt, f)
