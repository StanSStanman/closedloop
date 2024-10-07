import os
import os.path as op
import pickle
import numpy as np
import mne
import xarray as xr


def compute_topographies(subjects, nights, swori_fname, areas):
    """_summary_

    Args:
        sbj_dir (_type_): _description_
        swori_fname (_type_): _description_
        areas (dict): _description_
    """
    
    # Dictionary containing areas name and the rois belonging to that area.
    # If you want to take all the rois separately every roi should be an area
    # containing just itself.
    areas_avgs = {k: {'raw_sum': None, 
                      'epo_sum': None, 
                      'num': 0} 
                  for k in areas.keys()}
    
    # nights = ['N1', 'N2', 'N3', 'N4']
    
    for sbj in subjects:
        
        sbj_dir = op.join(data_dir, 'mne', sbj)
        
        for ngh in nights:
    
            ltc_dir = op.join(sbj_dir, ngh, 'ltc')
            
            aw = [a for a in os.listdir(ltc_dir) if a.startswith('aw_')]
            aw.sort()
            
            for _aw in aw:
                swofn = op.join(ltc_dir, _aw, swori_fname)
                # Skip awakening if file do not exist
                if not op.exists(swofn):
                    continue
                
                with open(swofn, 'rb') as f:
                    origins = np.array(pickle.load(f))
                
                # Load epochs and extract events
                epo_fname = op.join(sbj_dir, ngh, 'epo', _aw, 
                                    'envelope_sw_clean-epo.fif')
                epochs = mne.read_epochs(epo_fname)
                epochs = epochs.pick('eeg')
                tmin, tmax = epochs.tmin, epochs.tmax
                events = epochs.events
                # Load raw and extract epochs
                raw_fname = op.join(sbj_dir, ngh, 'raw', _aw, 
                                    f'{sbj}_{ngh}-raw.fif')
                raw = mne.io.read_raw_fif(raw_fname, preload=False)
                raw = raw.pick('eeg')
                events[:, 0] = np.round(events[:, 0] * 
                                        (raw.info['sfreq'] / 
                                            epochs.info['sfreq'])).astype(int)
                # raw.resample(epochs.info['sfreq'])
                raw_epo = mne.Epochs(raw, events=events, tmin=tmin, tmax=tmax, 
                                    baseline=None)
                
                for ak in areas.keys():
                    for r in areas[ak]:
                        ori_epo = np.where(origins == r)[0]
                        
                        if len(ori_epo) > 0:
                            # Sum raw over epochs
                            sum_raw = raw_epo[ori_epo].get_data(copy=True).sum(0)
                            # Sum epochs orver epochs
                            sum_epo = epochs[ori_epo].get_data(copy=True).sum(0)

                            # Update raw sum in dict
                            if areas_avgs[ak]['raw_sum'] is None:
                                areas_avgs[ak]['raw_sum'] = sum_raw
                            else:
                                areas_avgs[ak]['raw_sum'] += sum_raw
                            # Update epo sum in dict
                            if areas_avgs[ak]['epo_sum'] is None:
                                areas_avgs[ak]['epo_sum'] = sum_epo
                            else:
                                areas_avgs[ak]['epo_sum'] += sum_epo
                            # Update number of summed element
                            areas_avgs[ak]['num'] += len(ori_epo)
    
    # Delete empty ROIs
    del_list = []
    for k in areas_avgs.keys():
        if areas_avgs[k]['num'] == 0:
            del_list.append(k)
    for k in del_list:
        del areas_avgs[k]
    
    # Take info on how many recordings per area have been averaged
    n_rec = [areas_avgs[k]['num'] for k in areas_avgs.keys()]
    
    # Take montage info
    mont = raw_epo.get_montage().get_positions()
    ch_names = []
    ch_x_pos, ch_y_pos, ch_z_pos = [], [], []
    for c in mont['ch_pos'].keys():
        ch_names.append(c)
        ch_x_pos.append(mont['ch_pos'][c][0])
        ch_y_pos.append(mont['ch_pos'][c][1])
        ch_z_pos.append(mont['ch_pos'][c][2])
    ch_x_pos = np.hstack(ch_x_pos)
    ch_y_pos = np.hstack(ch_y_pos)
    ch_z_pos = np.hstack(ch_z_pos)
    mont_dict = {'ch_names': ch_names, 'ch_x_pos': ch_x_pos, 
                 'ch_y_pos': ch_y_pos, 'ch_z_pos': ch_z_pos, 
                 'coord_frame': mont['coord_frame'], 'nasion': mont['nasion'], 
                 'lpa': mont['lpa'], 'rpa': mont['rpa']}
    
    
    # Make the DataArray for raws
    raw_info = {'subjects': subjects,
                'nights': nights,
                'sfreq': raw_epo.info['sfreq'], 
                'highpass': raw_epo.info['highpass'], 
                'lowpass': raw_epo.info['lowpass'],
                'n_rec': n_rec}
    raw_data = [(areas_avgs[k]['raw_sum'] / areas_avgs[k]['num']) 
                for k in areas_avgs.keys()]
    raw_data = np.stack(raw_data, 0)
    raw_data = xr.DataArray(raw_data, 
                            coords=(list(areas_avgs.keys()), 
                                    raw_epo.ch_names, 
                                    raw_epo.times), 
                            dims=['rois', 'channels', 'times'],
                            attrs=raw_info | mont_dict)
    # Make the DataArray for epochs
    epo_info = {'subjects': subjects,
                'nights': nights,
                'sfreq': epochs.info['sfreq'], 
                'highpass': epochs.info['highpass'], 
                'lowpass': epochs.info['lowpass'],
                'n_rec': n_rec}
    epo_data = [(areas_avgs[k]['epo_sum'] / areas_avgs[k]['num']) 
                for k in areas_avgs.keys()]
    epo_data = np.stack(epo_data, 0)
    epo_data = xr.DataArray(epo_data, 
                            coords=(list(areas_avgs.keys()), 
                                    epochs.ch_names, 
                                    epochs.times), 
                            dims=['rois', 'channels', 'times'],
                            attrs=epo_info | mont_dict)
    
    return raw_data, epo_data
    
    
if __name__ == '__main__':
    import closedloop
    import pandas as pd
    
    # prj_data = '/home/ruggero.basanisi/data/tweakdreams'
    prj_data = '/media/jerry/ruggero/tweakdreams'

    data_dir = prj_data
    subjects = ['TD028', 'TD034']
    nights = ['N1', 'N2', 'N3', 'N4']
    # nights = ['N4']
    swori_fnames = ['surf_point_origins.txt', 
                    'geodesic_origins.txt', 
                    'spherical_origins.txt']
    swori_labels = ['source', 'geodesic', 'spherical']
    
    # # aparc_fname = op.join(os.getcwd(), 'closedloop', 'data', 'aparc.xlsx')
    # abs_path = op.dirname(closedloop.__file__)
    # aparc_fname = op.join(abs_path, 'viz', 'aparc.xlsx')
    
    # aparc = pd.read_excel(aparc_fname)
    # hemi = ['-lh', '-rh']
    
    # # ROIs by lobes
    # areas_lobes = {}
    # lobes = np.unique(np.array(aparc['lobe']))
    # for h in hemi:
    #     for l in lobes:
    #         areas_lobes[l+h] = []
    # for l, r in zip(aparc['lobe'], aparc['roi']):
    #     for h in hemi:
    #         areas_lobes[l+h].append(r+h)
            
    # # ROIs by position
    # areas_pos = {}
    # pos = np.unique(np.array(aparc['position']))
    # for h in hemi:
    #     for p in pos:
    #         areas_pos[p+h] = []
    # for p, r in zip(aparc['position'], aparc['roi']):
    #     for h in hemi:
    #         areas_pos[p+h].append(r+h)
            
    # # ROIs by ROIs
    # areas_rois = {}
    # rois = np.unique(np.array(aparc['roi']))
    # for h in hemi:
    #     for r in rois:
    #         areas_rois[r+h] = [r+h]
            
    # areas = [areas_lobes, areas_pos, areas_rois]
    # areas_labels = ['lobes', 'areas', 'rois']
            
    areas_fo = {'cingulate-lh': ['rostralanteriorcingulate-lh',
                                 'caudalanteriorcingulate-lh'],
                'cingulate-rh': ['rostralanteriorcingulate-rh',
                                 'caudalanteriorcingulate-rh'],
                'occipital-lh': ['lateraloccipital-lh',
                                 'cuneus-lh',
                                 'pericalcarine-lh',
                                 'lingual-lh'],
                'occipital-rh': ['lateraloccipital-rh',
                                 'cuneus-rh',
                                 'pericalcarine-rh',
                                 'lingual-rh']}
    
    areas = [areas_fo]
    areas_labels = ['fronto-occipital']
    
    # areas = {'subcortical-lh': ['subcortical-lh'], 
    #          'subcortical-rh': ['subcortical-rh']}
    # areas = [areas_rois]
    
    for swfn, swlb in zip(swori_fnames, swori_labels):
        for ar, arlb in zip(areas, areas_labels):
            raw_topo, epo_topo = compute_topographies(subjects, nights, 
                                                      swfn, ar)

            # res_dir = '/home/ruggero.basanisi/results/closedloop'
            res_dir = '/media/jerry/ruggero/results/closedloop'
            topo_dir = op.join(res_dir, 'topographies', 'subjects', 'nights')
            os.makedirs(topo_dir, exist_ok=True)
            
            rt_fname = op.join(topo_dir, f'raw_topo_{arlb}_{swlb}.nc')
            et_fname = op.join(topo_dir, f'epo_topo_{arlb}_{swlb}.nc')
            
            raw_topo.to_netcdf(rt_fname, format='NETCDF4')
            epo_topo.to_netcdf(et_fname, format='NETCDF4')
    
    # for sbj in subjects:
    #     for n in nights:
    #         for swfn, swlb in zip(swori_fnames, swori_labels):
    #             for ar, arlb in zip(areas, areas_labels):
    #                 raw_topo, epo_topo = compute_topographies(sbj, n, swfn, ar)

    #                 res_dir = '/home/ruggero.basanisi/results/closedloop'
    #                 topo_dir = op.join(res_dir, 'topographies', sbj, n)
    #                 os.makedirs(topo_dir, exist_ok=True)
                    
    #                 rt_fname = op.join(topo_dir, f'raw_topo_{arlb}_{swlb}.nc')
    #                 et_fname = op.join(topo_dir, f'epo_topo_{arlb}_{swlb}.nc')
                    
    #                 raw_topo.to_netcdf(rt_fname, format='NETCDF4')
    #                 epo_topo.to_netcdf(et_fname, format='NETCDF4')
