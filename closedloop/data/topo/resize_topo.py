import mne
import os.path as op
import xarray as xr
import numpy as np
from scipy.spatial.transform import Rotation
from closedloop.data.utils.utils import read_elc

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('QtAgg')

def kabsch_algorithm(P, Q):
    """
    Computes the optimal rotation matrix and translation vector to align two point sets.
    
    :param P: (N, 3) array of common points in set 1 (reference set)
    :param Q: (N, 3) array of corresponding points in set 2 (to be transformed)
    :return: 3x3 rotation matrix R and 3x1 translation vector t
    """
    # Compute centroids
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    
    # Center the points around their centroids
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    # Compute covariance matrix
    H = P_centered.T @ Q_centered
    
    # Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation matrix
    R_opt = U @ Vt

    # Ensure a proper rotation (avoid reflections)
    if np.linalg.det(R_opt) < 0:
        Vt[-1, :] *= -1
        R_opt = U @ Vt
    
    # Compute translation vector
    t_opt = centroid_P - R_opt @ centroid_Q
    
    return R_opt, t_opt

def align_point_sets(set1, set2, labels_s1, labels_s2):
    """
    Rotates and translates set2 to align its common points with set1.
    
    :param set1: Dictionary {label: (x, y, z)} for 256 points
    :param set2: Dictionary {label: (x, y, z)} for 64 points
    :param common_labels: List of labels common to both sets
    :return: Transformed version of set2 and the transformation parameters (R, t)
    """
    # Extract common points
    P = np.array([set1[label] for label in labels_s1])  # From set1 (reference)
    Q = np.array([set2[label] for label in labels_s2])  # From set2 (to be transformed)
    
    # Compute optimal rotation matrix and translation vector
    # R_opt = kabsch_algorithm(P, Q)
    R_opt, t_opt = kabsch_algorithm(P, Q)
    
    # Rotate all points in set2
    # set2_new = {label: tuple(R_opt @ np.array(pos)) for label, pos in set2.items()}
    # Apply transformation to all points in set2
    set2_new = {label: np.array(R_opt @ np.array(pos) + t_opt) for label, pos in set2.items()}
    
    return set2_new, R_opt, t_opt


def compute_translation(set1, set2, labels_s1, labels_s2):
    """
    Computes the optimal translation vector to align the common points in set2 to set1.
    
    :param set1: Dictionary {label: (x, y, z)} for 256 points (reference)
    :param set2: Dictionary {label: (x, y, z)} for 64 points (to be translated)
    :param common_labels: List of labels common to both sets
    :return: Transformed set2 and the translation vector
    """
    # Extract common points
    P = np.array([set1[label] for label in labels_s1])  # From set1 (reference)
    Q = np.array([set2[label] for label in labels_s2])  # From set2 (to be translated)
    
    # Compute centroids
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    
    # Compute translation vector
    t_opt = centroid_P - centroid_Q
    
    # Apply translation to all points in set2
    set2_translated = {label: tuple(np.array(pos) + t_opt) for label, pos in set2.items()}
    
    return set2_translated, t_opt


def resize_topo_templates(orig_template, orig_topo, new_template, new_topo):
    
    r = Rotation.from_euler('xyz', [0, 0, 90], degrees=True)
    
    ori_temp = xr.load_dataarray(orig_template)
    ori_topo = read_elc(orig_topo)
    
    rot_ori_topo = {}
    for ch in ori_topo.get_positions()['ch_pos'].keys():
        if ch != 'VEOGR':
            rot_ori_topo[ch] = r.apply(ori_topo.get_positions()['ch_pos'][ch])
        
    ori_topo = mne.channels.make_dig_montage(ch_pos=rot_ori_topo, 
                                             nasion=None, 
                                             lpa=None, 
                                             rpa=None, 
                                             coord_frame='head')
    
    end_topo = read_elc(new_topo)
    
    rot_end_topo = {}
    for ch in end_topo.get_positions()['ch_pos'].keys():
        if ch != 'EOG':
            rot_end_topo[ch] = r.apply(end_topo.get_positions()['ch_pos'][ch])
        
    end_topo = mne.channels.make_dig_montage(ch_pos=rot_end_topo, 
                                             nasion=None, 
                                             lpa=None, 
                                             rpa=None, 
                                             coord_frame='head')
    
    
    labels_s1 = ['L1Z', 'R1Z', 'Z9Z', 'L5G', 'R5G', 'L5E', 'R5E', 'Z19Z']
    labels_s2 = ['1L', '1R', '4Z', '3LD', '3RD', '3LC', '3RC', '9Z']
    # labels_s1 = ['Z9Z', 'L4H', 'R4H', 'Z19Z']
    # labels_s2 = ['4Z', '3LD', '3RD', '9Z']
    # labels_s1 = ['Z9Z', 'Z19Z']
    # labels_s2 = ['4Z', '9Z']
    

    aligned_topo, R, t = align_point_sets(ori_topo.get_positions()['ch_pos'],
                                          end_topo.get_positions()['ch_pos'],
                                          labels_s1, labels_s2)
    # aligned_topo, t = compute_translation(ori_topo.get_positions()['ch_pos'],
    #                                       end_topo.get_positions()['ch_pos'],
    #                                       labels_s1, labels_s2)
    
    align_topo = mne.channels.make_dig_montage(ch_pos=aligned_topo, 
                                               nasion=None, 
                                               lpa=None, 
                                               rpa=None, 
                                               coord_frame='head')

    eeg_pos = {}
    for ch in ori_temp.ch_names:
        if ch in ori_topo.get_positions()['ch_pos'].keys():
            eeg_pos[ch] = ori_topo.get_positions()['ch_pos'][ch]
            
    for ch in aligned_topo.keys():
        if ch not in eeg_pos.keys():
            eeg_pos[ch] = aligned_topo[ch]
    
    mix_mont = mne.channels.make_dig_montage(ch_pos=eeg_pos, 
                                             nasion=None, 
                                             lpa=None, 
                                             rpa=None, 
                                             coord_frame='head')
    
    mix_info = mne.create_info(ch_names=mix_mont.ch_names, 
                               sfreq=ori_temp.sfreq, 
                               ch_types='eeg')
    
    zeros = np.zeros((len(ori_temp.rois), len(end_topo.ch_names), len(ori_temp.times)))
    data = np.concatenate((ori_temp.values, zeros), axis=1)
            
    epo = mne.EpochsArray(data, mix_info, tmin=ori_temp.times[0].values)
    
    epo = epo.set_montage(mix_mont)
    
    epo.info['bads'] = list(end_topo.ch_names)
    
    epo = epo.interpolate_bads(reset_bads=True, mode='accurate', method='spline')
    
    epo = epo.drop_channels(ori_topo.ch_names)
    
    for e in range(len(epo)):
        epo.copy()[e].average().plot_topomap(times=[-1., -.5, 0., .5, 1.], show=False)
        
    plt.show(block=True)
    
    new_ch_pos = epo.get_montage().get_positions()['ch_pos']
        
    attributes = ori_temp.attrs
    attributes['ch_names'] = epo.ch_names
    attributes['ch_x_pos'] = [new_ch_pos[c][0] for c in new_ch_pos]
    attributes['ch_y_pos'] = [new_ch_pos[c][1] for c in new_ch_pos]
    attributes['ch_z_pos'] = [new_ch_pos[c][2] for c in new_ch_pos]
    attributes['nasion'] = 'None'
    attributes['lpa'] = 'None'
    attributes['rpa'] = 'None'
    final_template = xr.DataArray(epo.get_data(), 
                                  coords=[ori_temp.rois, 
                                          epo.ch_names, 
                                          ori_temp.times],
                                  dims=['rois', 'channels', 'times'],
                                  attrs=attributes)
    
    final_template.to_netcdf(new_template)
    
    return orig_template


def resize_topo_raws(orig_raw, orig_topo, new_raw, new_topo):
    
    # define rotation matrix for the standard 256 an 64 channel caps
    r = Rotation.from_euler('xyz', [0, 0, 90], degrees=True)
    
    # remove non-EEG channels
    ori_raw = mne.io.read_raw_fif(orig_raw, preload=True)
    ori_raw.drop_channels(['EMG', 'ECG', 'RES', 'VEOG', 'HEOG']) 
    # PAY ATTENTION TO Z12Z
    
    # load and rotate 256 channels cap EEG positions
    ori_topo = read_elc(orig_topo)
    rot_ori_topo = {}
    for ch in ori_topo.get_positions()['ch_pos'].keys():
        if ch != 'VEOGR':
            rot_ori_topo[ch] = r.apply(ori_topo.get_positions()['ch_pos'][ch])
    # create a 256 channels montage object
    ori_topo = mne.channels.make_dig_montage(ch_pos=rot_ori_topo, 
                                             nasion=None, 
                                             lpa=None, 
                                             rpa=None, 
                                             coord_frame='head')
    
    # set the rotated montage to the raw data 
    ori_raw = ori_raw.set_montage(ori_topo)
    
    # load and rotate the standard 64 channel cap montage
    end_topo = read_elc(new_topo)
    rot_end_topo = {}
    for ch in end_topo.get_positions()['ch_pos'].keys():
        if ch != 'EOG':
            rot_end_topo[ch] = r.apply(end_topo.get_positions()['ch_pos'][ch])
    # create a 64 channels montage object
    end_topo = mne.channels.make_dig_montage(ch_pos=rot_end_topo, 
                                             nasion=None, 
                                             lpa=None, 
                                             rpa=None, 
                                             coord_frame='head')
    # define the common labels between the two montages
    labels_s1 = ['L1Z', 'R1Z', 'Z9Z', 'L5G', 'R5G', 'L5E', 'R5E', 'Z19Z']
    labels_s2 = ['1L', '1R', '4Z', '3LD', '3RD', '3LC', '3RC', '9Z']
    # labels_s1 = ['Z9Z', 'L4H', 'R4H', 'Z19Z']
    # labels_s2 = ['4Z', '3LD', '3RD', '9Z']
    # labels_s1 = ['Z9Z', 'Z19Z']
    # labels_s2 = ['4Z', '9Z']
    
    # align the two montages
    aligned_topo, R, t = align_point_sets(ori_topo.get_positions()['ch_pos'],
                                          end_topo.get_positions()['ch_pos'],
                                          labels_s1, labels_s2)
    # aligned_topo, t = compute_translation(ori_topo.get_positions()['ch_pos'],
    #                                       end_topo.get_positions()['ch_pos'],
    #                                       labels_s1, labels_s2)
    
    # create a montage object for the aligned 64 channels cap
    align_topo = mne.channels.make_dig_montage(ch_pos=aligned_topo, 
                                               nasion=None, 
                                               lpa=None, 
                                               rpa=None, 
                                               coord_frame='head')

    zero_data = np.zeros((len(align_topo.ch_names), ori_raw._data.shape[1]))
    align_info = mne.create_info(ch_names=align_topo.ch_names, 
                                 sfreq=ori_raw.info['sfreq'], 
                                 ch_types='eeg')
    
    zero_data = mne.io.RawArray(zero_data, align_info, 
                                first_samp=ori_raw.first_samp)
    zero_data.set_montage(align_topo)

    ori_raw.add_channels([zero_data], force_update_info=True)
    
    ori_raw.info['bads'] = list(align_topo.ch_names)
    
    ori_raw = ori_raw.interpolate_bads(reset_bads=True, mode='accurate', 
                                       method='spline')
    
    ori_raw = ori_raw.drop_channels(ori_topo.ch_names)
    
    ori_raw.plot_sensors(show_names=True, show=False)
    
    ori_raw.save(new_raw, overwrite=True)
    
    return
    
if __name__ == '__main__':
    
    sig = 'epo'
    ref = 'fronto-occipital'
    met = 'geodesic'
    
    template_dir = '/media/jerry/ruggero/results/closedloop/topographies'
    orig_template = op.join(template_dir, 'subjects', 'nights', f'{sig}_topo_{ref}_{met}.nc')
    # orig_raw = '/home/jerry/python_projects/space/closedloop/test_data/TweakDreams/TD010_N1-raw.fif'
    orig_raw = '/media/jerry/ruggero/tweakdreams/mne/TD005/N3/raw/TD005-N3-raw.fif'
    # orig_topo = '/media/jerry/ruggero/eego_montages/CA-205_256_ch_general.elc'
    # orig_topo = '/media/jerry/ruggero/eego_montages/DDE-OP-3598rev01 Electrodes positions for the CA-205.elc'
    orig_topo = '/media/jerry/ruggero/eego_montages/CA-205_ref.elc'
    # orig_topo = '/media/jerry/ruggero/eego_montages/CA-205.nlr.cosimo.elc'
    
    new_template = op.join(template_dir, 'subjects', 'nights', f'{sig}_topo_{ref}_{met}_64ch.nc')
    # new_raw = '/home/jerry/python_projects/space/closedloop/test_data/TweakDreams/TD010_N1_64-raw.fif'
    new_raw = '/home/jerry/python_projects/space/closedloop/test_data/TweakDreams/TD005-N3_64-raw.fif'
    # new_topo = '/media/jerry/ruggero/eego_montages/CA-212_63_ch_general.elc'
    # new_topo = '/media/jerry/ruggero/eego_montages/DDE-OP-3597rev01 Electrodes positions for the CA-212.elc'
    new_topo = '/media/jerry/ruggero/eego_montages/CA-212_ref.elc'
    # new_topo = '/media/jerry/ruggero/eego_montages/CA-212.nlr.cosimo.elc'
    
    # resize_topo_templates(orig_template=orig_template, orig_topo=orig_topo, 
    #                       new_template=new_template, new_topo=new_topo)
    
    resize_topo_raws(orig_raw=orig_raw, orig_topo=orig_topo,
                     new_raw=new_raw, new_topo=new_topo)