import os
import os.path as op
import numpy as np
import mne
from tqdm import tqdm
import pygeodesic.geodesic as geo
from joblib import Parallel, delayed
import pickle
from signal_se import compute_inverse_sources


def compute_distances(idx, vert, tris, used_vert_idx, used_vert_pos, 
                      radius, n_src):
    mesh = geo.PyGeodesicAlgorithmExact(vert, tris)
    
    v_pos = used_vert_pos[idx]
    v_idx = used_vert_idx[idx]
    
    # Computing distances using a spheric radial distance
    dist_sph = np.linalg.norm(used_vert_pos - v_pos, axis=1)
    neighbors_sph = np.where(dist_sph <= radius)[0]
    
    # Add the number of sources used by the previous surface
    neighbors_sph += n_src
    
    # # Collecting spherical neighbors position
    # ngh_sph.append(neighbors_sph)
    
    # Computing distances using geodesic radial distance
    dist_geo, _ = mesh.geodesicDistances([v_idx], used_vert_idx)
    neighbors_geo = np.where(dist_geo <= radius)[0]
    
    # Add the number of sources used by the previous surface
    neighbors_geo += n_src
    
    # # Collecting geodesic neighbors position
    # ngh_geo.append(neighbors_geo)
    return neighbors_sph, neighbors_geo


def find_sw_origin(subject, subjects_dir, epochs_stcs, src_fname, radius=0.01, 
                   t_dist=.05, value='min', n_jobs=-1):
    # Reading labels
    labels = mne.read_labels_from_annot(subject=subject, parc='aparc',
                                        hemi='both', surf_name='white',
                                        subjects_dir=subjects_dir)
    # At least 3 vertices are needed to define an area
    lone_vertices = []
    for i, l in enumerate(labels):
        if len(l.vertices) < 3:
            lone_vertices.append(i)
    if len(lone_vertices) >= 1:
        for i in sorted(lone_vertices, reverse=True):
            del labels[i]
        
    lh_labels = [l for l in labels if l.hemi == 'lh']
    rh_labels = [l for l in labels if l.hemi == 'rh']
        
    l_name = [l.name for l in labels]
    
    # Reading source space
    sources = mne.read_source_spaces(src_fname)
    
    # Compute time window of interest
    tmin, tmax = 0 - t_dist, 0 + t_dist
    
    n_src = 0
    ngh_sph, ngh_geo = [], []
    print('\tComputing spherical and geodesic neighbors of each source',
          f'(selected distance {radius*1000}mm). This may take a while...\n')
    for h, l in zip(range(len(sources)), [lh_labels, rh_labels]):
        # Select one element in sources (usually one hemisphere)
        h_src = sources[h]
        
        # Take all the positions of the vertices composing the surface of 
        # one hemisphere
        h_vert = h_src['rr']
        
        # Take the list of indices of vertices used as sources
        used_vert_idx = h_src['vertno']
        # Take the positions of the vertices used as sources
        used_vert_pos = h_vert[used_vert_idx]
        
        _ngh_sph, _ngh_geo = zip(*Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(compute_distances)
            (idx, vert=h_vert, tris=h_src['tris'],
             used_vert_idx=used_vert_idx, used_vert_pos=used_vert_pos, 
             radius=radius, n_src=n_src) 
            for idx in tqdm(range(len(used_vert_idx)))))
        
        ngh_sph += _ngh_sph
        ngh_geo += _ngh_geo
        
        # Collect the number of sources composing the last surface
        n_src += h_src['nuse']
    
    print('Done!')
    del _ngh_sph
    del _ngh_geo
    
    labels_sph, labels_geo = [], []
    for _i, ep in enumerate(epochs_stcs):
        # Cutting source estimate around the desired time window
        _ep = ep.copy().crop(tmin, tmax)
        
        # print(f'Computing sources values for epoch {_i}')
        nsph_vals, ngeo_vals = [], []
        for nsph, ngeo in zip(ngh_sph, ngh_geo):
            
            nsval = _ep.data[nsph, :].mean(axis=0)
            
            # If we want to take the max(abs) across neighbors sources
            # idx_max_abs = np.where(abs(nsval)==np.max(abs(nsval)))[0][0]
            # nsph_vals.append(nsval[idx_max_abs])
            
            # If we want to take the average across neighbors sources
            nsph_vals.append(nsval.mean())
            
            ngval = _ep.data[ngeo, :].mean(axis=0)
            
            # If we want to take the max(abs) across neighbors sources
            # idx_max_abs = np.where(abs(ngval)==np.max(abs(ngval)))[0][0]
            # ngeo_vals.append(ngval[idx_max_abs])
            
            # If we want to take the average across neighbors sources
            ngeo_vals.append(ngval.mean())
        
        nsph_vals = np.array(nsph_vals)
        ngeo_vals = np.array(ngeo_vals)
        
        if value == 'max':
            idx_sph = np.where(nsph_vals == np.max(nsph_vals))[0][0]
            idx_geo = np.where(ngeo_vals == np.max(ngeo_vals))[0][0]
        elif value == 'min':
            idx_sph = np.where(nsph_vals == np.min(nsph_vals))[0][0]
            idx_geo = np.where(ngeo_vals == np.min(ngeo_vals))[0][0]
        elif value == 'abs':
            idx_sph = np.where(abs(nsph_vals) == np.max(nsph_vals))[0][0]
            idx_geo = np.where(abs(ngeo_vals) == np.max(ngeo_vals))[0][0]
        else:
            raise(ValueError)
        
        lname = None
        n_lh_src = sources[0]['nuse']
        if idx_sph < n_lh_src:
            vert = sources[0]['vertno'][idx_sph]
            for l in lh_labels:
                if vert in l.vertices:
                    lname = l.name
        elif idx_sph >= n_lh_src:
            vert = sources[1]['vertno'][idx_sph - n_lh_src]
            for l in rh_labels:
                if vert in l.vertices:
                    lname = l.name
                    
        if lname is None:
            if idx_sph < n_lh_src:
                lname = 'subcortical-lh'
            elif idx_sph >= n_lh_src:
                lname = 'subcortical-rh'
            
        labels_sph.append(lname)
        
        print(f'Spherical detection, {value} value in: {lname}')
        
        lname = None
        n_lh_src = sources[0]['nuse']
        if idx_geo < n_lh_src:
            vert = sources[0]['vertno'][idx_geo]
            for l in lh_labels:
                if vert in l.vertices:
                    lname = l.name
        elif idx_geo >= n_lh_src:
            vert = sources[1]['vertno'][idx_geo - n_lh_src]
            for l in rh_labels:
                if vert in l.vertices:
                    lname = l.name
                    
        if lname is None:
            if idx_geo < n_lh_src:
                lname = 'subcortical-lh'
            elif idx_geo >= n_lh_src:
                lname = 'subcortical-rh'
        
        labels_geo.append(lname)
        
        print(f'Geodesic detection, {value} value in: {lname}')
        
    return labels_sph, labels_geo


if __name__ == '__main__':
    
    prj_data = '/home/ruggero.basanisi/data/tweakdreams'
    
    subjects = ['TD001']
    nights = ['N1']
    
    fs_dir = op.join(prj_data, 'freesurfer')
    
    for sbj in subjects:
        for n in nights:
            
            epo_dir = op.join(prj_data, 'mne', sbj, n, 'epo')
            bem_dir = op.join(prj_data, 'mne', sbj, n, 'bem')
            fwd_dir = op.join(prj_data, 'mne', sbj, n, 'fwd')
            src_dir = op.join(prj_data, 'mne', sbj, n, 'src')
            
            stc_dir = op.join(prj_data, 'mne', sbj, n, 'stc')
            ltc_dir = op.join(prj_data, 'mne', sbj, n, 'ltc')
            
            fwd_fname = op.join(fwd_dir, f'{sbj}_{n}-fwd.fif')
            src_fname = op.join(src_dir, f'{sbj}_{n}-src.fif')
            
            epochs = []
            aw = [a for a in os.listdir(epo_dir) if a.startswith('aw_')]
            aw.sort()
            # aw = ['aw_2']
            
            for _aw in aw:
                epo_fname = op.join(epo_dir, _aw, 'envelope_sw_clean-epo.fif')
                bln_fname = epo_fname
                
                stc_fname = op.join(stc_dir, _aw, 'sws_surf.stc')
                ltc_fname = op.join(ltc_dir, _aw, 'sws_labels_tc.nc')
                
                sph_fname = op.join(ltc_dir, _aw, 'spherical_origins.txt')
                geo_fname = op.join(ltc_dir, _aw,  'geodesic_origins.txt')
                
                if op.exists(epo_fname):
                    # os.makedirs(op.join(stc_dir, _aw), exist_ok=True)
                    os.makedirs(op.join(ltc_dir, _aw), exist_ok=True)
                    
                    # stc = compute_lcmv_sources(epo_fname, bln_fname, 
                    #                            fwd_fname, events=None)
                    stc = compute_inverse_sources(epo_fname, bln_fname, 
                                                  fwd_fname)
                    # stc.save(stc_fname, format='stc', overwrite=True)
                    
                    sph_orig, geo_orig = find_sw_origin(sbj, fs_dir, stc, 
                                                        src_fname, 
                                                        radius=0.01, 
                                                        t_dist=.05, 
                                                        value='abs',
                                                        n_jobs=64)
                    
                    with open(sph_fname, 'wb') as f:
                        pickle.dump(sph_orig, f)
                        
                    with open(geo_fname, 'wb') as f:
                        pickle.dump(geo_orig, f)
                    
    # Concatenate all SWs epochs of a subject over one night      
    # for sbj in subjects:
    #     for n in nights:
            
    #         epo_dir = op.join(prj_data, 'mne', sbj, n, 'epo')
    #         bem_dir = op.join(prj_data, 'mne', sbj, n, 'bem')
    #         fwd_dir = op.join(prj_data, 'mne', sbj, n, 'fwd')
    #         src_dir = op.join(prj_data, 'mne', sbj, n, 'src')
            
    #         stc_dir = op.join(prj_data, 'mne', sbj, n, 'stc')
    #         ltc_dir = op.join(prj_data, 'mne', sbj, n, 'ltc')
            
    #         fwd_fname = op.join(fwd_dir, f'{sbj}_{n}-fwd.fif')
    #         src_fname = op.join(src_dir, f'{sbj}_{n}-src.fif')
            
    #         aw = [a for a in os.listdir(epo_dir) if a.startswith('aw_')]
    #         aw.sort()
    #         # aw = ['aw_3']
            
    #         all_epo, all_bln = [], []
    #         for _aw in aw:
    #             epo_fname = op.join(epo_dir, _aw, 'envelope_sw_clean-epo.fif')
    #             bln_fname = epo_fname
                
    #             if op.exists(epo_fname):
    #                 all_epo.append(mne.read_epochs(epo_fname))
    #                 all_bln.append(mne.read_epochs(bln_fname))
                
    #         all_epo = mne.concatenate_epochs(all_epo)
    #         all_bln = mne.concatenate_epochs(all_bln)
                
    #         stc_fname = op.join(stc_dir, 'all_sws_surf.stc')
    #         ltc_fname = op.join(ltc_dir, 'all_sws_labels_tc.nc')
            
    #         # os.makedirs(op.join(stc_dir, _aw), exist_ok=True)
    #         os.makedirs(ltc_dir, exist_ok=True)
            
    #         # stc = compute_lcmv_sources(epo_fname, bln_fname, 
    #         #                            fwd_fname, events=None)
    #         stc = compute_inverse_sources(all_epo, all_bln, 
    #                                       fwd_fname)
    #         # stc.save(stc_fname, format='stc', overwrite=True)
            
    #         find_sw_origin(sbj, fs_dir, stc, src_fname, 
    #                        radius=0.02, t_dist=.05, value='abs')
