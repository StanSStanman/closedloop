import mne
from mne.beamformer import (make_lcmv, apply_lcmv_epochs, 
                            make_dics, apply_dics_epochs)
import xarray as xr
import numpy as np


def compute_dics_sources(epochs, bln_epo, fwd_fname, events=None):

    assert isinstance(epochs, (mne.BaseEpochs, mne.Epochs, str)), \
        'epochs must be a mne epochs object or a string'
    assert isinstance(bln_epo, (mne.BaseEpochs, mne.Epochs, str)), \
        'bln_epo must be a mne epochs object or a string'
    
    if isinstance(epochs, str):
        epochs = mne.read_epochs(epochs, preload=True)
        
    epochs_ev = epochs.copy()
    if events is not None:
        assert isinstance(events, list)
        eve_pick = []
        for e in events:
            eve_pick.append(np.where(epochs_ev.events[:, -1] == e)[0])
        eve_pick = np.sort(np.concatenate((eve_pick)))
        epochs_ev = epochs_ev[eve_pick]
    
    epochs_ev.pick_types(eeg=True)
    # epochs_ev = epochs_ev.apply_baseline((2., 5.))
    # epochs_ev = apply_artifact_rejection(epochs_ev, subject, session,
    #                                      event, reject='trials')
    # epochs_ev = epochs_ev.filter(.5, 4., n_jobs='cuda')
    
    if isinstance(bln_fname, str):
        bln_epo = mne.read_epochs(bln_epo, preload=True)

    bln_epo.pick_types(eeg=True)   
    # bln_epo = bln_epo.filter(.5, 4., n_jobs='cuda')
    # Make noise epochs
    # bln_epo = bln_epo.apply_baseline((2., 5.))
    # bln_epo._data -= bln_epo._data.mean(axis=0, keepdims=True)
    bln_epo = bln_epo.subtract_evoked()
    
    # Load the forward solution
    fwd = mne.read_forward_solution(fwd_fname)
    # fwd = mne.forward.convert_forward_solution(fwd, surf_ori=True,
    #                                            force_fixed=True,
    #                                            use_cps=True)
    fwd = mne.forward.convert_forward_solution(fwd, surf_ori=True,
                                               force_fixed=False,
                                               use_cps=True)
    
    # To apply a beamformer to EEG data is mandatory to have a reference 
    # projection, we can create this by doing:
    epochs_ev = epochs_ev.set_eeg_reference(ref_channels='average',
                                            projection=True, forward=None)
    bln_epo = bln_epo.set_eeg_reference(ref_channels='average',
                                        projection=True, forward=None)
    # This is a classical method, but one can also use a REST reference, 
    # for which the forwrd model is needed in order to compute the noise 
    # outside the brain 
    # (to test, see: https://doi.org/10.1088/0967-3334/22/4/305)
    # epochs_ev = epochs_ev.set_eeg_reference(ref_channels='REST',
    #                                         projection=False, forward=fwd)
    # bln_epo = bln_epo.set_eeg_reference(ref_channels='REST',
    #                                     projection=False, forward=fwd)
    
    
    # DICS algorithm uses CSD instead of covariance
    frequencies = np.geomspace(.5, 4., 10)
    n_cycles = frequencies / 2.
    
    csd = mne.time_frequency.csd_morlet(epochs_ev, frequencies, 
                                        tmin=-1., tmax=1., 
                                        n_cycles=n_cycles, n_jobs=32)
    csd = csd.mean()
    
    noise_csd = mne.time_frequency.csd_morlet(bln_epo, frequencies, 
                                              tmin=-5., tmax=-2., 
                                              n_cycles=n_cycles, n_jobs=32)
    noise_csd = noise_csd.mean()
    
    # csd = mne.time_frequency.csd_multitaper(epochs_ev, fmin=2., fmax=2.5,
    #                                         tmin=-2., tmax=2., bandwidth=4.,
    #                                         n_jobs=32)
    
    # noise_csd = mne.time_frequency.csd_multitaper(bln_epo, fmin=2., 
    #                                               fmax=2.5, tmin=-2., tmax=2.,
    #                                               bandwidth=4., n_jobs=32)
    
    # Fixed orientation
    # filters = make_lcmv(epochs_ev.info, fwd, data_cov=covariance,
    #                     noise_cov=noise_cov, reg=0.05, pick_ori='normal',
    #                     weight_norm='unit-noise-gain', reduce_rank=False,
    #                     depth=.8, inversion='matrix')
    
    # Free orientation, surface
    filters = make_dics(epochs_ev.info, fwd, csd=csd,
                        reg=0.1, noise_csd=noise_csd, pick_ori='normal',
                        weight_norm='nai', reduce_rank=False,
                        depth=.8, inversion='matrix')
    # Free orientation, volume
    # filters = make_lcmv(epochs_ev.info, fwd, data_cov=covariance,
    #                     noise_cov=noise_cov, reg=0.1, pick_ori='vector',
    #                     weight_norm='nai', reduce_rank=False,
    #                     depth=.8, inversion='matrix')

    epochs_stcs = apply_dics_epochs(epochs_ev, filters, return_generator=True)
    
    # stc = []
    # for es in epochs_stcs:
    #     stc.append(es)
    # stc = np.mean(stc)
    # evk = mne.apply_forward(fwd, stc, epochs_ev.info, use_cps=True)
    # evk.plot_topomap(times=np.arange(-.25, .251, .05), colorbar=False)
    # epochs_ev.copy().average().plot_topomap(times=np.arange(-.25, .251, .05), 
    #                                         colorbar=False)
    
    for _i, es in enumerate(epochs_stcs):
        evk = mne.apply_forward(fwd, es, epochs_ev.info, use_cps=True)
        evk.plot_topomap(times=np.arange(-.25, .251, .05), colorbar=False)
        epochs_ev.copy()[_i].average().apply_proj().plot_topomap(
            times=np.arange(-.25, .251, .05), colorbar=False)
        
    # brain = es.plot(hemi='split', subjects_dir='/home/ruggero.basanisi/data/tweakdreams/freesurfer', colormap='RdBu_r')
    # lh_pk = es.get_peak(hemi='lh', tmin=-.05, tmax=.05, mode='neg')[0]
    # rh_pk = es.get_peak(hemi='rh', tmin=-.05, tmax=.05, mode='neg')[0]
    # brain.add_foci(lh_pk, hemi='lh', coords_as_verts=True, alpha=.6)
    # brain.add_foci(rh_pk, hemi='rh', coords_as_verts=True, alpha=.6)
    
    epochs_stcs = apply_lcmv_epochs(epochs_ev, filters, return_generator=True)

    return epochs_stcs


def compute_lcmv_sources(epochs, bln_epo, fwd_fname):

    assert isinstance(epochs, (mne.BaseEpochs, mne.Epochs, str)), \
        'epochs must be a mne epochs object or a string'
    assert isinstance(bln_epo, (mne.BaseEpochs, mne.Epochs, str)), \
        'bln_epo must be a mne epochs object or a string'
    
    if isinstance(epochs, str):
        epochs = mne.read_epochs(epochs, preload=True)
        
    epochs_ev = epochs.copy()    
    epochs_ev.pick_types(eeg=True)
    # epochs_ev = epochs_ev.apply_baseline((2., 5.))
    # epochs_ev = epochs_ev.filter(.5, 4., n_jobs='cuda')
    
    if isinstance(bln_fname, str):
        bln_epo = mne.read_epochs(bln_epo, preload=True)

    bln_epo.pick_types(eeg=True)   
    # bln_epo = bln_epo.filter(.5, 4., n_jobs='cuda')
    
    # Make noise epochs
    bln_epo = bln_epo.subtract_evoked()

    # Convert fwd solution to use surface infos
    fwd = mne.read_forward_solution(fwd_fname)
    fwd = mne.forward.convert_forward_solution(fwd, surf_ori=True,
                                               force_fixed=False,
                                               use_cps=True)
    
    # To apply a beamformer to EEG data is mandatory to have a reference 
    # projection, we can create this by doing:
    epochs_ev = epochs_ev.set_eeg_reference(ref_channels='average',
                                            projection=True, forward=None)
    bln_epo = bln_epo.set_eeg_reference(ref_channels='average',
                                        projection=True, forward=None)
    # This is a classical method, but one can also use a REST reference, 
    # for which the forwrd model is needed in order to compute the noise 
    # outside the brain 
    # (to test, see: https://doi.org/10.1088/0967-3334/22/4/305)
    # epochs_ev = epochs_ev.set_eeg_reference(ref_channels='REST',
    #                                         projection=False, forward=fwd)
    # bln_epo = bln_epo.set_eeg_reference(ref_channels='REST',
    #                                     projection=False, forward=fwd)
    
    # Covariance and noise covariance estimate
    covariance = mne.compute_covariance(epochs_ev, tmin=-8., tmax=-2.,
                                        keep_sample_mean=True,
                                        method='shrunk',
                                        rank=None,
                                        n_jobs=-1)
    covariance = mne.cov.regularize(covariance, epochs_ev.info, eeg=0.5)
    # covariance = mne.make_ad_hoc_cov(epochs_ev.info)
    
    # noise_cov = mne.compute_covariance(bln_epo, tmin=None, tmax=None,
    #                                    method=['shrunk', 'diagonal_fixed',
    #                                            'empirical'],
    #                                    cv=3, rank=None, n_jobs=-1)
    noise_cov = mne.compute_covariance(bln_epo, tmin=-8., tmax=-2.,
                                        keep_sample_mean=True,
                                        method='shrunk',
                                        rank=None,
                                        n_jobs=-1)
    noise_cov = mne.cov.regularize(noise_cov, bln_epo.info, eeg=0.5)
    # noise_cov = mne.make_ad_hoc_cov(bln_epo.info)
    
    # Build spatial filter
    # Fixed orientation
    # filters = make_lcmv(epochs_ev.info, fwd, data_cov=covariance,
    #                     noise_cov=noise_cov, reg=0.05, pick_ori='normal',
    #                     weight_norm='unit-noise-gain', reduce_rank=False,
    #                     depth=.8, inversion='matrix')
    
    # Free orientation, surface
    filters = make_lcmv(epochs_ev.info, fwd, data_cov=covariance,
                        noise_cov=noise_cov, reg=0.1, pick_ori='normal',
                        weight_norm='nai', reduce_rank=False,
                        depth=.8, inversion='matrix')
    # Free orientation, volume
    # filters = make_lcmv(epochs_ev.info, fwd, data_cov=covariance,
    #                     noise_cov=noise_cov, reg=0.1, pick_ori='vector',
    #                     weight_norm='nai', reduce_rank=False,
    #                     depth=.8, inversion='matrix')

    # Apply filters, compute sources time course
    epochs_stcs = apply_lcmv_epochs(epochs_ev, filters, return_generator=True)
    
    # Plot stuff while debugging
    # stc = []
    # for es in epochs_stcs:
    #     stc.append(es)
    # stc = np.mean(stc)
    # evk = mne.apply_forward(fwd, stc, epochs_ev.info, use_cps=True)
    # evk.plot_topomap(times=np.arange(-.25, .251, .05), colorbar=False)
    # epochs_ev.copy().average().plot_topomap(times=np.arange(-.25, .251, .05), 
    #                                         colorbar=False)
        
    # brain = es.plot(hemi='split', subjects_dir='/home/ruggero.basanisi/data/tweakdreams/freesurfer', colormap='RdBu_r')
    # lh_pk = es.get_peak(hemi='lh', tmin=-.05, tmax=.05, mode='neg')[0]
    # rh_pk = es.get_peak(hemi='rh', tmin=-.05, tmax=.05, mode='neg')[0]
    # brain.add_foci(lh_pk, hemi='lh', coords_as_verts=True, alpha=.6)
    # brain.add_foci(rh_pk, hemi='rh', coords_as_verts=True, alpha=.6)
    
    # Uncomment if plotting is uncommented 
    # epochs_stcs = apply_lcmv_epochs(epochs_ev, filters, return_generator=True)

    return epochs_stcs


def compute_inverse_sources(epochs, bln_epo, fwd_fname):
    
    assert isinstance(epochs, (mne.BaseEpochs, mne.Epochs, str)), \
        'epochs must be a mne epochs object or a string'
    assert isinstance(bln_epo, (mne.BaseEpochs, mne.Epochs, str)), \
        'bln_epo must be a mne epochs object or a string'
    
    if isinstance(epochs, str):
        epochs = mne.read_epochs(epochs, preload=True)
        
    epochs_ev = epochs.copy()
    epochs_ev.pick(['eeg'])

    if isinstance(bln_epo, str):
        bln_epo = mne.read_epochs(bln_epo, preload=True)

    bln_epo.pick(['eeg'])
    
    # Make noise epochs
    bln_epo = bln_epo.subtract_evoked()
    
    # Convert fwd solution to use surface infos
    fwd = mne.read_forward_solution(fwd_fname)
    # fwd = mne.forward.convert_forward_solution(fwd, surf_ori=True,
    #                                            force_fixed=True,
    #                                            use_cps=True)
    fwd = mne.forward.convert_forward_solution(fwd, surf_ori=True,
                                               force_fixed=False,
                                               use_cps=True)
    
    # To apply a beamformer to EEG data is mandatory to have a reference 
    # projection, we can create this by doing:
    epochs_ev = epochs_ev.set_eeg_reference(ref_channels='average',
                                            projection=True, forward=None)
    # epochs_ev.apply_proj()
    bln_epo = bln_epo.set_eeg_reference(ref_channels='average',
                                        projection=True, forward=None)
    
    # This is a classical method, but one can also use a REST reference, 
    # for which the forwrd model is needed in order to compute the noise 
    # outside the brain 
    # (to test, see: https://doi.org/10.1088/0967-3334/22/4/305)
    # epochs_ev = epochs_ev.set_eeg_reference(ref_channels='REST',
    #                                         projection=False, forward=fwd)
    # bln_epo = bln_epo.set_eeg_reference(ref_channels='REST',
    #                                     projection=False, forward=fwd)

    # Covariance estimated on the data
    # covariance = mne.compute_covariance(epochs_ev, keep_sample_mean=False,
    #                                     method=['shrunk', 'diagonal_fixed',
    #                                             'empirical'], cv=3, n_jobs=-1)
    # covariance = mne.compute_covariance(epochs_ev, keep_sample_mean=True,
    #                                     tmin=1., tmax=5., method_params={'diagonal_fixed': {'eeg': .001}},
    #                                     method='diagonal_fixed', n_jobs=-1)
    covariance = mne.compute_covariance(epochs_ev, keep_sample_mean=True,
                                        tmin=-8., tmax=-2., rank=None,
                                        method='shrunk', n_jobs=-1)
    covariance = mne.cov.regularize(covariance, epochs_ev.info, eeg=.5)

    # Covariance estimated on the baseline
    # covariance = mne.compute_covariance(bln_epo, keep_sample_mean=True,
    #                                     tmin=-5., tmax=-2.,
    #                                     method=['shrunk', 'diagonal_fixed',
    #                                             'empirical'], cv=3, n_jobs=-1)
    # covariance = mne.compute_covariance(bln_epo, keep_sample_mean=True,
    #                                     tmin=-.5, tmax=.5,
    #                                     method='empirical', n_jobs=-1)
    # covariance = mne.cov.regularize(covariance, bln_epo.info, eeg=.1)
    
    # Using ad-hoc covariance (makes an identity matrix)    
    # covariance = mne.make_ad_hoc_cov(epochs_ev.info)
    
    # Free orientation, depth weighted
    # inverse_operator = mne.minimum_norm.make_inverse_operator(epochs_ev.info,
    #                                                           fwd, covariance,
    #                                                           fixed=False,
    #                                                           loose=1,
    #                                                           depth=0.8)
    
    # Loose constraint, depth weighted
    inverse_operator = mne.minimum_norm.make_inverse_operator(epochs_ev.info,
                                                              fwd, covariance,
                                                              fixed=False,
                                                              loose=.2,
                                                              depth=.8)
    
    # Fixed constraint, depth weighted
    # inverse_operator = mne.minimum_norm.make_inverse_operator(epochs_ev.info,
    #                                                           fwd, covariance,
    #                                                           fixed=True,
    #                                                           loose=0.,
    #                                                           depth=.8)
    
    # Signal to noise ratio, one should assume an SNR of 3 for averaged and 1 
    # for non-averaged data
    snr = 1.
    lambda2 = 1. / snr ** 2
    
    # Free orientation, depth weighted
    epochs_stcs = mne.minimum_norm.apply_inverse_epochs(epochs_ev,
                                                        inverse_operator,
                                                        lambda2=lambda2,
                                                        method='eLORETA',
                                                        label=None,
                                                        nave=1,
                                                        pick_ori='normal',
                                                        return_generator=True,
                                                        prepared=False,
                                                        method_params={
                                                            'force_equal':False,
                                                            'max_iter': 30},
                                                        use_cps=True,
                                                        verbose=None)
    
    # Fixed constraint, depth weighted
    # epochs_stcs = mne.minimum_norm.apply_inverse_epochs(epochs_ev,
    #                                                     inverse_operator,
    #                                                     lambda2=lambda2,
    #                                                     method='dSPM',
    #                                                     label=None,
    #                                                     nave=1,
    #                                                     pick_ori=None,
    #                                                     return_generator=False,
    #                                                     prepared=False,
    #                                                     method_params=None,
    #                                                     use_cps=True,
    #                                                     verbose=None)

    # Uncomment some sections for plotting during debugging
    # stc = []
    # for es in epochs_stcs:
    #     stc.append(es)
    # stc = np.mean(stc)
    # evk = mne.apply_forward(fwd, stc, epochs_ev.info, use_cps=True)
    # evk.plot_topomap(times=np.arange(-.25, .251, .05), colorbar=False)
    # epochs_ev.copy().average().plot_topomap(times=np.arange(-.25, .251, .05), 
    #                                         colorbar=False)
    
    # for _i, es in enumerate(epochs_stcs):
    #     evk = mne.apply_forward(fwd, es, epochs_ev.info, use_cps=True)
    #     evk.plot_topomap(times=np.arange(-.25, .251, .05), colorbar=False)
    #     epochs_ev.copy()[_i].average().plot_topomap(
    #         times=np.arange(-.25, .251, .05), colorbar=False)
        
    # brain = es.plot(hemi='split', subjects_dir='/home/ruggero.basanisi/data/tweakdreams/freesurfer', colormap='RdBu_r', initial_time=0., surface='white')
    # neg_pk_lh = es.get_peak(hemi='lh', tmin=-.05, tmax=.05, mode='neg')[0]
    # neg_pk_rh = es.get_peak(hemi='rh', tmin=-.05, tmax=.05, mode='neg')[0]
    # pos_pk_lh = es.get_peak(hemi='lh', tmin=-.05, tmax=.05, mode='pos')[0]
    # pos_pk_rh = es.get_peak(hemi='rh', tmin=-.05, tmax=.05, mode='abs')[0]
    # brain.add_foci(neg_pk_lh, hemi='lh', coords_as_verts=True, alpha=.6, color='black')
    # brain.add_foci(neg_pk_rh, hemi='rh', coords_as_verts=True, alpha=.6, color='black')
    # brain.add_foci(pos_pk_lh, hemi='lh', coords_as_verts=True, alpha=.6, color='green')
    # brain.add_foci(pos_pk_rh, hemi='rh', coords_as_verts=True, alpha=.6, color='green')
    # #brain.add_foci(abs_pk, hemi='lh', coords_as_verts=True, alpha=.6, color='white')
    # #brain.add_foci(abs_pk, hemi='rh', coords_as_verts=True, alpha=.6, color='white')
        
    # epochs_stcs = mne.minimum_norm.apply_inverse_epochs(epochs_ev,
    #                                                     inverse_operator,
    #                                                     lambda2=lambda2,
    #                                                     method='dSPM',
    #                                                     label=None,
    #                                                     nave=1,
    #                                                     pick_ori='normal',
    #                                                     return_generator=True,
    #                                                     prepared=False,
    #                                                     method_params=None,
    #                                                     use_cps=True,
    #                                                     verbose=None)

    return epochs_stcs


def compute_psd_sources(epo_fname, bln_fname, fwd_fname):
    
    # Load epochs data
    epochs = mne.read_epochs(epo_fname, preload=True)
    epochs_ev = epochs.copy()
    epochs_ev.pick_types(eeg=True)
    
    # Load baseline data
    bln_epo = mne.read_epochs(bln_fname, preload=True)
    bln_epo.pick_types(eeg=True)
    bln_epo = bln_epo.set_eeg_reference(ref_channels='average', 
                                        projection=True, ch_type='eeg')
    
    # Compute the forward model
    fwd = mne.read_forward_solution(fwd_fname)
    fwd = mne.forward.convert_forward_solution(fwd, surf_ori=True,
                                               force_fixed=False,
                                               use_cps=True)
    
    # Covariance estimated on the data
    # covariance = mne.compute_covariance(epochs_ev, keep_sample_mean=True,
    #                                     method=['shrunk', 'diagonal_fixed',
    #                                             'empirical'], cv=3, n_jobs=-1)
    # covariance = mne.cov.regularize(covariance, epochs_ev.info, eeg=.5)
    
    # Covariance estimated on the baseline
    covariance = mne.compute_covariance(bln_epo, keep_sample_mean=True,
                                        method=['shrunk', 'diagonal_fixed',
                                                'empirical'], cv=3, n_jobs=-1)
    covariance = mne.cov.regularize(covariance, bln_epo.info, eeg=.1)
    
    # Loose constraint, depth weighted
    inverse_operator = mne.minimum_norm.make_inverse_operator(epochs_ev.info,
                                                              fwd, covariance,
                                                              fixed=False,
                                                              loose=.2,
                                                              depth=.8)
    
    # Signal to noise ratio, one should assume an SNR of 3 for averaged and 1 
    # for non-averaged data
    snr = 1.
    lambda2 = 1. / snr ** 2
    
    comp_spsd = mne.minimum_norm.compute_source_psd_epochs
    epochs_spsd = comp_spsd(epochs_ev, inverse_operator, lambda2=lambda2, 
                            method='dSPM', fmin=0., fmax=40., 
                            pick_ori='normal', label=None, pca=True, 
                            inv_split=None, bandwidth=6., adaptive=True,
                            low_bias=True, return_generator=True, n_jobs=-1,
                            prepared=False, use_cps=True, verbose=False)
    
    return epochs_spsd


def labeling(subject, subjects_dir, epochs_stcs, src_fname, ltc_fname, 
             events=None):
    labels = mne.read_labels_from_annot(subject=subject, parc='aparc',
                                        hemi='both', surf_name='white',
                                        subjects_dir=subjects_dir)
    # at least 3 vertices are needed to define an area
    lone_vertices = []
    for i, l in enumerate(labels):
        if len(l.vertices) < 3:
            lone_vertices.append(i)
    if len(lone_vertices) >= 1:
        for i in sorted(lone_vertices, reverse=True):
            del labels[i]
    
    l_name = [l.name for l in labels]
    
    sources = mne.read_source_spaces(src_fname)
    
    epo_tc = []
    for ep in epochs_stcs:
        labels_tc = ep.extract_label_time_course(labels, sources,
                                                 mode='mean_flip')
        # labels_tc = ep.extract_label_time_course(labels, sources,
        #                                          mode='max')
        epo_tc.append(labels_tc)
        
    epo_tc = np.stack(tuple(epo_tc), axis=-1)
    epo_label_tc = xr.DataArray(epo_tc,
                                coords=[l_name, ep.times,
                                        range(epo_tc.shape[-1])],
                                dims=['roi', 'time', 'trials'])
    
    if events is not None:
        epo_label_tc = epo_label_tc.assign_coords(condition=('trials', events))
    
    epo_label_tc.to_netcdf(ltc_fname)
    print('Labels time course saved at ', ltc_fname)
    
    return epo_label_tc


def labeling_psd(subject, subjects_dir, epochs_spsd, src_fname, ltc_fname, 
                 events=None):
    labels = mne.read_labels_from_annot(subject=subject, parc='aparc',
                                        hemi='both', surf_name='white',
                                        subjects_dir=subjects_dir, 
                                        verbose=False)
    # at least 3 vertices are needed to define an area
    lone_vertices = []
    for i, l in enumerate(labels):
        if len(l.vertices) < 3:
            lone_vertices.append(i)
    if len(lone_vertices) >= 1:
        for i in sorted(lone_vertices, reverse=True):
            del labels[i]
    
    l_name = [l.name for l in labels]
    
    sources = mne.read_source_spaces(src_fname)
    
    epo_fq = []
    for ep in epochs_spsd:
        labels_fq = ep.extract_label_time_course(labels, sources,
                                                 mode='mean', 
                                                 verbose=False)
        # labels_fq = ep.extract_label_time_course(labels, sources,
        #                                          mode='max')
        epo_fq.append(labels_fq)
        
    epo_fq = np.stack(tuple(epo_fq), axis=-1)
    epo_label_fq = xr.DataArray(epo_fq,
                                coords=[l_name, ep.times,
                                        range(epo_fq.shape[-1])],
                                dims=['roi', 'freqs', 'trials'])
    
    if events is not None:
        epo_label_fq = epo_label_fq.assign_coords(condition=('trials', events))
    
    epo_label_fq.to_netcdf(ltc_fname)
    print('Labels power spectral density saved at ', ltc_fname)
    
    return epo_label_fq
    

if __name__ == '__main__':
    
    import os
    import os.path as op
    
    prj_data = '/home/ruggero.basanisi/data/tweakdreams'
    
    subjects = ['TD001']
    nights = ['N3']
    
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
            aw = ['aw_2']
            for _aw in aw:
                epo_fname = op.join(epo_dir, _aw, 'envelope_sw_clean-epo.fif')
                bln_fname = epo_fname
                
                stc_fname = op.join(stc_dir, _aw, 'sws_surf.stc')
                ltc_fname = op.join(ltc_dir, _aw, 'sws_labels_tc.nc')
                
                if op.exists(epo_fname):
                    # os.makedirs(op.join(stc_dir, _aw), exist_ok=True)
                    os.makedirs(op.join(ltc_dir, _aw), exist_ok=True)
                    
                    # stc = compute_lcmv_sources(epo_fname, bln_fname, 
                    #                            fwd_fname, events=None)
                    stc = compute_inverse_sources(epo_fname, bln_fname, 
                                                  fwd_fname)
                    # stc = compute_dics_sources(epo_fname, bln_fname, 
                    #                            fwd_fname)
                    # stc.save(stc_fname, format='stc', overwrite=True)
                    
                    events = mne.read_epochs(epo_fname).events[:, -1]
                    labeling(sbj, fs_dir, stc, src_fname, ltc_fname, events)
    
    

    # mne.viz.plot_alignment(info=epochs_ev.info, fwd=fwd, 
    #                        subjects_dir='/home/ruggero.basanisi/data/tweakdreams/freesurfer', 
    #                        surfaces={'outer_skin':.6, 'white':.8}, 
    #                        trans=fwd['mri_head_t'], subject='TD001',
    #                        eeg=dict(original=0.2, projected=0.8))
    
    # brain = es.plot(hemi='split', subjects_dir='/home/ruggero.basanisi/data/tweakdreams/freesurfer', colormap='RdBu_r')
    # lh_pk = es.get_peak(hemi='lh', tmin=-.05, tmax=.05, mode='neg')[0]
    # rh_pk = es.get_peak(hemi='rh', tmin=-.05, tmax=.05, mode='neg')[0]
    # brain.add_foci(lh_pk, hemi='lh', coords_as_verts=True, alpha=.6)
    # brain.add_foci(rh_pk, hemi='rh', coords_as_verts=True, alpha=.6)
    
    
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
            
    #         epochs = []
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
    #                                      fwd_fname)
    #         stc.save(stc_fname, format='stc', overwrite=True)
            
    #         # events = mne.read_epochs(epo_fname).events[:, -1]
    #         labeling(sbj, fs_dir, stc, src_fname, ltc_fname, events=None)
