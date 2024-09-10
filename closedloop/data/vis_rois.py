import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as ss
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vis_utils import load_aparc, scaling
# from vis_utils import get_peaks

ss.set_context("talk")


def plot_rois(data, pvals=None, threshold=.05, time=None, contrast=.05,
              cmap='hot_r', alpha=.3, title=None, vlines=None, brain=False,
              show=False):
    
    # check that data is a 2D DataArray with the correct name of dims
    if isinstance(data, xr.DataArray):
        data_dims = data.coords._names
        assert ('roi' in data_dims and 'times' in data_dims), AssertionError(
            "DataArray must contain two dimensions with dims names "
            "'roi' and 'times'.")
    else:
        ValueError('data should be in xarray.DataArray format.')

    if data.dims == ('times', 'roi'):
        data = data.transpose('roi', 'times')

    # check if pvalues is None or a 2D DataArray with the correct dims
    if isinstance(pvals, xr.DataArray):
        pval_dims = pvals.coords._names
        assert ('roi' in pval_dims and 'times' in pval_dims), AssertionError(
            "DataArray must contain two dimensions with dims names "
            "'roi' and 'times'.")
        if pvals.dims == ('times', 'roi'):
            pvals = pvals.transpose('roi', 'times')

    else:
        assert pvals is None, ValueError('pvalues can be of type None or '
                                         'xarray.DataArray')
        
    if pvals is None:
        alpha = 1.

    # play with rois
    # standardizing names
    rois = []
    for _r in data.roi.values:
        if _r.startswith('Left-'):
            _r.replace('Left-', '')
            _r += '-lh'
        elif _r.startswith('Right-'):
            _r.replace('Right-', '')
            _r += '-rh'
        rois.append(_r)
    data['roi'] = rois

    # check if one or both hemispheres are considered
    lh_r, rh_r = [], []
    for _r in rois:
        if _r.endswith('-lh'):
            lh_r.append(_r)
        elif _r.endswith('-rh'):
            rh_r.append(_r)
        else:
            lh_r.append(_r)

    mode = 'single'
    if len(lh_r) != 0 and len(rh_r) != 0:
        mode = 'double'
        _lh = [_r.replace('-lh', '') for _r in lh_r]
        _rh = [_r.replace('-rh', '') for _r in rh_r]
        if len(lh_r) != len(rh_r):
            mode = 'bordel'
            # list of rois in lh but not in rh
            lh_uniq = list(set(lh_r) - set(rh_r))
            # list of rois in rh but not in lh
            rh_uniq = list(set(rh_r) - set(lh_r))
            # add missing right regions
            for u in lh_uniq:
                _d = xr.DataArray(np.full((1, len(data.times)), np.nan),
                                  coords=[[u.replace('-lh', '-rh')],
                                          data.times],
                                  dims=['roi', 'times'])
                data = xr.concat([data, _d], 'roi')
                if pvals is not None:
                    pvals = xr.concat([pvals, _d])
            # add missing left regions
            for u in rh_uniq:
                _d = xr.DataArray(np.full((1, len(data.times)), np.nan),
                                  coords=[[u.replace('-rh', '-lh')],
                                          data.times],
                                  dims=['roi', 'times'])
                data = xr.concat([data, _d], 'roi')
                if pvals is not None:
                    pvals = xr.concat([pvals, _d])
            # sort DataArrays by rois name
            data.sortby('roi')
            if pvals is not None:
                pvals.sortby('roi')
            # reinitialize rois lists
            _lh = [_r.replace('-lh', '') for _r in data.roi
                   if _r.endswith('-lh')]
            _rh = [_r.replace('-rh', '') for _r in data.roi
                   if _r.endswith('-rh')]

    #
    ordered_labels = load_aparc(_lh)

    # crop time window
    if time is not None:
        data = data.sel({'times': slice(time[0], time[1])})
        if pvals is not None:
            pvals = pvals.sel({'times': slice(time[0], time[1])})

    # picking data on p-values threshold
    if pvals is not None:
        pvals = pvals.fillna(1.)
        pv_data = xr.where(pvals >= threshold, np.nan, data)

    # get colorbar limits
    if isinstance(contrast, float):
        vmin = data.quantile(contrast, skipna=True).values
        vmax = data.quantile(1 - contrast, skipna=True).values
    elif isinstance(contrast, (tuple, list)) and (len(contrast) == 2):
        vmin, vmax = contrast
    else:
        vmin, vmax = data.min(skipna=True), data.max(skipna=True)
    kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)

    # plot specs
    if vlines is None:
        vlines = {0.: dict(color='k', linewidth=1)}
    title = '' if not isinstance(title, str) else title

    times = data.times.values.round(5)
    tp = np.hstack((np.flip(np.arange(0, times.min(), -.2)),
                    np.arange(0, times.max(), .2)))
    tp = np.unique(tp.round(3))
    time_ticks = np.where(np.isin(times, tp))[0]

    # design plots
    if mode == 'single':
        h, w = len(data.roi), 9
        if brain is True:
            fig, [lbr, lh] = plt.subplots(2, 1, figsize=(w, scaling(h)),
                                          gridspec_kw={'height_ratios':
                                                       [scaling(h), h]})
            # TODO vep plot of right hemisphere
            # TODO put a small colorbar aside
            # ma_brain = plot_vep_brain(data, ax=lbr)

            lh.pcolormesh(data.times, data.roi, data, rasterized=True)
        else:
            fig, lh = plt.subplots(1, 1, figsize=(w, scaling(h)))

    elif mode == 'double' or mode == 'bordel':
        h, w = len(data.roi), 14
        if brain is True:
            fig, [lbr, lh, rbr, rh] = \
                plt.subplots(2, 2, figsize=(w, scaling(h)), gridspec_kw={
                    'height_ratios': [scaling(h), h, scaling(h), h]},
                             sharey=True)

            # TODO vep plot of right hemisphere
            # TODO put a small colorbar aside
            # ma_brain = plot_vep_brain(data, ax=rbr)

        else:
            fig, [lh, rh] = plt.subplots(1, 2, figsize=(w, scaling(h)))
            # fig, [lh, rh] = plt.subplots(1, 2, figsize=(14, 20))

        _data = data.sel({'roi': lh_r})
        _data['roi'] = _lh
        _data = _data.sel({'roi': ordered_labels['roi']})
        _data['roi'] = ordered_labels['label']
        
        if pvals is not None:
            _pv_data = pv_data.sel({'roi': lh_r})
            _pv_data['roi'] = _lh
            _pv_data = _pv_data.sel({'roi': ordered_labels['roi']})
            _pv_data['roi'] = ordered_labels['label']

        if mode == 'single':
            ss.heatmap(_data.to_pandas(), yticklabels=True, xticklabels=False,
                       vmin=vmin, vmax=vmax, cmap=cmap, ax=lh,
                       zorder=0, rasterized=True, alpha=alpha)
            
            if pvals is not None:
                ss.heatmap(_pv_data.to_pandas(), yticklabels=True, 
                           xticklabels=False, vmin=vmin, vmax=vmax, cmap=cmap, 
                           ax=lh, zorder=0, rasterized=True)

            for k, kw in vlines.items():
                _k = np.where(data.times.values == k)[0][0]
                lh.axvline(_k, **kw)

            lh.set_xticks(time_ticks)
            lh.set_xticklabels(tp, rotation='horizontal')
            lh.tick_params(axis='y', which='major', labelsize=10)
            lh.tick_params(axis='y', which='minor', labelsize=10)
            lh.yaxis.set_label_text('')
            plt.tight_layout()

        elif mode == 'double' or mode == 'bordel':
            ss.heatmap(_data.to_pandas(), yticklabels=True, xticklabels=False,
                       vmin=vmin, vmax=vmax, cmap=cmap, ax=lh,
                       cbar=False, zorder=0, rasterized=True, alpha=alpha)
            
            if pvals is not None:
                ss.heatmap(_pv_data.to_pandas(), yticklabels=True, 
                           xticklabels=False, vmin=vmin, vmax=vmax, cmap=cmap, 
                           ax=lh, cbar=False, zorder=0, rasterized=True)

            for k, kw in vlines.items():
                _k = np.where(data.times.values == k)[0][0]
                lh.axvline(_k, **kw)

            lh.set_xticks(time_ticks)
            lh.set_xticklabels(tp, rotation='horizontal')

            ylabs = [item.get_text() for item in lh.get_yticklabels()]
            lh.set_yticklabels(['' for yl in ylabs])
            lh.tick_params(axis='y', bottom=True, top=False, left=False,
                           right=True, direction="out", length=3, width=1.5)
            lh.yaxis.set_label_text('')

            _data = data.sel({'roi': rh_r})
            _data['roi'] = _rh
            _data = _data.sel({'roi': ordered_labels['roi']})
            _data['roi'] = ordered_labels['label']

            ss.heatmap(_data.to_pandas(), yticklabels=True, xticklabels=False,
                       vmin=vmin, vmax=vmax, cmap=cmap, ax=rh,
                       cbar=False, zorder=0, rasterized=True, alpha=alpha)
            
            if pvals is not None: # new add to check
                _pv_data = pv_data.sel({'roi': rh_r})
                _pv_data['roi'] = _rh
                _pv_data = _pv_data.sel({'roi': ordered_labels['roi']})
                _pv_data['roi'] = ordered_labels['label']
                
                ss.heatmap(_pv_data.to_pandas(), yticklabels=True, 
                           xticklabels=False, vmin=vmin, vmax=vmax, cmap=cmap, 
                           ax=rh, cbar=False, zorder=0, rasterized=True)

            for k, kw in vlines.items():
                _k = np.where(data.times.values == k)[0][0]
                rh.axvline(_k, **kw)

            rh.set_xticks(time_ticks)
            rh.set_xticklabels(tp, rotation='horizontal')
            rh.set_yticklabels(_data.roi.values, ha='center',
                               position=(-.23, -.50), weight='bold')
            rh.tick_params(axis='y', which='major', labelsize=9)
            rh.tick_params(axis='y', which='minor', labelsize=9)
            rh.tick_params(axis='y', bottom=True, top=False, left=True,
                           right=False, direction="out", length=3, width=1)
            rh.yaxis.set_label_text('')

            for ytl, col in zip(rh.get_yticklabels(), ordered_labels['color']):
                ytl.set_color(col)

            cbar = fig.add_axes([.3, .05, .4, .015])
            norm = mpl.colors.Normalize(vmin=kwargs['vmin'],
                                        vmax=kwargs['vmax'])
            cb_cmap = mpl.colormaps.get_cmap(kwargs['cmap'])
            mpl.colorbar.ColorbarBase(cbar, cmap=cb_cmap, norm=norm,
                                      orientation='horizontal')
            cbar.tick_params(labelsize=10)

            fig.tight_layout()
            fig.subplots_adjust(bottom=0.1)

    # plt.text(-45, -60, title)
    plt.figtext(0.06, 0.04, title, ha="center", fontsize=16,
                bbox={"facecolor": "white", "alpha": 0.5, "pad": 5}, 
                weight='bold')
    if show:
        plt.show(block=True)

    return fig


def scatter_rois(data, threshold=.05, show=False):
    assert isinstance(data, xr.DataArray), ValueError('data should be in'
                                                      'DataArray format')
    data_dims = data.coords._names
    assert ('roi' in data_dims), AssertionError('DataArray must contain '
                                                'a roi dimension')
    
    rois = []
    for _r in data.roi.values:
        if _r.startswith('Left-'):
            _r.replace('Left-', '')
            _r += '-lh'
        elif _r.startswith('Right-'):
            _r.replace('Right-', '')
            _r += '-rh'
        rois.append(_r)
    data['roi'] = rois

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))

    hemi = ['lh', 'rh']
    for h in hemi:
        h_lab = [l for l in data.roi.values if l.endswith(h)]
        abs_lab = [l.replace('-{0}'.format(h), '') for l in h_lab]

        _data = data.sel({'roi': h_lab})
        _data['roi'] = abs_lab

        ordered_labels = load_aparc(abs_lab)
        
        _data = _data.sel({'roi': ordered_labels['roi']})
        _data['roi'] = ordered_labels['label']

        _data = _data.squeeze()
        
        # for dim in data_dims:
        #     if dim != 'roi':
        #         _data = _data.mean(dim)

        if h == 'lh':
            c = 'crimson'
        elif h == 'rh':
            c = 'navy'
        
        ax.scatter(range(len(_data.roi)), _data.values, c=c, 
                   alpha=.6, label='{0} hemi'.format(h))

    ax.axhline(threshold, 0, len(_data.roi), ls='--', lw=.6, c='k')
    ax.set_xticks(range(len(_data.roi)))
    ax.set_xticklabels(ordered_labels['label'], rotation='vertical', 
                       ha='center')
    ax.tick_params(axis='x', which='major', labelsize=9)
    ax.tick_params(axis='x', which='minor', labelsize=9)

    plt.legend()
    plt.tight_layout()
    if show:
        plt.show(block=True)

    return fig


def descriptive_violin(data):
    # same x ranges for the plot
    rmin, rmax = data.min() - .5, data.max() + .5
    # loop over left and right
    figs = []
    hemi = ['lh', 'rh']
    for h in hemi:
        h_lab = [l for l in data.roi.values if l.endswith(h)]
        abs_lab = [l.replace('-{0}'.format(h), '') for l in h_lab]
        
        _data = data.sel({'roi': h_lab})
        # _data = _data.sel({'times': slice(-.15, .15)})
        _data['roi'] = abs_lab
        
        ordered_labels = load_aparc(abs_lab)
        
        _data = _data.sel({'roi': ordered_labels['roi']})
        _data['roi'] = ordered_labels['label']
        
        colors = ordered_labels['color']
        
        conditions = [1, 2, 3]  # corresponding to negative, neutral, positive
    
        # fig = go.Figure()
        fig = make_subplots(rows=1, cols=3, 
                            subplot_titles=['negative', 'neutral', 'positive'])
        if h == 'lh':
            plot_title = 'Left hemisphere'
        elif h == 'rh':
            plot_title = 'Right hemisphere'
        
        for cond in conditions:
            _d = _data.sel({'trials': _data.condition == cond})#.mean('times')
            for _dl, color in zip(_d, colors):
                
                if cond == 3:
                    sl = True
                else:
                    sl = False
                
                fig.add_trace(go.Violin(x=_dl, line_color=color,
                                        box_visible=True,
                                        name=str(_dl.roi.values),
                                        legendgroup=str(_dl.roi.values),
                                        showlegend=sl),
                              row=1, col=cond)
            fig.update_traces(orientation='h', side='negative', width=3,
                              points=False)
            fig.update_layout(title_text=plot_title,
                              yaxis={'showticklabels': False,
                                     'autorange': 'reversed'},
                              yaxis2={'showticklabels': False,
                                      'autorange': 'reversed'},
                              yaxis3={'showticklabels': False,
                                      'autorange': 'reversed'},
                              legend_tracegroupgap=4)
            fig.update_xaxes(range=[rmin, rmax])
        figs.append(fig)
        # fig.show()
    return figs


if __name__ == '__main__':

    import os
    import os.path as op
    import time
    
    prj_data = '/home/ruggero.basanisi/data/tweakdreams'
    
    subjects = ['TD001']
    nights = ['N1']
    
    for sbj in subjects:
        for n in nights:
            
            ltc_dir = op.join(prj_data, 'mne', sbj, n, 'ltc')
            
            aw = [a for a in os.listdir(ltc_dir) if a.startswith('aw_')]
            for _aw in aw:
                
                ltc_fname = op.join(ltc_dir, _aw, 'sws_labels_tc.nc')
                
                print('\nPlotting:', ltc_fname, '\n')
                if op.exists(ltc_fname):
                    data = xr.load_dataarray(ltc_fname)
                    data = data.rename({'time': 'times'})
                    plot_rois(data.mean('trials'), cmap='viridis', 
                              vlines={0.: dict(color='r', linewidth=1.5)}, 
                              show=True)
                    a=0
                    
                    
                    # for t in data.trials:
                    #     plot_rois(data.sel({'trials': t}), 
                    #               cmap='viridis', 
                    #               vlines={0.: dict(color='r', linewidth=1.5)}, 
                    #               show=False)
                    #     plt.show(block=True)
                    # #     # time.sleep(5)
                    # #     # plt.close()
                        
                    
    
    # from emosleep.amplitude import compute_amplitude

    ### TESTING PURPOSES ###
    # data_fname = '/media/jerry/ruggero/EmoSleep/mne/ltc/label_tc.nc'
    
    # data = xr.load_dataarray(data_fname)
    # negative = data.sel({'trials': data.condition == 1}).mean('trials')
    # neutral = data.sel({'trials': data.condition == 2}).mean('trials')
    # positive = data.sel({'trials': data.condition == 3}).mean('trials')
    
    
    # # data = data.mean('time')
    # data = compute_amplitude(data, fmin=.5, fmax=5.)
    # # data = compute_amplitude(data, fmin=5., fmax=12.)
    # data = data.max('freq')
    # descriptive_violin(data)  # (rois, trials) plus conditions
    
    # data = data.rename({'time': 'times'})
    # data = data.mean('trials')
    # plot_rois(data, pvals=None, cmap='RdBu_r', show=True)
    # plot_rois(positive - neutral, cmap='Reds')
    # plot_rois(negative - neutral, cmap='Blues_r')
    # plot_rois(positive - negative, cmap='RdBu_r')
    # plot_rois(np.sqrt(positive**2 + negative**2), cmap='RdBu_r')
    #######################

    # datapath = '/Disk2/EmoSleep/derivatives/'
    # subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 
    #             'sub-06', 'sub-07', 'sub-10', 'sub-11', 'sub-12', 
    #             'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-19', 
    #             'sub-22', 'sub-23', 'sub-24', 'sub-25', 'sub-26', 
    #             'sub-27', 'sub-28', 'sub-29', 'sub-30', 'sub-32']
    # # subjects = ['sub-07']
    # ses = '01'

    # dest_dir = '/Disk2/EmoSleep/derivatives/results/labels_time_course/070923/dSPM'

    # ltc_fname = op.join(datapath, '{0}', 'mne', 'ltc', '{0}_ses-{1}_900_ltc.nc')
    # #######################

    # all_sbjs = []
    # for sbj in subjects:
    #     data_fname = ltc_fname.format(sbj, ses)
    #     data = xr.load_dataarray(data_fname)
    #     data = data.rename({'time': 'times'})
    #     all_sbjs.append(data)
    # all_sbjs = xr.concat(all_sbjs, 'trials')
    # all_sbjs = all_sbjs.mean('trials')
    # fig = plot_rois(all_sbjs, title='subs avg', cmap='RdBu_r', show=True)
    # # plt.savefig(op.join(dest_dir, 'subjects_average_all_trials'), format='png')

    # # for sbj in subjects:
    # #     data_fname = ltc_fname.format(sbj, ses)
    # #     data = xr.load_dataarray(data_fname)
    # #     data = data.rename({'time': 'times'})
    # #     data = data.mean('trials')
    # #     fig = plot_rois(data, title=sbj, cmap='RdBu_r', show=True)
    # #     # plt.savefig(op.join(dest_dir, '{0}_all_trials'.format(sbj)), format='png')

    # conditions = {'neg_trials': 1, 'neu_trials': 2, 'pos_trials': 3}
    # # conditions = {'no_kc': 0, 'kc': 1}
    # for c in conditions.keys():
    #     all_sbjs = []
    #     for sbj in subjects:
    #         data_fname = ltc_fname.format(sbj, ses)
    #         data = xr.load_dataarray(data_fname)
    #         data = data.rename({'time': 'times'})
    #         data = data.sel({'trials': data.condition == conditions[c]})
    #         all_sbjs.append(data)
    #     all_sbjs = xr.concat(all_sbjs, 'trials')
    #     all_sbjs = all_sbjs.mean('trials')
    #     fig = plot_rois(all_sbjs, title='subs avg', cmap='RdBu_r', show=True)
    #     # plt.savefig(op.join(dest_dir, 'subjects_average_{0}'.format(c)), format='png')

        # for sbj in subjects:
        #     data_fname = ltc_fname.format(sbj, ses)
        #     data = xr.load_dataarray(data_fname)
        #     data = data.rename({'time': 'times'})
        #     data = data.sel({'trials': data.condition == conditions[c]})
        #     data = data.mean('trials')
        #     fig = plot_rois(data, title=sbj, cmap='RdBu_r', show=True)
        #     # plt.savefig(op.join(dest_dir, '{0}_{1}'.format(sbj, c)), format='png')
    #######################

    # all_sbjs = []
    # for sbj in subjects:
    #     data_fname = ltc_fname.format(sbj, ses)
    #     data = xr.load_dataarray(data_fname)
    #     data = data.rename({'time': 'times'})
    #     data = data.sel({'times': slice(-.25, .25)})
    #     all_sbjs.append(data)
    # all_sbjs = xr.concat(all_sbjs, 'trials')
    # all_sbjs = get_peaks(all_sbjs, 'times')
    # # all_sbjs = abs(all_sbjs).max('times')
    # figs = descriptive_violin(all_sbjs)
    # # plt.savefig(op.join(dest_dir, 'subjects_average_all_trials'), format='png')
    # figs[0].write_html(op.join(dest_dir, 'subjects_violins_all_trials_max_lh'))
    # figs[1].write_html(op.join(dest_dir, 'subjects_violins_all_trials_max_rh'))

    
