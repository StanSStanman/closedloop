import os
import os.path as op
import mne
from utils import (fname_finder, brainvision_reader, 
                   brainvision_loader, read_elc)


def brainvision_to_mne(vhdr_fnames, elc_fname, events_id, raw_dir, fif_fname,
                        divide_by='nights'):
    """Read and convert TweakDreams' brainvision raw files in mne fif files.
       The raw fif files are diveded by nights and awakenings and saved with 
       the correct references and names for the autonomic signals channels.
       A small event matrix is also saved with the raw file.

    Args:
        vhdr_fname (_type_): _description_
        elc_fname (_type_): _description_
        events_id (_type_): _description_
        raw_dir (_type_): _description_
        fif_fname (_type_): _description_
        divide_by (_type_): _description_

    Return:

    """
    if divide_by == 'nights':
        raw_by_nights(vhdr_fnames, elc_fname, events_id, raw_dir, fif_fname)
    elif divide_by == 'awakenings':
        raw_by_awakenings(vhdr_fnames, elc_fname, events_id, 
                          raw_dir, fif_fname)

    return


def brvs_pipeline(raw_data, montage):
    # Load data in memory
    brainvision_loader(raw_data)
    # Some subjects have extra BIP channels we should drop
    # ch_drop = ['BIP' + str(b) for b in list(range(4, 25))]
    # crop_raw = crop_raw.drop_channels(ch_drop, on_missing='warn')
    # Adding the reference channel
    raw_data.add_reference_channels('Z12Z')
    # Renaming and assing a type to autonomic channels
    raw_data.rename_channels(mapping={'BIP1': 'EMG',
                                      'BIP2': 'ECG',
                                      'BIP3': 'RES'})
    raw_data.set_channel_types(mapping={'EMG': 'emg',
                                        'ECG': 'ecg',
                                        'RES': 'resp'})
    # Adding the montage after defining autonomic channels
    raw_data.set_montage(montage)
    # Referencing and renaming the vertical ocular channel
    raw_data = mne.set_bipolar_reference(raw_data, 'VEOGR', 'R1Z', 
                                         ch_name='VEOG',
                                         drop_refs=False, 
                                         copy='False')
    # Deleting the old unreferenced VEOGR channel
    raw_data.drop_channels(ch_names=['VEOGR'])
    # Creating the horizontal ocular channel 
    raw_data = mne.set_bipolar_reference(raw_data, 'L1G', 'R1G', 
                                         ch_name='HEOG', 
                                         drop_refs=False, 
                                         copy='False')
    # Assing type to ocular channels
    raw_data.set_channel_types(mapping={'VEOG': 'eog',
                                        'HEOG': 'eog'})

    return raw_data


def raw_by_awakenings(vhdr_fnames, elc_fname, events_id, raw_dir, fif_fname):

    # Reading the montage
    digi_mont = read_elc(elc_fname, head_size=None)

    # Setting counter for awakenings
    awakening = 0

    for vhdr_fname in vhdr_fnames:
        # Read brainvision raw (no memory loading)
        raw = brainvision_reader(vhdr_fname)
        # Extracting salient events
        try:
            events, event_dict = mne.events_from_annotations(
                raw, event_id=events_id)
        except:
            continue


        # Setting counters for events
        start, stop = False, False
        # For loop to cut raw in chunks
        for i, ev in enumerate(events):
            if ev[-1] == 20 and not start:
                start = True
                start_tp = ev[0]
                start_idx = i
            elif ev[-1] == 40 and not stop:
                stop = True
                stop_tp = ev[0]
                stop_idx = i

            # Events checkers:
            # Avoid tmin being higher than tmax (if a s20 is lacking)
            if start and stop:
                if start_tp > stop_tp:
                    stop_tp = False
            # Avoid data loss (cut data also if a final s40 is not detected)
            if (start and not stop) and i==(len(events) - 1):
                stop = True
                stop_tp = ev[0]
                stop_idx = i

            # Once a chunk is detected, cut the raw file, load data, 
            # add montage, do some mumbojambos, and save as a mne .fif file
            if start and stop:
                # Define destination directory
                aw_raw_dir = op.join(raw_dir, 'aw_{0}'.format(awakening))
                # Create the whole paths tree
                os.makedirs(aw_raw_dir, exist_ok=True)

                # Cut the data chunk for this awakening
                tmin = raw.times[start_tp]
                tmax = raw.times[stop_tp]
                crop_raw = raw.copy().crop(tmin, tmax)

                # Adding features to data
                crop_raw = brvs_pipeline(crop_raw, digi_mont)
                
                # Saving the raw fif file
                crop_raw.save(op.join(aw_raw_dir, 
                                      '{0}-raw.fif'.format(fif_fname)),
                              split_naming='neuromag',
                              overwrite=True)

                ev_chunk = events.copy()[start_idx:stop_idx, :]
                mne.write_events(op.join(aw_raw_dir, 
                                        '{0}-eve.fif'.format(fif_fname)), 
                                        ev_chunk, overwrite=True)
                # # Rename raw files
                # rename_raws(aw_raw_dir)

                start, stop = False, False
                awakening += 1
                del crop_raw

    return


def raw_by_nights(vhdr_fnames, elc_fname, events_id, raw_dir, fif_fname):

    # Reading the montage
    digi_mont = read_elc(elc_fname, head_size=None)

    raws, eves = [], []

    for vhdr_fname in vhdr_fnames:
        # Read brainvision raw (no memory loading)
        raw = brainvision_reader(vhdr_fname)
        # Extracting salient events
        try:
            events, event_dict = mne.events_from_annotations(
                raw, event_id=events_id)
        except:
            continue
        
        # Adding features to data
        raw = brvs_pipeline(raw, digi_mont)

        raws.append(raw)
        eves.append(events)

    # Concatenate raws and events
    raw, events = mne.concatenate_raws(raws=raws, preload=True, 
                                       events_list=eves, on_mismatch='raise',
                                       verbose=False)
    # Deleting list containing raws (reduce memory usage)
    del raws
    # Deleting annotations
    # raw.annotations.delete()

    # # Adding features to data
    # raw = brvs_pipeline(raw, digi_mont)

    # Creating raw files directory
    os.makedirs(raw_dir, exist_ok=True)

    # Saving the raw fif file
    raw.save(op.join(raw_dir, '{0}-raw.fif'.format(fif_fname)),
             split_naming='neuromag', overwrite=True)
    # Saving events
    mne.write_events(op.join(raw_dir, '{0}-eve.fif'.format(fif_fname)), 
                            events, overwrite=True)
    # # Rename raw files
    # rename_raws(raw_dir)
    
    return


# def rename_raws(raw_dir):
#     import glob
#     raw_files = glob.glob(op.join(raw_dir, '*-raw.fif'))
#     for i, f in enumerate(raw_files):
#         sp = '_split-0{0}'.format(i+1)
#         nf = f.replace(sp, '').replace('-raw.fif', sp + '-raw.fif')
#         os.rename(f, nf)
#     return


def rename_raws(raw_dir, fif_fname):
    raw_fname = op.join(raw_dir, '{0}-raw.fif').format(fif_fname)
    new_fname = op.join(raw_dir, '{0}-raw-0.fif').format(fif_fname)
    os.rename(raw_fname, new_fname)
    return


if __name__ == '__main__':
    
    data_dir = '/home/ruggero.basanisi/data/tweakdreams'
    # data_dir = '/media/jerry/ruggero/tweakdreams'
    
    prj = 'TD'
    sub_n = ['001', '002', '003', '005', '006',
             '007', '008', '009', '010', '011']
    sub_n = ['005', '006',
             '007', '008', '009', '010', '011']
    sub_n = ['001']
    subjects = [prj + sn for sn in sub_n]
    nights = ['N1', 'N2', 'N3', 'N4']
    nights = ['N1']

    for sbj in subjects:
        for ngt in nights:
            _vhdr = op.join(data_dir, '{0}', '{0}_{1}', 'eeg',
                            '*.vhdr').format(sbj, ngt)
            vhdr_fnames = fname_finder(_vhdr)
            vhdr_fnames.sort()
            _elc = op.join(data_dir, '{0}', '{0}_{1}', 'eeg', 
                           '{0}_{1}*.elc').format(sbj, ngt)
            elc_fname = fname_finder(_elc)[0]
            events_id = {'Stimulus/s20': 20,
                        'Stimulus/s30': 30,
                        'Stimulus/s40': 40,
                        'Stimulus/s22': 22,
                        'Stimulus/s24': 24,
                        'Stimulus/s26': 26,
                        'Stimulus/s28': 28}
            raw_dir = op.join(data_dir, 'mne', '{0}', '{1}', 
                              'raw').format(sbj, ngt)
            fif_fname = '{0}_{1}'.format(sbj, ngt)
            
            brainvision_to_mne(vhdr_fnames, elc_fname, events_id, 
                               raw_dir, fif_fname, divide_by='nights')
            brainvision_to_mne(vhdr_fnames, elc_fname, events_id, 
                               raw_dir, fif_fname, divide_by='awakenings')
