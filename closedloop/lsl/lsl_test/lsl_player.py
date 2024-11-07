import mne_lsl
import numpy as np
import time

def four_ampli_player(raw_fnames):
    
    chunk_size = 33
    n_repeat = np.inf
    names = ['EE225-000000-000625',
             'EE225-000000-000626',
             'EE225-000000-000627',
             'EE225-000000-000628']
    sources_id = ['amp1', 'amp2', 'amp3', 'amp4']
    
    ampli_1 = mne_lsl.player.PlayerLSL(fname = raw_fnames[0], 
                                       chunk_size=chunk_size,
                                       n_repeat=n_repeat,
                                       name=names[0],
                                       source_id=sources_id[0])
    ampli_2 = mne_lsl.player.PlayerLSL(fname = raw_fnames[1], 
                                       chunk_size=chunk_size,
                                       n_repeat=n_repeat,
                                       name=names[1],
                                       source_id=sources_id[1])
    ampli_3 = mne_lsl.player.PlayerLSL(fname = raw_fnames[2], 
                                       chunk_size=chunk_size,
                                       n_repeat=n_repeat,
                                       name=names[2],
                                       source_id=sources_id[2])
    ampli_4 = mne_lsl.player.PlayerLSL(fname = raw_fnames[3], 
                                       chunk_size=chunk_size,
                                       n_repeat=n_repeat,
                                       name=names[3],
                                       source_id=sources_id[3])
    
    no_exit = True
    while no_exit:
        if not ampli_1.running:
            print("Starting LSL servers, press 'ctrl + c' to exit.")
            ampli_1.start()
            ampli_2.start()
            ampli_3.start()
            ampli_4.start()
        
    return


if __name__ == '__main__':
    raw_fnames = ['/home/jerry/python_projects/space/closedloop/test_data/TweakDreams/four_ampli/amp1-raw.fif',
                  '/home/jerry/python_projects/space/closedloop/test_data/TweakDreams/four_ampli/amp2-raw.fif',
                  '/home/jerry/python_projects/space/closedloop/test_data/TweakDreams/four_ampli/amp3-raw.fif',
                  '/home/jerry/python_projects/space/closedloop/test_data/TweakDreams/four_ampli/amp4-raw.fif']
    
    four_ampli_player(raw_fnames=raw_fnames)