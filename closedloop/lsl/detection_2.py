import numpy as np
import scipy as sp
import mne
import xarray as xr
from typing import Tuple
import multiprocessing
import threading
from closedloop.lsl.utils import envelope

import time


class SWCatcher:
    """
    Class to detect slow waves in the data.
    """
    
    def __init__(self, sfreq: int=500, 
                 stable_decrease_time: float=0.06, 
                 stable_increase_time: float=0.06,
                 neg_peaks_range: tuple=(-150e-6, -45e-6),
                 pos_peaks_range: tuple=(45e-6, 150e-6),
                 correlation_threshold: float=0.9,
                 distance_threshold: float=0.2):
        """
        Initialize the SW_Catcher class.
        
        Parameters
        ----------
        sfreq : int
            The sampling frequency of the data.
        """
        
        self.sfreq = sfreq
        self.stable_decrease_time = stable_decrease_time # 60ms default
        self.stable_increase_time = stable_increase_time # 60ms default
        self.neg_peaks_range = neg_peaks_range
        self.pos_peaks_range = pos_peaks_range
        self.correlation_threshold = correlation_threshold
        self.distance_threshold = distance_threshold
        
        # Calculate the number of samples for the stable decrease and increase times
        self.stable_decrease_samples = int(self.stable_decrease_time * self.sfreq)
        self.stable_increase_samples = int(self.stable_increase_time * self.sfreq)
        
        self.templates = None
        self.process = None
        self.queue = None
        self.manager = None
        self.shared_results = None
        
    
    def set_templates(self, templates: list):
        self.templates = templates
        self.num_listeners = len(self.templates)
        # self.queues = {i: multiprocessing.Queue() for i in range(self.num_listeners)}
        self.queues = multiprocessing.Queue()
        self.shared_results = {i: multiprocessing.Manager().list([False, 0, 0, 0]) for i in range(self.num_listeners)}
        self.processes = {}
        return            
    

    def detect_neg_peak(self, data: np.ndarray, q)->bool:
        """
        Detect if data has a negative peak in the specified range and if the
        signal amplitude is decreasing.

        Parameters
        ----------
        data : np.ndarray
            The data to detect peaks in.

        Returns
        -------
        bool
            True if the last peak is in the range and the signal amplitude is 
            still decreasing, False otherwise.
        """
        
        # Check on data shape
        assert data.ndim == 2, "Data should be 2D."
        assert data.shape[0] == 1, "Data should have only one channel."
        
        # Since is an envelope, we can remove the first dimension
        data  = data.squeeze()

        # Check if the last sample is in the range of the down state of a slow wave
        last_samp_amp = data[-1]
        is_in_range = self.neg_peaks_range[0] < last_samp_amp < self.neg_peaks_range[1]
        
        # Check if the signal amplitude is decreasing in the last 60ms
        samples_diff = np.diff(data)
        samples_sign = np.sign(samples_diff)
        is_decreasing = np.all(samples_sign[-self.stable_decrease_samples:] == -1)
        
        # q.put(is_in_range and is_decreasing)
        q[0] = is_in_range and is_decreasing
        
        # Return True if the last sample is in the range and 
        # the signal amplitude is still decreasing
        return is_in_range and is_decreasing


    def detect_pos_peak(self, data: np.ndarray, q)->bool:
        """
        Detect if data has a positive peak in the specified range and if the
        signal amplitude is increasing.

        Parameters
        ----------
        data : np.ndarray
            The data to detect peaks in.

        Returns
        -------
        bool
            True if the last peak is in the range and the signal amplitude is 
            still increasing, False otherwise.
        """
        
        # Check on data shape
        assert data.ndim == 2, "Data should be 2D."
        assert data.shape[0] == 1, "Data should have only one channel."
        
        # Since is an envelope, we can remove the first dimension
        data  = data.squeeze()

        # Check if the last sample is in the range of the up state of a slow wave
        last_samp_amp = data[-1]
        is_in_range = self.pos_peaks_range[0] < last_samp_amp < self.pos_peaks_range[1]
        
        # Check if the signal amplitude is increasing in the last 60ms
        samples_diff = np.diff(data)
        samples_sign = np.sign(samples_diff)
        is_increasing = np.all(samples_sign[-self.stable_increase_samples:] == 1)
        
        # q.put(is_in_range and is_increasing)
        q[0] = is_in_range and is_increasing
        
        # Return True if the last sample is in the range and
        # the signal amplitude is still increasing
        return is_in_range and is_increasing


    def detect_correlation(self, data: np.ndarray, template: np.ndarray, q)->bool:
        """
        Detect if the data has a correlation with the template above the threshold.

        Parameters
        ----------
        data : np.ndarray
            The data to detect correlation in.
        template : np.ndarray
            The template to correlate with.

        Returns
        -------
        bool
            True if the correlation is above the threshold, False otherwise.
        """
        
        # Add assertion check for data shape
        
        # Calculate mean-centered data
        data_centered = data - data.mean(axis=0, keepdims=True)
        temp_centered = template - template.mean()

        # Calculate the correlation coefficient
        num = np.sum(data_centered * temp_centered[:, np.newaxis], axis=0)
        den = np.sqrt(np.sum(data_centered**2, axis=0) * np.sum(temp_centered**2))
        corr = num / den
        
        # Check if the last correlation value is above the threshold
        is_high_corr = corr[-1] >= self.correlation_threshold
        
        # Check if the correlation is increasing in the last 60ms
        corr_diff = np.diff(corr)
        corr_sign = np.sign(corr_diff)
        is_increasing = np.all(corr_sign[-self.stable_increase_samples:] == 1)
        
        # q.put((is_high_corr and is_increasing, corr[-1]))
        q[1] = (is_high_corr and is_increasing, corr[-1])
        
        # Return True if the last correlation value is above the threshold and
        # the correlation is still increasing
        return (is_high_corr and is_increasing, corr[-1])


    def detect_distance(self, data: np.ndarray, template: np.ndarray, q)->bool:
        """
        Detect if the data has a distance with the template below the threshold.

        Parameters
        ----------
        data : np.ndarray
            The data to detect distance in.
        template : np.ndarray
            The template to calculate distance with.
        threshold : float
            The distance threshold.

        Returns
        -------
        bool
            True if the distance is below the threshold, False otherwise.
        """
        
        # Add assertion check for data shape
        
        # Calculate the distance between the data and the template
        # TODO check if the data and template are in the right shape
        dist = sp.spatial.distance.cdist(data.T, template.reshape(1, -1), 
                                        metric='cosine')
        
        # Check if the last distance value is below the threshold
        is_low_dist = dist[-1] <= self.distance_threshold
        
        # Check if the distance is decreasing in the last 60ms
        dist_diff = np.diff(dist)
        dist_sign = np.sign(dist_diff)
        is_decreasing = np.all(dist_sign[-self.stable_decrease_samples:] == -1)
        
        # q.put((is_low_dist and is_decreasing, dist[-1]))
        q[2] = (is_low_dist and is_decreasing, dist[-1])
        
        # Return True if the last distance value is below the threshold and
        # the distance is still decreasing
        return is_low_dist and is_decreasing


    def detect_phase(self, data: np.ndarray, phase='neg', q=None)->Tuple[bool, float, int]:
        """
        Detect if the data are in the ascending or descending phase of a slow wave.

        Parameters
        ----------
        data : np.ndarray
            The data to detect phase in.
        phase : str
            The type of the slow wave phase, 'neg' for negative and 'pos' for
            positive.
        sfreq : int
            The sampling frequency of the data.

        Returns
        -------
        tuple
            The phase of the slow wave, 'down' for down state and 'up' for up state.
        """
        
        # Add assertion check for data shape
        
        # Compute the Hilbert transform and the angle of the data
        hilb_data = sp.signal.hilbert(data)
        angle_data = np.angle(hilb_data)
        
        # Compute zero crossings of the angle
        angle_diff = np.diff(np.sign(angle_data))
        zero_crossings = np.where(np.logical_or(angle_diff==2, angle_diff==-2))[0]
        
        # Check if the signs of the angle from the last zero crossing to the last sample are all positive or all negative
        if phase == 'neg':
            # Angle is positive in the down state
            if np.all(np.sign(angle_data[zero_crossings[-1]:]) == 1):
                is_phase = True
            else:
                is_phase = False
        elif phase == 'pos':
            # Angle is negative in the up state
            if np.all(np.sign(angle_data[zero_crossings[-1]:]) == -1):
                is_phase = True
            else:
                is_phase = False
        
        # Compute putative slow wave frequency
        n_samp = len(angle_data) - zero_crossings[-1]
        sw_freq = (self.sfreq / (n_samp * 2))
        
        # Compute the possible number of samples to the next up-phase
        if phase == 'pos':
            next_target = n_samp * 3
        else:
            next_target = 0
            
        # q.put((is_phase, sw_freq, next_target))
        q[3] = (is_phase, sw_freq, next_target)
        
        # Return the phase of the slow wave, the slow wave frequency, and the
        # possible number of samples to the next up-phase
        return (is_phase, sw_freq, next_target)


    def set_data(self, data: np.ndarray):
        
        self.queues.put(data)
        # for i in range(self.num_listeners):
        #     self.queues[i].put(data)
        return
    
    
    def get_results(self):
        
        results = []
        for i in range(self.num_listeners):
            results.append(self.shared_results[i])
        self.detection_results = results
        return results
    

    def start_sw_detection(self, data: np.ndarray):
        """ Open a listener for each template to detect slow waves in parallel.

        Args:
            data (np.ndarray): _description_
        """
        
        for i in range(self.num_listeners):
            process = multiprocessing.Process(target=self.detection_pipeline,
                                              args=(data,))
            process.start()
            self.processes[i] = process
            print(f"Listener {i} started.")
        return
    
    
    def stop_sw_detection(self):
            
        for i in range(self.num_listeners):
            self.queues.put(None)
            if self.processes[i] and self.processes[i].is_alive():
                # self.queues[i].put(None)  # Send sentinel value to stop listener
                self.processes[i].join(timeout=0.05)  # Wait for process to finish
                print(f"Listener {i} stopped.")
        return
        
    # My listener
    def detection_pipeline(self, data: np.ndarray):

        while True:
            try:
                data = self.queues.get(block=True)  # Wait for data
                for i in range(self.num_listeners):
                    # data = self.queues[i].get(block=True)  # Wait for data
                    # print(data)
                    if data is None:  # Sentinel value to terminate
                        print(f"Listener {i} shutting down.")
                        # break
                        return
                    print(f"Listener {i} got data.")
                    self._detection_pipeline(data, 
                                             self.templates[i],
                                             self.shared_results[i],
                                             self.neg_peaks_range,
                                             self.pos_peaks_range, 
                                             self.correlation_threshold, 
                                             self.distance_threshold, 
                                             self.sfreq)
                    print(f"Listener {i} process started.")
            except Exception as e:
                print(f"Error: {e}")
                

    def _detection_pipeline(self, data: np.ndarray, template: list,
                            shared_results: list,
                            neg_peaks_range: tuple=(-150e-6, -45e-6),
                            pos_peaks_range: tuple=(45e-6, 150e-6),
                            corr_threshold: float=0.9, 
                            dist_threshold: float=0.2,
                            sfreq: int=500)->Tuple[bool, float, int]:
        """
        Run all the detections in parallel to check if a stimulation should be
        triggered.
        
        Parameters
        ---------- 
        data : np.ndarray
            The data to detect slow waves in.
        target : np.ndarray
            The target template to compare with.
        phase : str 
            The type of the slow wave phase, 'neg' for negative and 'pos' for
            positive.
        peaks_range : tuple
            The range of the peaks to detect.
        corr_threshold : float
            The correlation threshold.
        dist_threshold : float  
            The distance threshold.
        sfreq : int 
            The sampling frequency of the data.
        
        Returns
        -------
        tuple
            A tuple with the result of the detections, the slow wave frequency,
            and the possible number of samples to the next up-phase.
        """
        
        start = time.time()
        
        target, roi, phase = template 
        
        # Compute the envelope of the data
        envp = envelope(data)
        
        if phase == 'neg':
            peaks_range = neg_peaks_range
            detect_peak = self.detect_neg_peak
        elif phase == 'pos':
            peaks_range = pos_peaks_range
            detect_peak = self.detect_pos_peak
            
        algorithms = [detect_peak, 
                      self.detect_correlation, 
                      self.detect_distance, 
                      self.detect_phase]
        
        # queues = [multiprocessing.Queue() for i in range(len(algorithms))]
        # args = [(envp, queues[0]), 
        #         (data, target, queues[1]), 
        #         (data, target, queues[2]), 
        #         (envp, phase, queues[3])]
        
        queues = [None, None, None, None]
        
        args = [(envp, queues), 
                (data, target, queues), 
                (data, target, queues), 
                (envp, phase, queues)]
        
        threads = []
        for i in range(len(algorithms)):
            threads.append(threading.Thread(target=algorithms[i], 
                                                    args=args[i]))
            threads[i].start()
            print(f"Thread {i} started.")
            
        for t in threads:
            t.join()
        
        results = queues
        # results = []
        # for i in range(len(algorithms)):
        #     results.append(algorithms[i](*args[i]))
            
        # proc = []
        # for i in range(len(algorithms)):
        #     proc.append(multiprocessing.Process(target=algorithms[i], 
        #                                         args=args[i]))
        #     proc[i].start()
        #     print(f"Process {i} started.")
            
        # # Retrieve the results
        # results = []
        # for q in queues:
        #     try:
        #         results.append(q.get())
        #     except Exception as e:
        #         print(f"Error occurred: {e}")
            
        
        # # Run all the detections in parallel
        # proc = []
        # results = []
        # with multiprocessing.Pool() as pool:
        #     for alg, arg in zip(algorithms, args):
        #         proc.append(pool.apply_async(alg, arg))
                
        #     # Retrieve the results
        #     for p in proc:
        #         try:
        #             results.append(p.get())
        #         except Exception as e:
        #             print(f"Error occurred: {e}")
            
        # for p in proc:
        #     p.join()
            
        for r in results:
            print(r)
                
        # Check if all the detections are True
        if all(results[:-1]):
            result = [*results[-1], results[1][-1]]
        else:
            result = [False, results[-1][1], results[-1][2], results[1][-1]]
            
        shared_results = result
        
        print(f"Detection time: {time.time() - start}")
        return result


if __name__ == '__main__':
    
    
    # Create a random data
    data = np.random.uniform(0, 1, (64, 5000))
    
    # Create a list of templates
    templates = [[np.random.uniform(0, 1, (64)), 'roi1', 'neg'],
                 [np.random.uniform(0, 1, (64)), 'roi1', 'neg']]
    
    # Create a SWCatcher object
    sw_catcher = SWCatcher()
    
    # Set the templates
    sw_catcher.set_templates(templates)
    
    # Start the slow wave detection
    sw_catcher.start_sw_detection(data)
    
    for i in range(10000):
        # Set the data
        sw_catcher.set_data(data)
        
        # Get the results
        results = sw_catcher.get_results()
        
        data = np.random.uniform(0, 1, (64, 5000))
        time.sleep(0.02)
    
    # Stop the slow wave detection
    sw_catcher.stop_sw_detection()
    
    print(results)
    
    # Test the detection pipeline
    # sw_catcher.detection_pipeline(data)
    
    print(results)