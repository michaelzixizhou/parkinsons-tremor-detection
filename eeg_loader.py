import os
import mne
import pywt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mne.time_frequency import psd_array_welch
from scipy.signal import find_peaks, periodogram

class EEGDataLoader:
    def __init__(self, file_path=None, dir_path=None):
        self.file_path = file_path
        self.dir_path = dir_path
        self.epochs = None
        self.psds = None
        self.freqs = None
        self.features = None # This is extracted from epochs
        self.load_data()

    def load_data(self):
        """
        Load the EEG data from the specified file path.
        If a file path is provided, it will load the data from the file.
        Otherwise, it will load the data from the directory and concatenate all 
        files into a single Epoch object. This allows you to process it all at once.
        """

        if self.file_path is None:
            for file in os.listdir(self.dir_path):
                if file.endswith(".fif"):
                    self.file_path = os.path.join(self.dir_path, file)
                    mne.concatenate_epochs([self.epochs, mne.read_epochs(self.file_path, preload=True)])
            return
        
        self.epochs = mne.read_epochs(self.file_path, preload=True)
        print("self.epochs shape:", np.shape(self.epochs))

    def get_data_raw(self, copy=False):
        """
        Get the EEG data as a 3D numpy array.
        This returns the epoch data without any feature extraction.
        Shape: (n_epochs, n_channels, n_times)
        NOTE: you need to refer to the event labels to know which class each 
        epoch belongs to. They follow the same order.
        """

        if self.epochs is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.epochs.get_data(copy=copy)
    
    def extract_features(self):
        """
        Extract features (self.features) from the EEG data (self.epochs).
        Shape after extraction: (n_epochs, n_features), n_features = 8 for this method
        If you loaded multiple files into one Epoch object, all epochs will be processed.

        This should include multiple methods for each feature.
        """

        if self.epochs is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        entropy = self.extract_entropy_feature()
        rms = self.extract_rms_feature()

        #eventually concatenate all features...
        self.features = np.concatenate((entropy, rms), axis = 1)
        pass

    def extract_rms_feature(self):
        """
        Extract RMS features from the EEG data (self.epochs).
        Shape after extraction: (n_epochs, n_features)
        """
        if self.epochs is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        data = self.get_data_raw()
        rms_features = np.sqrt(np.mean(data ** 2, axis=2))  # RMS across the time axis
        self.features = rms_features
        return rms_features
        
    def extract_entropy_feature():
        """
        Extract Shannon entropy features from EEG data
        First feature in Table 1 from https://bmcneurol.biomedcentral.com/articles/10.1186/s12883-023-03468-0/tables/1
        Shape after extraction is (n_epochs, n_features)
        """
        
        if self.epochs is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        data = self.get_data_raw()
        entropy_features = []
        for epoch in data:
            epoch_features = []
            for channel_data in epoch:
                wp = pywt.WaveletPacket(data=channel_data, wavelet = 'db1', mode='symmetric', maxlevel=3)
                nodes = wp.get_level(wp.maxlevel, 'natural')
                entropy = sum(self.shannon_entropy(node.data) for node in nodes)
                epoch_features.append(entropy)
        entropy_features.append(epoch_features)
        self.features = entropy_features
        return entropy_features
    
    def shannon_entropy(self, data):
        """
        Calculate the Shannon entropy of the given data.
        """
        p_data = np.abs(data) ** 2
        p_data = p_data / np.sum(p_data)  # Normalize data
        entropy = -np.sum(p_data * np.log2(p_data + np.finfo(float).eps))  # Add epsilon to avoid log(0)
        return entropy
    
    def peak_frequency(self):
        """
        return peak frequency based on PSD across all channels
        shape after extraction (n_epochs, n_channels)

        """

        data = self.get_data_raw() #Shape: (n_epochs, n_channels, n_times)
        self.get_psd(data) #run once and self.psds and self.freqs could be used for band_power and power_bandwodth 
        n_epochs, n_channels = np.shape(self.psds)

        peak_freqs = np.zeros(shape=(n_epochs, n_channels))

        for epoch in range(n_epochs):
            for channel in range(n_channels):
                peaks = find_peaks(self.psds[epoch, channel, :])
                # Find the index from the maximum peak
                i_max_peak = peaks[np.argmax(self.psds[epoch, channel, peaks])]
                # Find the peak freq value from that index
                peak_freq = self.freqs[epoch, channel, i_max_peak]
                peak_freqs[epoch, channel] = peak_freq

        return peak_freqs



    def band_power(self):
        """
        return Average PSD across all channels
        shape after extraction (n_epochs, n_channels)
        """
        mean_psds = self.psds.mean(axis=2)  
        return mean_psds


    def power_band_width(self):
        """
        return power band width based on periodgram 
        shape after extraction (n_epochs, n_channels)
        """

        data = self.get_data_raw()
        n_epochs, n_channels = np.shape(self.psds)
        power_band_widths = np.zeros(shape=(n_epochs, n_channels))

        for epoch in range(n_epochs):
            for channel in range(n_channels):
                psd_period = periodogram(data[epoch, channel, :], self.data.info['sfreq'])

                peaks = find_peaks(psd_period)

                #log conversion 
                psd_period = 10 * np.log(psd_period)
                # Find the index from the maximum peak
                i_max_peak = peaks[np.argmax(psd_period[peaks])]
                # obtain magnitude of periodogram at the peak 
                peak_magnitude = psd_period[i_max_peak]
                #obtain half power bandwidth
                half_dB = peak_magnitude - 3

                #get left boundary 
                left_bound = self.get_boundaries(psd_period, half_dB, i_max_peak, -1)
                right_bound = self.get_boundaries(psd_period, half_dB, i_max_peak, 1)

                power_band_width = left_bound + right_bound

                power_band_widths[epoch, channel] = power_band_width

        return power_band_widths
    
    def get_boundaries(self, data, half_dB, start_ind, direction):
        """
        determine half bandwdith index location for left and right side
        """
        count = 0
        ind = start_ind
        while ind >= 0 and ind < len(data):
            count += 1
            ind += direction
            if data[ind] < half_dB:
                return count - 1
            
        if ind < 0:
            return 0
        else: 
            return len(data) 


    def get_psd(self, data):
        """
        extract psd from epoched data 
        input: epoched data (n_epohs,n_channels, n_times)
        Shape after extraction: (n_epochs, channel, window_length) for self.psds and self.freqs
        """        

        n_epochs, n_channels, n_times = np.shape(data)

        self.psds = np.zeros(shape=(n_epochs, n_channels, n_times))
        self.freqs = np.zeros(shape=(n_epochs, n_channels, n_times))
        for epoch in range(n_epochs):
            for channel in range(n_channels):
                window_length = n_times
                overlap = window_length // 2

                psds, freqs = psd_array_welch(
                    data[epoch, channel, :],
                    sfreq=self.data.info['sfreq'],
                    fmin=1,
                    fmax=49,
                    n_fft=window_length,
                    n_overlap=overlap
                )
                self.psds[epoch, channel, :] = psds
                self.freqs[epoch, channel, :] = freqs
        
        


    
    def scale_features(self):
        """
        Apply StandardScaler to (all, by default) features.
        """
        pass

    def get_labels(self):
        """
        Get the labels (self.epochs.events) for each epoch.
        The labels can be used directly for classification.
        """

        if self.epochs is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.epochs.events[:, -1]
    
    def display_info(self):
        """
        Display basic information about the EEG data.
        """

        if self.epochs is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Display basic information about the epochs
        print("\n--- Epochs Information ---")
        print(f"Number of epochs: {len(self.epochs)}")
        print(f"Number of channels: {len(self.epochs.info['ch_names'])}")
        print(f"Channel names: {self.epochs.info['ch_names']}")
        print(f"Sampling frequency: {self.epochs.info['sfreq']} Hz")
        print(f"Epoch duration: {self.epochs.tmax - self.epochs.tmin:.2f} seconds")
        print(f"Time range: {self.epochs.tmin:.2f} to {self.epochs.tmax:.2f} seconds")
        print(f"Event labels: {self.epochs.event_id}")
        print(f"Available event labels: {self.epochs.event_id.keys()}")
        print("-----------------------------\n")

    def plot_epochs(self):
        """
        Plot the EEG epochs.
        """

        if self.epochs is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        self.epochs.plot()

    def plot_events(self):
        """
        Plot the events.
        """

        if self.epochs is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        mne.viz.plot_events(self.epochs.events, sfreq=self.epochs.info['sfreq'])

    def train_test_split(self, test_size=0.2, random_state=None):
        """
        Split the data into training and testing sets.
        Returns X_train, X_test, y_train, y_test which can be used 
        directly for training and testing in a streamlined process.
        """

        if self.features is None:
            raise ValueError("Features not extracted. Call extract_features() first")

        X = self.features
        y = self.get_labels()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
