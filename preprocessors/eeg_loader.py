import os
import mne
import pywt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mne.time_frequency import psd_array_welch
from scipy.signal import find_peaks, periodogram
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, moment, lmoment
from scipy.interpolate import UnivariateSpline
from mne_features.univariate import compute_samp_entropy

class EEGDataLoader:
    """
    This class can be used to access preprocessed EEG data, extract feature, and facilitate training/testing.
    Args:
        file_path (str): Path to the EEG data file (.fif).
        dir_path (str): Path to the directory containing multiple EEG data files (.fif).
        features_path (str): Path to the extracted features file (.npy).
    NOTE: If you pass in feature_path, you can only use utility functions such as train_test_split.
    """
    def __init__(self, file_path=None, dir_path=None, features_path=None):
        self.file_path = file_path
        self.dir_path = dir_path
        self.epochs = None
        self.psds = None
        self.freqs = None
        self.features = None # This is extracted from epochs
        if features_path is not None: # load direct feautres
            self.features = np.load(features_path)
        else: # load normal EEG data
            self.load_data()

    def load_data(self):
        """
        Load the EEG data from the specified file path.
        If a file path is provided, it will load the data from the file.
        Otherwise, it will load the data from the directory and concatenate all 
        files into a single Epoch object. This allows you to process it all at once.
        """

        if self.file_path is None and self.dir_path is None:
            raise ValueError("Provide either file_path or dir_path")

        if self.file_path is None:
            for file in os.listdir(self.dir_path):
                if file.endswith(".fif"):
                    file_path = os.path.join(self.dir_path, file)
                    if self.epochs is None:
                        # Initialize self.epochs with the first file
                        self.epochs = mne.read_epochs(file_path, preload=True)
                    else:
                        # Concatenate new epochs with existing ones
                        self.epochs = mne.concatenate_epochs([self.epochs, mne.read_epochs(file_path, preload=True)])
        else:
            self.epochs = mne.read_epochs(self.file_path, preload=True)
        
        print("self.epochs shape:", np.shape(self.epochs))

    def get_data_raw(self, copy=False) -> np.ndarray:
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
    
    def extract_features(self, scale=True, save=True):
        """
        Extract features (self.features) from the EEG data (self.epochs).
        Shape after extraction: (n_epochs, n_channels, n_features), n_features = 8 for this method
        If you loaded multiple files into one Epoch object, all epochs will be processed.

        - scale: Bool to indicate whether to scale the features.
        - save: Bool to indicate whether to save the features to a file.

        This should include multiple methods for each feature.
        """
        if self.epochs is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Precompute PSD once to avoid redundant calculations
        self.get_psd()
        
        # Extract all features
        entropy = self.extract_entropy_feature()
        rms = self.extract_rms_feature()
        power_band_widths = self.power_band_width()
        peak_frequencies = self.peak_frequency()
        band_power = self.band_power()
        conventional_statistics = self.conventional_statistics()
        L_moments = self.L_moments()
        FormFactor = self.FormFactor()
        SampleEntropy = self.SampleEntropy()

        # increase dimensions
        entropy = entropy[:, :, np.newaxis]
        rms = rms[:, :, np.newaxis]
        power_band_widths = power_band_widths[:, :, np.newaxis]
        peak_frequencies = peak_frequencies[:, :, np.newaxis]
        band_power = band_power[:, :, np.newaxis]
        FormFactor = FormFactor[:, :, np.newaxis]
        SampleEntropy = SampleEntropy[:, :, np.newaxis]

        # eventually concatenate all features
        self.features = np.concatenate((
            entropy, rms, power_band_widths, peak_frequencies, 
            band_power, conventional_statistics, L_moments, 
            FormFactor, SampleEntropy), axis=2)
        
        # Check for NaN or infinite values
        if np.isnan(self.features).any() or np.isinf(self.features).any():
            print("Warning: NaN or infinite values found in features. Replacing with zeros.")
            self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print("Feature shape:", np.shape(self.features))
        print("Extraction complete")

        if scale:
            self._scale_features()
            print("Scaling complete")

        if save:
            self._save_features("eeg_features.npy")
    
    def _save_features(self, file_path):
        """
        Save the extracted features to a file.
        The features should be in the shape (n_epochs, n_channels, n_features).
        """
        np.save(file_path, self.features)
        print(f"Features saved to {file_path}")

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
        
    def extract_entropy_feature(self):
        """
        Extract Shannon entropy features from EEG data
        First feature in Table 1 from https://bmcneurol.biomedcentral.com/articles/10.1186/s12883-023-03468-0/tables/1
        Shape after extraction is (n_epochs, n_features)
        """
        
        if self.epochs is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        data = self.get_data_raw()
        n_epochs, n_channels, n_times = np.shape(data)

        entropy_features = np.zeros((n_epochs, n_channels))

        for epoch_num, epoch in enumerate(data):
            for channel_num, channel_data in enumerate(epoch):
                wp = pywt.WaveletPacket(data=channel_data, wavelet = 'db1', mode='symmetric', maxlevel=3)
                nodes = wp.get_level(wp.maxlevel, 'natural')
                entropy = sum(self.shannon_entropy(node.data) for node in nodes)
                entropy_features[epoch_num, channel_num] = entropy
        self.features = entropy_features
        print(np.shape(entropy_features))
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
        Return peak frequency based on PSD across all channels.
        Shape after extraction: (n_epochs, n_channels)
        """
        # Don't recalculate PSDs since they should already be computed
        if self.psds is None or self.freqs is None:
            self.get_psd()  # Compute PSD if not already available
            
        (n_epochs, n_channels, n_fft) = np.shape(self.psds)
        peak_freqs = np.zeros(shape=(n_epochs, n_channels))

        for epoch in range(n_epochs):
            for channel in range(n_channels):
                psd_vals = np.squeeze(self.psds[epoch, channel, :])  # Extract PSD for this channel
                
                # Apply smoothing to reduce noise
                psd_smooth = np.convolve(psd_vals, np.ones(5)/5, mode='same')
                
                peaks, _ = find_peaks(psd_smooth)  # Find peaks in smoothed data

                if len(peaks) > 0:
                    # Find the index of the maximum peak
                    i_max_peak = peaks[np.argmax(psd_smooth[peaks])]
                else:
                    # Fallback: Choose the frequency bin with max power
                    i_max_peak = np.argmax(psd_smooth)  

                # Assign the corresponding frequency
                peak_freqs[epoch, channel] = self.freqs[epoch, channel, i_max_peak]

        return peak_freqs

    def band_power(self):
        """
        return Average PSD across all channels
        shape after extraction (n_epochs, n_channels)
        """
        self.get_psd()
        mean_psds = self.psds.mean(axis=2)
        return mean_psds

    def power_band_width(self):
        """
        return power band width based on periodgram 
        shape after extraction (n_epochs, n_channels)
        """
        # Don't recalculate PSDs since they should already be computed
        if self.psds is None or self.freqs is None:
            self.get_psd()
            
        (n_epochs, n_channels, n_fft) = np.shape(self.psds)
        power_band_widths = np.zeros(shape=(n_epochs, n_channels))

        for epoch in range(n_epochs):
            for channel in range(n_channels):
                psd_period = self.psds[epoch, channel,:]
                freqs = self.freqs[epoch, channel,:]
                
                # Apply smoothing to reduce noise
                psd_smooth = np.convolve(psd_period, np.ones(5)/5, mode='same')
                peaks, _ = find_peaks(psd_smooth)

                if len(peaks) == 0:
                    # No peaks found, use the median bandwidth as fallback
                    power_band_widths[epoch, channel] = 3.0  # Default bandwidth 
                    continue

                try:
                    # log conversion with safety against negative/zero values
                    psd_period = np.maximum(psd_period, 1e-10)  # Avoid log(0)
                    psd_log = 10 * np.log10(psd_period)
                    
                    # Find the index from the maximum peak
                    i_max_peak = peaks[np.argmax(psd_smooth[peaks])]

                    # obtain magnitude of periodogram at the peak
                    peak_magnitude = psd_log[i_max_peak]
                    # obtain half power bandwidth (3dB down)
                    half_dB = peak_magnitude - 3

                    # get left and right boundaries
                    left_bound = self.get_boundaries(psd_log, half_dB, i_max_peak, -1)
                    right_bound = self.get_boundaries(psd_log, half_dB, i_max_peak, 1)

                    power_band_width = left_bound + right_bound
                    
                    # Get frequency resolution
                    freqs_resolution = freqs[1] - freqs[0] 
                    
                    power_band_widths[epoch, channel] = power_band_width * freqs_resolution
                except Exception as e:
                    print(f"Warning in power_band_width: {e}")
                    power_band_widths[epoch, channel] = 3.0  # Default bandwidth

        return power_band_widths
    
    def get_boundaries(self, data, half_dB, start_ind, direction):
        """
        determine half bandwdith index location for left and right side
        """
        count = 0
        ind = start_ind
        while 0 <= ind < len(data):
            if data[ind] < half_dB:
                return count

            count += 1
            ind += direction
        
        return count

    def get_psd(self):
        """
        Extract psd from epoched data 
        input: epoched data (n_epochs,n_channels, n_times)
        Shape after extraction: (n_epochs, channel, window_length) for self.psds and self.freqs
        """        
        data = self.get_data_raw()
        n_epochs, n_channels, n_times = np.shape(data)
        
        # Initialize arrays before the loop
        # Use smaller window sizes with overlap for better results
        window_sec = 2.0  # 2 second window
        window_length = int(window_sec * 500)  # assuming 500 Hz sampling rate
        overlap = 0.5  # 50% overlap
        
        # First compute one PSD to get dimensions
        sample_psds, sample_freqs = psd_array_welch(
            data[0, 0, :],
            sfreq=500,
            fmin=1,
            fmax=49,
            n_fft=window_length,
            n_overlap=int(window_length * overlap)
        )
        
        # Initialize arrays with correct dimensions
        self.psds = np.zeros(shape=(n_epochs, n_channels, len(sample_psds)))
        self.freqs = np.zeros(shape=(n_epochs, n_channels, len(sample_freqs)))
        
        for epoch in range(n_epochs):
            for channel in range(n_channels):
                try:
                    psds, freqs = psd_array_welch(
                        data[epoch, channel, :],
                        sfreq=500,
                        fmin=1,
                        fmax=49,
                        n_fft=window_length,
                        n_overlap=int(window_length * overlap)
                    )
                    
                    self.psds[epoch, channel, :] = psds
                    self.freqs[epoch, channel, :] = freqs
                except Exception as e:
                    print(f"Warning: PSD computation failed for epoch {epoch}, channel {channel}: {e}")
                    # Use zeros for failed calculations
                    self.psds[epoch, channel, :] = np.zeros_like(sample_psds)
                    self.freqs[epoch, channel, :] = sample_freqs
        
    def conventional_statistics(self):
        """
        Returns mean value, median value, variance, kurtosis, skewness and 5th and 6th order statistics.
        input: epoched data (n_epochs,n_channels, n_times)
        Shape after extraction: (n_epochs, channel, 7)
        """ 
        data = self.get_data_raw()

        # Each would have dimensions of (n_epochs, channel)
        mean = np.mean(data, axis=2)
        variance = np.var(data, axis=2)
        median = np.median(data, axis=2)
        
        # Handle potential numerical instabilities in higher-order statistics
        try:
            kurtosis_val = kurtosis(data, axis=2)
            kurtosis_val = np.nan_to_num(kurtosis_val, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            kurtosis_val = np.zeros_like(mean)
            
        try:
            skewness_val = skew(data, axis=2)
            skewness_val = np.nan_to_num(skewness_val, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            skewness_val = np.zeros_like(mean)
        
        try:
            moment_five = moment(data, axis=2, order=5)
            moment_five = np.nan_to_num(moment_five, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            moment_five = np.zeros_like(mean)
            
        try:
            moment_six = moment(data, axis=2, order=6)
            moment_six = np.nan_to_num(moment_six, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            moment_six = np.zeros_like(mean)

        return np.stack((mean, variance, median, kurtosis_val, skewness_val, moment_five, moment_six), axis=2)

    def L_moments(self):
        """
        returns L-moments (L-scale, L-skewness, L-kurtosis)
        input: epoched data (n_epochs,n_channels, n_times)
        Shape after extraction: (n_epochs, channel, 3)
        """ 
        data = self.get_data_raw()

        #each would have dimensions of (n_epochs, channel)
        l_scale = lmoment(data, axis = 2, order = 2)
        l_skewness = lmoment(data, axis = 2, order = 3)
        l_kurtosis = lmoment(data, axis = 2, order =  4)

        return np.stack((l_scale, l_skewness, l_kurtosis), axis = 2)

    def FormFactor(self):
        """
        returns form factor (defined below)
        The ratio of the mobility of the first derivative of the signal to the mobility of the signal [36], 
        where mobility is the ratio of standard deviation for first derivative of time-series and the time-series itself
        Shape after extraction: (n_epochs, channel)
        """
        data = self.get_data_raw()
        n_epochs, n_channels, n_times = np.shape(data)
        FormFactor = np.zeros((n_epochs, n_channels))
        
        for epoch in range(n_epochs):
            for channel in range(n_channels):
                try:
                    data_epoch = data[epoch, channel, :]
                    
                    # Calculate derivatives directly (more stable than spline)
                    derivative_one = np.diff(data_epoch)
                    # Add a zero at the end to maintain length
                    derivative_one = np.append(derivative_one, 0)
                    
                    derivative_two = np.diff(derivative_one)
                    derivative_two = np.append(derivative_two, 0)
                    
                    # Calculate standard deviations with safeguards
                    std_signal = max(np.std(data_epoch), 1e-10)
                    std_d1 = max(np.std(derivative_one), 1e-10)
                    std_d2 = max(np.std(derivative_two), 1e-10)
                    
                    mobility_one = std_d1 / std_signal
                    mobility_two = std_d2 / std_d1
                    
                    FormFactor[epoch, channel] = mobility_two / mobility_one
                    
                    # Check for invalid values
                    if np.isnan(FormFactor[epoch, channel]) or np.isinf(FormFactor[epoch, channel]):
                        FormFactor[epoch, channel] = 0.0
                        
                except Exception as e:
                    print(f"Error calculating FormFactor for epoch {epoch}, channel {channel}: {e}")
                    FormFactor[epoch, channel] = 0.0

        return FormFactor

    def SampleEntropy(self):
        """
        returns Sample Entropy (defined below)
        Negative logarithm of conditional probability of the successive segmented time-series samples. 
        It is an indicator of time-series complexity
        """
        data = self.get_data_raw()
        n_epochs, n_channels, n_times = np.shape(data)
        SampleEntropy = np.zeros((n_epochs, n_channels))
        
        for epoch in range(n_epochs):
            try:
                data_epoch = data[epoch, :, :]
                samp_entropy = compute_samp_entropy(data_epoch)
                
                # Handle possible NaN or inf values
                samp_entropy = np.nan_to_num(samp_entropy, nan=0.0, posinf=0.0, neginf=0.0)
                SampleEntropy[epoch, :] = samp_entropy
            except Exception as e:
                print(f"Error calculating Sample Entropy for epoch {epoch}: {e}")
                SampleEntropy[epoch, :] = 0.0

        return SampleEntropy
    
    def _scale_features(self):
        """
        Apply StandardScaler to (all, by default) features.
        """
        scaler = StandardScaler()
        for ch in range(self.features.shape[1]):
            self.features[:, ch, :] = scaler.fit_transform(self.features[:, ch, :])

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
            self.extract_features()

        X = self.features
        y = self.get_labels()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
    
    def save_as_tensor(self, file_path):
        """
        Save the raw time series (epochs/windows) as a pytorch tensor.
        
        Args:
            file_path (str): Path to save the tensor file
            
        Shape of saved tensor:
            windows: [n_epochs, n_channels, n_times] (torch.float32)
            labels: [n_epochs] (torch.long)
        """
        if self.epochs is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        data_np = self.get_data_raw()
        labels_np = self.get_labels()

        # Convert labels from {1,2,3} to {0,1,2}
        labels_np = labels_np - 1

        # Input validation
        assert len(data_np) == len(labels_np), "Data and labels have different lengths"
        assert not np.isnan(data_np).any(), "NaN values found in data"
        assert not np.isinf(data_np).any(), "Infinite values found in data"
        
        # Ensure data is in float32 range
        if np.max(np.abs(data_np)) > 3.4e38:
            print("Warning: Data contains values outside float32 range. Scaling down...")
            data_np = data_np.astype(np.float32)

        import torch
        
        # Convert to tensors with explicit dtypes
        windows_tensor = torch.tensor(data_np, dtype=torch.float32)
        labels_tensor = torch.tensor(labels_np, dtype=torch.long)
        
        print(f"Windows tensor shape: {windows_tensor.shape}, dtype: {windows_tensor.dtype}")
        print(f"Labels tensor shape: {labels_tensor.shape}, dtype: {labels_tensor.dtype}")
        print(f"Label values: {torch.unique(labels_tensor).tolist()}")
        
        # Save tensors
        torch.save({
            "windows": windows_tensor,
            "labels": labels_tensor
        }, file_path)
        
        print(f"Data saved as tensor to {file_path}")
