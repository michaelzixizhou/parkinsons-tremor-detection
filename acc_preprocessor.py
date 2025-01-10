import os
import numpy as np
from scipy.signal import lfilter, firwin, filtfilt, savgol_filter, find_peaks
from spectrum import arburg, arma2psd
import matplotlib.pyplot as plt
from data_loader import DataLoader

class AccelerometerPreprocessor(DataLoader):
    def __init__(self, file_path, frequency):
        super().__init__(
            file_path,
            meta_data={
                "freq(Hz)": frequency,
            },
        )
        self.features = []  # (start, end, duration)

    def print_data(self):
        # Access the data
        print("Accelerometer data shape:", self.data.shape)
        print("Accelerometer data (first 10):", self.data[:10])

    # Remove drift
    def _remove_drift(self, window_size=50):
        """
        Removes drift using a moving average filter.
        Args:
            data: Array of accelerometer data.
            window_size: Size of the moving average window.
        Returns:
            Drift-removed data.
        """
        b = np.ones(window_size) / window_size
        a = [1]
        self.data = lfilter(b, a, self.data)

    # Bandpass filter
    def _bandpass_filter(self, lowcut=1.0, highcut=30.0, fs=100, plot=False):
        """
        Filters the signal to only retain components in the 1-30 Hz range.
        Args:
            data: Array of accelerometer data.
            fs: Sample frequency.
            lowcut: Lower cut-off frequency.
            highcut: Upper cut-off.
        Returns:
            Bandpass filtered data.
        """
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b = firwin(numtaps=101, cutoff=[low, high], pass_zero=False)
        self.data = filtfilt(b, [1.0], self.data)
        if plot:
            self.plot_data()

    # Windowing with Hamming Function
    def _segment_data(self, window_size, overlap_ratio):
        """
        Segments data into overlapping windows with Hamming weights.
        Hamming windows taper the edges of the windows to reduce spectral leakage.
        This makes more suitable for Fourier or AR analysis.
        Takes input of shape (n_channels, n_samples) and returns (n_channels, n_windows, window_size).
        Args:
            data: Array of accelerometer data.
            window_size: Number of samples per window.
            overlap_ratio: Fraction of overlap between consecutive windows.
        Returns:
            Array of segmented windows.
        """
        step_size = int(window_size * (1 - overlap_ratio))
        n_channels, n_samples = self.data.shape
        n_windows = (n_samples - window_size) // step_size + 1
        hamming_window = np.hamming(window_size)
        channel_windowed = []

        for ch in range(n_channels):
            windowed_data = np.zeros((n_windows, window_size))
            for i in range(n_windows):
                start_idx = i * step_size
                end_idx = start_idx + window_size
                segment = self.data[ch, start_idx:end_idx]
                windowed_data[i, :] = segment * hamming_window
            channel_windowed.append(windowed_data)
        self.data = np.stack(channel_windowed, axis=0)

        print("Shape after window segmenting:", self.data.shape)
        return self.data

    def _detect_peak_frequency(self, fs=100, low_freq=3, high_freq=8, ar_order=6):
        """
        Detects peak frequency using an autoregressive model.
        For each window segment, the AR model is fitted to the data and the PSD is calculated.
        The peak frequency is then identified within the tremor range (3-8 Hz).
        The returned peak frequency is the frequency with the highest PSD amplitude.
        Args:
            window: Windowed segment of accelerometer data.
            fs: Sampling frequency.
            low_freq: Lower frequency bound for tremor detection.
            high_freq: Upper frequency bound for tremor detection.
            ar_order: Order of the autoregressive model.
        Return:
            Peak frequency if within tremor range, otherwise 1.
        """
        new_data = [[], [], []]
        # from pylab import plot, log10, legend, show
        # curr_max = 0
        for i in range(3):
            for d in self.data[i]:
                # Step 1: Fit AR model to the windowed data
                try:
                    ar_coeffs, noise_variance, _ = arburg(d, ar_order)
                except Exception as e:
                    print(f"Error in AR model fitting: {e}")
                    return 1
                
                # print(noise_variance)
                # Step 2: Calculate the PSD using the AR coefficients
                psd = arma2psd(ar_coeffs, rho = noise_variance) #* noise_variance
                psd = psd[len(psd):len(psd)//2:-1] #get only half of the graph 
                nfft = len(psd)  # PSD length matches arma2psd's output
                freqs = np.linspace(0, fs / 2, nfft)

                # Step 4: Identify the peak frequency
                
                peaks, _ = find_peaks(psd) #find peaks in the psd graph 

                if peaks.size > 0:
                    max_peak = max(psd[peaks]) #get max peak to filter out low vibrations 
                else:
                    new_data[i].append(1) #if no peak exists, append 1
                    continue
  
                # get frequency at which peaks are found
                freq_peaks = freqs[peaks]
                # filter out irrelevant frequency
                freq_indices = (freq_peaks < high_freq) & (freq_peaks > low_freq)
                # get the frequnecies that are within the low and high range
                filtered_freqs = freq_peaks[freq_indices]
            
                # there are frequencies within the range, find the corrsponding indices on freqs, else append 1
                if filtered_freqs.size > 0:
                    peak_filtered_indices = (freqs == filtered_freqs)
                else:
                    new_data[i].append(1)
                    continue
                
                freqs_filtered = freqs[peak_filtered_indices] # get frequency of interest 
                psd_max_freq_index = np.argmax(psd[peak_filtered_indices]) #get index of the max peak within the range 
                psd_filtered = psd[peak_filtered_indices] # get the psd that are within the range 

                # detect only if the amplitude is greater than maxpeak amplitude / 10
                if len(filtered_freqs) > 0 and np.any(psd_filtered[psd_max_freq_index] > max_peak / 10):
                    # append the frequecy value with the highest psd ampltiude value
                    new_data[i].append(freqs_filtered[psd_max_freq_index])
                else:
                    new_data[i].append(1)
            
            # print("Peak power:", curr_max)

        self.data = np.array(new_data)
        
    def _map_windows_to_timesteps(self, window_size, overlap_ratio):
        """
        Maps window-based dominant frequencies to the original time scale of n_timesteps.
        The dominant frequency for each window segments is upsampled to match the
        length of the original accelerometer data. This results in a time-frequency representation
        of the accelerometer data cross the three axes.

        NOTE: The window size and overlap ratio should match the values used in _segment_data().

        Args:
            window_size (int): Number of samples per window.
            overlap_ratio (float): Fraction of overlap between consecutive windows.
        Returns:
            np.ndarray: Array of shape (n_channels, n_timesteps) with mapped dominant frequencies.
        """
        step_size = int(window_size * (1 - overlap_ratio))  # Step size between windows
        n_windows = self.data.shape[1]  # Number of windows
        n_channels = self.data.shape[0]  # Number of channels

        # Initialize output array
        time_freq_map = np.zeros((n_channels, self.n_timesteps))

        for ch in range(n_channels):  # Process each channel independently
            for i in range(n_windows):
                # Calculate the range of timesteps covered by the current window
                start_idx = i * step_size
                end_idx = min(start_idx + window_size, self.n_timesteps)
                
                # Assign the dominant frequency to all timesteps in the window range
                time_freq_map[ch, start_idx:end_idx] = self.data[ch, i]

        self.data = time_freq_map

        print("Shape after mapping:", self.data.shape)

    def _smooth_data(self, window_size=50):
        self.data = savgol_filter(self.data, window_size, 3) # does not change the shape of the data
        # self.data = np.convolve(self.data, np.ones(window_size)/window_size, mode='same')

    def _multiply(self):
        self.data = np.prod(self.data, axis=0)

    def _thresholding(self, threshold=3.5):
        self.data = np.where(self.data > threshold, 1, 0)

    def _feature_extraction(self, threshold=3):
        """
        Extracts features from the data based on a threshold.

        This method scans through the data to identify segments where the data equals 1.
        It records the start and end indices of these segments and calculates their duration.
        If the duration of a segment exceeds the specified threshold (converted from time steps to seconds),
        the segment is added to the features list.

        Args:
            threshold (int, optional): The minimum duration (in seconds) for a segment to be considered a feature. Defaults to 3.

        Returns:
            np.ndarray: An array of tuples, each containing the start index, end index, and duration of a feature.
        """
        start = None
        for i in range(self.data.shape[0]):
            if self.data[i] == 1:
                if start is None:
                    start = i
            else:
                if start is not None:
                    end = i - 1
                    # If the duration of the feature is greater than the threshold
                    # Need to convert from time steps to seconds
                    if end - start + 1 > threshold * self.meta_data["freq(Hz)"]:
                        self.features.append((start, end, end - start + 1))
                    start = None
        return self.features
    
    def visualize_features(self):
        """
        Visualizes the extracted features on the accelerometer data.
        NOTE: This uses tiimestep, but it should also work with time.
        """
        time = np.arange(self.data.shape[0])

        plt.figure(figsize=(12, 6))
        plt.plot(time, self.data, label="Processed Data")

        for (start, end, duration) in self.features:
            plt.axvspan(start, end, color='red', alpha=0.3)

        plt.title("Accelerometer Data with Extracted Features")
        plt.xlabel("Timesteps")
        plt.ylabel("Label")
        plt.legend()
        plt.grid(True)
        plt.show()

    def preprocess_data(self):
        """
        Preprocesses the accelerometer data to extract features.
        Returns the extracted features as a list of tuples (start, end, duration).
        """

        print("Drift removal")
        self._remove_drift()
        
        print("Bandpass filter")
        self._bandpass_filter()
        
        print("Segment data")
        self._segment_data(300, 0.9)

        print("Detect peak frequency")
        self._detect_peak_frequency()

        print("Map dominant frequencies to time")
        self._map_windows_to_timesteps(300, 0.9)

        print("Smooth data")
        self._smooth_data()

        print("Multiply")
        self._multiply()

        print("Plot data")
        self.plot_data()

        print("Thresholding")
        self._thresholding()
        
        print("Feature extraction")
        return self._feature_extraction()

    def save_features(self, file_path=None):
        """
        Saves the extracted features to a CSV file.
        
        Args:
            file_path (str): Path to save the file. Defaults to a generated path based on `self.file_path`.
        """
        if len(self.features) == 0:
            raise ValueError("No features to save. Run feature extraction first.")

        # Default file path generation
        if file_path is None:
            base_name = os.path.splitext(os.path.basename(self.file_path))[0]
            file_path = f"processed/accelerometer_data/{base_name}_features.csv"

        try:
            # Save as .csv manually
            with open(file_path, 'w') as f:
                f.write("start,end,duration\n")
                for feature in self.features:
                    f.write(f"{feature[0]},{feature[1]},{feature[2]}\n")
            print(f"Features successfully saved to {file_path}.")
        except Exception as e:
            print(f"Error saving features: {e}")

    def plot_data(self, t_start=0, t_end=None):
        """
        Plots the accelerometer data across the three axes.
        Or plots the frequency vs time if the data is already processed.
        """

        if len(self.data.shape) >= 2: # if you plot three channels
            time = np.arange(self.data.shape[1]) / self.meta_data["freq(Hz)"]

            # Set t_end to the end of the data if not provided
            if t_end is None:
                t_end = time[-1]

            # Select the data in the specified time range
            start_idx = int(t_start * self.meta_data["freq(Hz)"])
            end_idx = int(t_end * self.meta_data["freq(Hz)"])
            time_range = time[start_idx:end_idx]
            data_range = self.data[:, start_idx:end_idx]

            plt.subplot(3, 1, 1)
            plt.plot(time_range, data_range[0], "r")
            plt.title("Accelerometer Data - X Axis")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.grid(True)

            plt.subplot(3, 1, 2)
            plt.plot(time_range, data_range[1], "g")
            plt.title("Accelerometer Data - Y Axis")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.grid(True)

            plt.subplot(3, 1, 3)
            plt.plot(time_range, data_range[2], "b")
            plt.title("Accelerometer Data - Z Axis")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.grid(True)

            plt.tight_layout()
            plt.show()

        else: # frequency vs time three channels combined
            time = np.arange(self.data.shape[0])
            plt.plot(time, self.data, "b")
            plt.title("Accelerometer Data")
            plt.xlabel("Timesteps")
            plt.ylabel("Label")
            plt.grid(True)
            plt.show()

