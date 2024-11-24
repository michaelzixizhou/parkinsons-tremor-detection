from data_loader import DataLoader
import numpy as np
from scipy.signal import lfilter, firwin, filtfilt, savgol_filter, find_peaks
from scipy.signal.windows import hamming
from spectrum import arburg, arma2psd


class AccelerometerData(DataLoader):
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
        print("Accelerometer data:", self.data)

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
    def _bandpass_filter(self, lowcut=1.0, highcut=30.0, fs=100):
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

    # Windowing with Hamming Function
    def segment_data(self, window_size, overlap_ratio):
        """
        Segments data into overlapping windows with Hamming weights.
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

    def detect_peak_frequency(self, fs=100, low_freq=3, high_freq=8, ar_order=6):
        """
        Detects peak frequency using an autoregressive model.
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

        for i in range(3):
            for d in self.data[i]:
                # Step 1: Fit AR model to the windowed data

                try:
                    ar_coeffs, noise_variance, _ = arburg(d, ar_order)
                except Exception as e:
                    print(f"Error in AR model fitting: {e}")
                    return 1

                # Step 2: Calculate the PSD using the AR coefficients
                nfft = 1024  # Define the resolution of the FFT
                freqs = np.linspace(
                    0, fs / 2, nfft // 2 + 1
                )  # Frequency axis (up to Nyquist)
                psd = noise_variance / np.abs(np.fft.rfft(ar_coeffs, nfft)) ** 2

                # Step 3: Filter the PSD and frequencies within the desired range
                valid_indices = (freqs >= low_freq) & (freqs <= high_freq)
                filtered_freqs = freqs[valid_indices]
                filtered_psd = psd[valid_indices]

                # Step 4: Identify the peak frequency
                if filtered_psd.size > 0:
                    peak_index = np.argmax(filtered_psd)
                    peak_frequency = filtered_freqs[peak_index]
                    new_data[i].append(peak_frequency)
                else:
                    # Return 1 if no peak is found within the tremor frequency range
                    new_data[i].append(1)

        self.data = new_data
        print(self.data)

    def _smooth_data(self, window_size=50):
        self.data = savgol_filter(self.data, window_size, 3)
        # self.data = np.convolve(self.data, np.ones(window_size)/window_size, mode='same')

    def _multiply(self):
        self.data = self.data[0] * self.data[1] * self.data[2]

    def _thresholding(self, threshold=3.5):
        for i in range(self.data.shape[0]):
            self.data[i] = 1 if self.data[i] > threshold else 0

    def _feature_extraction(self, threshold=3):
        start = None
        for i in range(self.data.shape[0]):
            if self.data[i] == 1:
                if start is None:
                    start = i
            else:
                if start is not None:
                    end = i - 1
                    if end - start + 1 > threshold:
                        self.features.append((start, end, end - start + 1))
                    start = None

    def preprocess_data(self):
        print("Drift removal")
        self._remove_drift()
        print("Bandpass filter")
        self._bandpass_filter()
        print("Segment data")
        self.segment_data(300, 0.9)
        print("Detect peak frequency")
        self.detect_peak_frequency()
        ####
        print("Smooth data")
        self._smooth_data()
        print("Multiply")
        self._multiply()
        print("Thresholding")
        self._thresholding()

    def plot_data(self, t_start=0, t_end=None):
        import matplotlib.pyplot as plt

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
