from data_loader import DataLoader
import numpy as np
from scipy.signal import lfilter, firwin, filtfilt, savgol_filter, hamming, find_peaks
from spectrum import arburg

class AccelerometerData(DataLoader):
    def __init__(self, file_path, frequency):
        super().__init__(
            file_path,
            meta_data={
                "freq(Hz)": frequency,
            },
        )
        self.features = [] # (start, end, duration)

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
    def segment_data(self, data, window_size, overlap_ratio):
        """
        Segments data into overlapping windows with Hamming weights.
        Args:
            data: Array of accelerometer data.
            window_size: Number of samples per window.
            overlap_ratio: Fraction of overlap between consecutive windows.
        Returns:
            Array of segmented windows.
        """
        step_size = int((1 - overlap_ratio) * window_size)
        hamming_window = hamming(window_size)

        n_segments = (len(self.data)-window_size)//step_size + 1

        segments = np.zeros((n_segments, window_size))
        for i in range(n_segments):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            segments[i] = self.data[start_idx:end_idx] * hamming_window
        self.data = segments

    def detect_peak_frequency(
        self, window, fs=100, low_freq=3, high_freq=8, ar_order=6
    ):
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
        # Fit the data using Berg Method (use a library)
        [A, P, K] = arburg(window, ar_order)
        # Generate the frequency response from the AR coefficients
        freqs = np.linspace(0, fs / 2, 512)  # Frequency range up to Nyquist frequency
        _, h = np.linalg.eigh(np.polyval(A, np.exp(-1j * 2 * np.pi * freqs / fs)))
        psd = np.abs(h) ** 2
        # Normalize the power spectral density (PSD)
        psd /= np.sum(psd)
        # Find the frequency bin corresponding to the maximum power
        peaks, _ = find_peaks(psd)
        if peaks.size == 0:
            return 1  # No peaks detected
        # Convert peak indices to frequencies
        peak_freqs = freqs[peaks]
        # Filter peaks within the desired frequency range
        tremor_peaks = peak_freqs[(peak_freqs >= low_freq) & (peak_freqs <= high_freq)]
        # Return the dominant tremor frequency, if any
        if tremor_peaks.size > 0:
            return tremor_peaks[np.argmax(psd[peaks])]
        else:
            return 1  # No peak within the tremor ra

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
                        self.label.append((start, end, end - start + 1))
                    start = None

    def preprocess_data(self):
        self._smooth_data()
        self._multiply()
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
