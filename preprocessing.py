from data_loader import DataLoader
import numpy as np
from scipy.signal import lfilter, firwin, filtfilt


class AccelerometerData(DataLoader):
    def __init__(self, file_path, frequency):
        super().__init__(
            file_path,
            meta_data={
                "freq(Hz)": frequency,
            },
        )

    def print_data(self):
        # Access the data
        print("Accelerometer data shape:", self.data.shape)
        print("Accelerometer data:", self.data)

    # Remove drift
    def _remove_drift(self, window_size=50):
        b = np.ones(window_size) / window_size
        a = [1]
        self.data = lfilter(b, a, self.data)

    # Bandpass filter
    def _bandpass_filter(self, lowcut=1.0, highcut=30.0, fs=100):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b = firwin(numtaps=101, cutoff=[low, high], pass_zero=False)
        self.data = filtfilt(b, [1.0], self.data)

    def remove_drift(self, data, window_size=50):
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
        return np.array([lfilter(b, a, data[i]) for i in range(3)])
        pass

    def bandpass_filter(self, data, fs=100, lowcut=1.0, highcut=30.0, numtaps=101):
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
        pass

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
        pass

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

        # Use the Fast Fourier transform to get data into frequency-time domain
        # and separate data into frequency bins.

        # Find tremor times by looking for peak frequency in range 3-8.

    def preprocess_data(self):
        self._remove_drift()
        self._bandpass_filter()

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
