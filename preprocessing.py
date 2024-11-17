from data_loader import DataLoader
import numpy as np
from scipy.signal import lfilter, firwin, filtfilt, savgol_filter, hamming

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

    # Windowing with Hamming Function
    def segment_data(data, window_size, overlap_ratio):
        step_size = int((1 - overlap_ratio) * window_size)
        hamming_window = hamming(window_size)
        n_segments = (len(data)-window_size)//step_size + 1
        segments = np.zeros((n_segments, window_size))
        for i in range(n_segments):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            segments[i] = data[start_idx:end_idx] * hamming_window
        return segments

    def _smooth_data(self, window_size=50):
        self.data = savgol_filter(self.data, window_size, 3)
        # self.data = np.convolve(self.data, np.ones(window_size)/window_size, mode='same')

    def _multiply(self):
        self.data = self.data[0] * self.data[1] * self.data[2]
    
    def _thresholding(self, threshold=3.5):
        for i in range(self.data.shape[0]):
            self.data[i] = 1 if self.data[i] > threshold else 0

    def _extract_features(self):
        pass

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
