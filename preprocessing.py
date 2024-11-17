# how to access accelerometer data for one trial
from data_loader import DataLoader
import numpy as np
from scipy.signal import lfilter, firwin, filtfilt

class AcceleromemterData(DataLoader):
    def __init__(self, file_path, frequency):
        super().__init__(file_path, meta_data={
            "freq(Hz)": frequency,
        })

    def print_data(self):
        # Access the data
        print("Accelerometer data shape:", self.data.shape)
        print("Accelerometer data:", self.data)

    # Remove drift
    def _remove_drift(self, window_size=50):
        b = np.ones(window_size)/window_size
        a = [1]
        self.data = lfilter(b, a, self.data)

    # Bandpass filter
    def _bandpass_filter(self, lowcut=1.0, highcut=30.0, fs=100):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b = firwin(numtaps=101, cutoff=[low, high], pass_zero=False)
        self.data = filtfilt(b, [1.0], self.data)

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
        plt.plot(time_range, data_range[0], 'r')
        plt.title('Accelerometer Data - X Axis')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(time_range, data_range[1], 'g')
        plt.title('Accelerometer Data - Y Axis')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(time_range, data_range[2], 'b')
        plt.title('Accelerometer Data - Z Axis')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

