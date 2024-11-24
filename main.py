from preprocessing import AccelerometerData
from eeg_preprocessing import EEGData
import numpy as np
import argparse

# Generate sample frequency-time domain data
def generate_sample_data(num_samples=3600, freq=100):
    data = np.array([
        np.cumsum(np.random.randn(num_samples)),  # X-axis
        np.cumsum(np.random.randn(num_samples)),  # Y-axis
        np.cumsum(np.random.randn(num_samples))   # Z-axis
    ])
    # Add some periodic components to simulate accelerometer data
    t = np.linspace(0, num_samples / freq, num_samples)
    data[0] += 0.5 * np.sin(2 * np.pi * 1 * t)  # 1 Hz component on X-axis
    data[1] += 0.5 * np.sin(2 * np.pi * 2 * t)  # 2 Hz component on Y-axis
    data[2] += 0.5 * np.sin(2 * np.pi * 3 * t)  # 3 Hz component on Z-axis
    return data

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parkinson's Tremor Detection")
    parser.add_argument('--eeg', action='store_true', help='Include EEG data processing')
    args = parser.parse_args()

    if args.eeg:
        eeg_data_file_path = "data/eeg_data/801_2_PD_REST-epo.fif"
        eeg_data = EEGData(eeg_data_file_path, frequency=500)
        eeg_data.visualize_data()
        eeg_data.plot_psd()

    else:
        accelerometer_file_path = "data/accelerometer_data/809_1_accelerometer.pkl"
        accelerometer_data = AccelerometerData(accelerometer_file_path, frequency=100)
        accelerometer_data.plot_data()
        accelerometer_data.preprocess_data()
        accelerometer_data.plot_data()

'''
# Test the functions
if __name__ == "__main__":
    frequency = 100  # 100 Hz sampling frequency

    # Create an instance of AccelerometerData
    acc_data = AccelerometerData(file_path=None, frequency=frequency)

    # Generate and set sample data
    acc_data.data = generate_sample_data()

    acc_data.plot_data()

    acc_data.preprocess_data()

    acc_data.plot_data()
    # acc_data.visualize_features()
'''