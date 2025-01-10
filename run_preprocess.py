from eeg_preprocessor import EEGPreprocessor
from acc_preprocessor import AccelerometerPreprocessor
import numpy as np

if __name__ == "__main__":
    ACC_DATA_PATH = "data/accelerometer_data/801_1_accelerometer.pkl"
    EEG_DATA_PATH = "data/eeg_data/801_1_PD_REST-epo.fif"

    # accelerometer_loader = AccelerometerData(ACC_DATA_PATH, frequency=100)
    # accelerometer_loader.preprocess_data()
    # accelerometer_loader.visualize_features()
    # accelerometer_loader.save_features()
    # timestamps = accelerometer_loader.features

    TIMESTAMPS_PATH = "processed/accelerometer_data/801_1_accelerometer_features.txt"
    timestamps = []
    with open(TIMESTAMPS_PATH, 'r') as file:
        for line in file:
            timestamps.append(tuple(int(x.strip()) for x in line.split(',')))

    eeg_loader = EEGPreprocessor(EEG_DATA_PATH)
    eeg_loader.preprocess()
    epochs_dict = eeg_loader.segment_with_labels(timestamps, save=True)  
    eeg_loader.plot_epochs()

    print(epochs_dict)
