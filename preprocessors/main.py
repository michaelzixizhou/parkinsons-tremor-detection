from preprocessors.acc_preprocessor import AccelerometerPreprocessor
from preprocessors.eeg_preprocessor import EEGPreprocessor
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parkinson's Tremor Detection")
    parser.add_argument('--eeg', action='store_true', help='Include EEG data processing')
    args = parser.parse_args()

    if args.eeg:
        eeg_data_file_path = "data/eeg_data/801_2_PD_REST-epo.fif"
        loader = EEGPreprocessor(eeg_data_file_path)
        # loader.preprocess()
        # psds, freqs = loader.extract_psd()
        # loader.visualize_data()
        loader.plot_psd()
        loader._extract_psd()
        psds, freq = loader.get_psd_features()
        print(psds.shape)

    else:
        FILE_PATH = "data/accelerometer_data/801_1_accelerometer.pkl"
        accelerometer_data = AccelerometerPreprocessor(FILE_PATH, frequency=100)
        accelerometer_data.plot_data()
        accelerometer_data.preprocess_data()
        accelerometer_data.visualize_features()
        accelerometer_data.save_features()
        