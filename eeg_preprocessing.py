import mne
from mne.preprocessing import ICA
from mne.time_frequency import psd_array_welch
import matplotlib.pyplot as plt
import numpy as np

class EEGDataLoader:
    def __init__(self, file_path):
        """
        Initialize the EEGDataLoader with a path to a raw EEG file.
        :param file_path: Path to the raw EEG .fif file
        """
        self.file_path = file_path
        self.data = None
        self.psds = None
        self.freqs = None
        self.mean_psds = None
        self.scaled_data = None
        self.epochs = None
        self.load_data()

    def load_data(self):
        """
        Load raw EEG data from the provided file path.
        """
        try:
            self.data = mne.io.read_raw_fif(self.file_path, preload=True)
            print(f"Data loaded: {self.data.info}")
            self.data.set_montage(mne.channels.make_standard_montage("standard_1020"))
        except Exception as e:
            raise ValueError(f"Error loading data: {e}")

    def preprocess(self):
        """
        Apply preprocessing steps for raw EEG data.
        Includes ICA for artifact removal and PSD extraction.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Apply preprocessing steps
        self._filter_data()
        self._apply_ica()
        self._amplitude_scaling()

        # Extract PSD after preprocessing
        # This will be used for feature extraction
        self._extract_psd()
        return self.psds, self.freqs

    def _apply_ica(self, exclude=None):
        """
        Apply ICA for artifact removal on raw EEG data.
        """
        print("Applying ICA...")
        ica = ICA(n_components=20, random_state=97, max_iter=800)
        ica.fit(self.data)
        ica.plot_properties(self.data, picks=exclude)
        self.data = ica.apply(self.data)
        print("ICA applied successfully.")

    def _filter_data(self, l_freq=1, h_freq=49):
        """
        Apply bandpass filter to raw EEG data.
        :param l_freq: Lower frequency bound
        :param h_freq: Higher frequency bound
        """
        self.data.filter(l_freq=l_freq, h_freq=h_freq)
        print("Data filtered successfully")
    
    def plot_psd(self):
        '''
        Plot the power spectral denstiy (PSD)
        '''
        self.data.compute_psd().plot()
        plt.show()

    def _extract_psd(self):
        """
        Extract Power Spectral Density (PSD) for raw EEG data.
        This would compute PSD for all epochs so you should call this after extracting epochs.
        """
        print("Extracting PSD...")

        # Calculate window length as ¼ of signal length
        signal_length = self.data.get_data().shape[1]
        window_length = int(signal_length / 4)

        # Calculate overlap as ½ of window length
        overlap = int(window_length / 2)

        if self.epochs is not None:
            data = self.epochs.get_data()
        else:
            data = self.data.get_data()

        # Compute PSD using Welch's method
        psds, freqs = psd_array_welch(
            data,
            sfreq=self.data.info['sfreq'],
            fmin=1,
            fmax=49,
            n_fft=window_length,
            n_overlap=overlap
        )
        mean_psds = psds.mean(axis=0)  # Average PSD across all channels
        print(f"PSD shape: {psds.shape}")
        print(f"Frequency bins: {freqs.shape}")
        self.psds = psds
        self.freqs = freqs
        self.mean_psds = mean_psds

    def _amplitude_scaling(self, scale_factors=(0.9, 1.1)):
        """
        Apply amplitude scaling for data augmentation.
        :param scale_factors: List or tuple of scaling factors to apply.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print(f"Applying amplitude scaling with factors: {scale_factors}")
        original_data = self.data.get_data()
        scaled_datasets = [original_data * factor for factor in scale_factors]
        print(f"Generated {len(scaled_datasets)} scaled datasets for augmentation.")

        self.scaled_data = scaled_datasets

    def visualize_data(self):
        """
        Visualize the raw EEG data.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        self.data.plot(duration=5, n_channels=len(self.data.ch_names), proj=False, remove_dc=False, block=True)
        plt.show()
    
    def get_psd_features(self):
        """
        Get the PSD features from the preprocessed data.
        """
        if self.psds is None:
            raise ValueError("PSD not extracted. Call preprocess() first.")
        return self.psds, self.freqs