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
        
     def label_tremor_events(self, tremor_events, event_id='tremor-onset'):
        """
        Labels tremor events in EEG data using start and end times.
        
        :param tremor_events: List or array of tuples representing tremor start and end times [(start1, end1), (start2, end2), ...].
        :param event_id: The label to assign to the tremor onset events (default is 'tremor-onset').
        :return: The raw EEG data with labeled events and the corresponding events array.
        """
        # Ensure tremor_events is a numpy array
        tremor_events = np.array(tremor_events)

        # Extract start and end times from the tremor events
        start_times = tremor_events[:, 0]
        end_times = tremor_events[:, 1]

        # Convert start and end times to sample indices based on the raw sampling frequency
        start_samples = (start_times * self.data.info['sfreq']).astype(int)
        end_samples = (end_times * self.data.info['sfreq']).astype(int)

        # Create events array: [onset_sample, 0, event_id]
        events = np.column_stack((start_samples, np.zeros(len(start_samples)), np.ones(len(start_samples)) * 1))

        # Create annotations for visualization
        annotations = mne.Annotations(onset=start_times, duration=end_times - start_times, description=[event_id] * len(start_times))

        # Set annotations in the raw data
        self.data.set_annotations(annotations)

        # Return the raw data with annotations and the events array
        return self.data, events

    def extract_epochs(self, event_id, tmin=-0.2, tmax=0.5):
        """
        Extract epochs from the raw EEG data.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        try:
            events = mne.find_events(self.data)
        except ValueError as e:
            raise ValueError("No events found. Please label the data first using label_tremor_events().") from e
       
        epochs = mne.Epochs(self.data, events, event_id, tmin=tmin, tmax=tmax, baseline=(None,0), preload=True)
        epochs.plot()
        plt.show()

        filename = self.file_path.split('/')[-1].replace('.fif', '-epo.fif')
        epochs.save(filename, overwrite=True)
        print(f"Epochs saved successfully as {filename}.")
        self.epochs = epochs

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