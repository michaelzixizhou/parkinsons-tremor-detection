import os
import mne
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class EEGDataLoader:
    def __init__(self, file_path=None, dir_path=None):
        self.file_path = file_path
        self.dir_path = dir_path
        self.epochs = None
        self.features = None # This is extracted from epochs
        self.load_data()

    def load_data(self):
        """
        Load the EEG data from the specified file path.
        If a file path is provided, it will load the data from the file.
        Otherwise, it will load the data from the directory and concatenate all 
        files into a single Epoch object. This allows you to process it all at once.
        """

        if self.file_path is None:
            for file in os.listdir(self.dir_path):
                if file.endswith(".fif"):
                    self.file_path = os.path.join(self.dir_path, file)
                    mne.concatenate_epochs([self.epochs, mne.read_epochs(self.file_path, preload=True)])
            return
        
        self.epochs = mne.read_epochs(self.file_path, preload=True)

    def get_data_raw(self, copy=False):
        """
        Get the EEG data as a 3D numpy array.
        This returns the epoch data without any feature extraction.
        Shape: (n_epochs, n_channels, n_times)
        NOTE: you need to refer to the event labels to know which class each 
        epoch belongs to. They follow the same order.
        """

        if self.epochs is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.epochs.get_data(copy=copy)
    
    def extract_features(self):
        """
        Extract features (self.features) from the EEG data (self.epochs).
        Shape after extraction: (n_epochs, n_features), n_features = 8 for this method
        If you loaded multiple files into one Epoch object, all epochs will be processed.

        This should include multiple methods for each feature.
        """
        pass

    def scale_features(self):
        """
        Apply StandardScaler to (all, by default) features.
        """
        pass

    def get_labels(self):
        """
        Get the labels (self.epochs.events) for each epoch.
        The labels can be used directly for classification.
        """

        if self.epochs is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.epochs.events[:, -1]
    
    def display_info(self):
        """
        Display basic information about the EEG data.
        """

        if self.epochs is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Display basic information about the epochs
        print("\n--- Epochs Information ---")
        print(f"Number of epochs: {len(self.epochs)}")
        print(f"Number of channels: {len(self.epochs.info['ch_names'])}")
        print(f"Channel names: {self.epochs.info['ch_names']}")
        print(f"Sampling frequency: {self.epochs.info['sfreq']} Hz")
        print(f"Epoch duration: {self.epochs.tmax - self.epochs.tmin:.2f} seconds")
        print(f"Time range: {self.epochs.tmin:.2f} to {self.epochs.tmax:.2f} seconds")
        print(f"Event labels: {self.epochs.event_id}")
        print(f"Available event labels: {self.epochs.event_id.keys()}")
        print("-----------------------------\n")

    def plot_epochs(self):
        """
        Plot the EEG epochs.
        """

        if self.epochs is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        self.epochs.plot()

    def plot_events(self):
        """
        Plot the events.
        """

        if self.epochs is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        mne.viz.plot_events(self.epochs.events, sfreq=self.epochs.info['sfreq'])

    def train_test_split(self, test_size=0.2, random_state=None):
        """
        Split the data into training and testing sets.
        Returns X_train, X_test, y_train, y_test which can be used 
        directly for training and testing in a streamlined process.
        """

        if self.features is None:
            raise ValueError("Features not extracted. Call extract_features() first")

        X = self.features
        y = self.get_labels()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test