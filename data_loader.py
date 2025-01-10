import pickle
import mne

class DataLoader:
    def __init__(self, file_path, meta_data={}):
        self.file_path = file_path
        self.data = None
        self.meta_data = meta_data
        self.n_timesteps = None
        if file_path == None:
            self.data = None
        else:
            self._load_data()
    
    def _load_data(self):
        if self.file_path.endswith('.pkl'):
            with open(self.file_path, "rb") as f:
                self.data = pickle.load(f)
            self.n_timesteps = self.data.shape[1]
        elif self.file_path.endswith('.fif'):
            self.data = mne.io.read_raw_fif(self.file_path, preload=True)
            self.n_timesteps = self.data.n_times
        else:
            raise ValueError("Unsupported file format")

    def _get_data(self):
        return self.data

    def display_info(self):
        print("Data shape:", self.data.shape)
        print("Meta data:", self.meta_data)
