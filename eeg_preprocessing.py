from data_loader import DataLoader
import numpy as np
import mne

class EEGData(DataLoader):
    def __init__(self, file_path, frequency):
        super().__init__(
            file_path,
            meta_data={
                "freq(Hz)": frequency,
            },
        )
        self.channel_names = self.data.info['ch_names']
        self.freq = self.data.info['sfreq']
    
    def visualize_data(self):
        if self.data:
            raw = self.data
            raw.plot(duration=5, n_channels=30)
        else:
            raise ValueError("No data loaded to visualize")

    def plot_psd(self):
        self.data.compute_psd(fmax=50).plot(picks="data", exclude="bads", amplitude=False)

    def print_info(self):
        print(self.data.info)

    