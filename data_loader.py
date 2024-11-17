import pickle

class DataLoader:
    def __init__(self, file_path, meta_data={}):
        self.file_path = file_path
        self.data = None
        self.meta_data = meta_data
        self._load_data()
    
    def _load_data(self):
        with open(self.file_path, 'rb') as f:
            self.data = pickle.load(f)
    
    def _get_data(self):
        return self.data
    
    def display_info(self):
        print("Data shape:", self.data.shape)
        print("Meta data:", self.meta_data)
    
