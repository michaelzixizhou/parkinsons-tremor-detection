import torch
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

class TremorDataset(Dataset):
    """
    Dataset for EEG window data for Parkinson's tremor detection.
    
    Args:
        data_path (str): Path to the processed data file (.pt)
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, data_path, transform=None):
        """Initialize the dataset with data path and optional transforms."""
        self.transform = transform
        
        # Load the processed data
        data = torch.load(data_path)
        self.windows = data['windows']  # shape: [541, 41, 1536]
        self.labels = data['labels']    # shape: [541]
        
        # Print dataset information
        self._print_dataset_info()
    
    def _print_dataset_info(self):
        """Print useful information about the dataset."""
        class_counts = torch.bincount(self.labels)
        print(f"Dataset loaded with {len(self)} samples")
        print(f"Data shape: {self.windows.shape}")
        print(f"Class distribution:")
        for i, count in enumerate(class_counts):
            print(f"  Class {i}: {count} samples ({count/len(self)*100:.2f}%)")
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.windows)
    
    def __getitem__(self, idx):
        """Return a sample from the dataset."""
        window = self.windows[idx]
        label = self.labels[idx]
        
        if self.transform:
            window = self.transform(window)
            
        return window, label


def create_data_loaders(dataset, batch_size=32, train_ratio=0.7, val_ratio=0.15,
                        test_ratio=0.15, seed=42, num_workers=4):
    """
    Create train, validation, and test dataloaders from a dataset.
    
    Args:
        dataset: TremorDataset instance
        batch_size: Batch size for the dataloaders
        train_ratio: Proportion of data to use for training
        val_ratio: Proportion of data to use for validation
        test_ratio: Proportion of data to use for testing
        seed: Random seed for reproducibility
        num_workers: Number of worker processes for data loading
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Calculate split sizes
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # Split the dataset
    # Create sequential indices and split them
    indices = torch.arange(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # Create dataset subsets without shuffling
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def create_stratified_split_loaders(dataset, batch_size=32, train_ratio=0.7, val_ratio=0.15,
                                   test_ratio=0.15, seed=42, num_workers=4):
    """
    Create train, validation, and test dataloaders with stratified split 
    to maintain class distribution across splits.
    
    Args:
        dataset: TremorDataset instance
        batch_size: Batch size for the dataloaders
        train_ratio: Proportion of data to use for training
        val_ratio: Proportion of data to use for validation
        test_ratio: Proportion of data to use for testing
        seed: Random seed for reproducibility
        num_workers: Number of worker processes for data loading
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get labels
    labels = dataset.labels.numpy()
    
    # First split: train vs (val+test)
    train_idx, temp_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=labels
    )
    
    # Second split: val vs test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1-val_ratio_adjusted,
        random_state=seed,
        stratify=labels[temp_idx]
    )
    
    # Create samplers
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    
    # Create data loaders
    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=val_sampler,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler,
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"Train set: {len(train_idx)} samples")
    print(f"Validation set: {len(val_idx)} samples")
    print(f"Test set: {len(test_idx)} samples")
    
    return train_loader, val_loader, test_loader


def create_kfold_loaders(dataset, n_splits=5, batch_size=32, seed=42, num_workers=4, stratified=True):
    """
    Create k-fold cross-validation data loaders.
    
    Args:
        dataset: TremorDataset instance
        n_splits: Number of folds
        batch_size: Batch size for the dataloaders
        seed: Random seed for reproducibility
        num_workers: Number of worker processes for data loading
        stratified: Whether to use stratified k-fold
        
    Returns:
        list: List of tuples (train_loader, val_loader) for each fold
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get labels for stratification
    labels = dataset.labels.numpy()
    
    # Choose between regular or stratified k-fold
    if stratified:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = kf.split(np.arange(len(dataset)), labels)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = kf.split(np.arange(len(dataset)))
    
    # Create a list to hold all fold loaders
    fold_loaders = []
    
    # Create loaders for each fold
    for fold, (train_idx, val_idx) in enumerate(splits):
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(
            dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=True
        )
        
        val_loader = DataLoader(
            dataset, batch_size=batch_size, sampler=val_sampler,
            num_workers=num_workers, pin_memory=True
        )
        
        fold_loaders.append((train_loader, val_loader))
        print(f"Fold {fold+1}: Train={len(train_idx)} samples, Validation={len(val_idx)} samples")
        
    return fold_loaders


# Data augmentation transforms
class EEGAugmentation:
    """Base class for EEG data augmentations."""
    def __call__(self, sample):
        raise NotImplementedError


class GaussianNoise(EEGAugmentation):
    """Add Gaussian noise to EEG signal."""
    def __init__(self, std=0.1):
        self.std = std
        
    def __call__(self, sample):
        noise = torch.randn_like(sample) * self.std
        return sample + noise


class ChannelDropout(EEGAugmentation):
    """Randomly zero out channels to simulate electrode dropout."""
    def __init__(self, p=0.1, max_channels=5):
        self.p = p
        self.max_channels = max_channels
        
    def __call__(self, sample):
        if torch.rand(1) < self.p:
            num_channels = sample.shape[0]
            n_drop = torch.randint(1, min(self.max_channels+1, num_channels), (1,)).item()
            drop_idx = torch.randperm(num_channels)[:n_drop]
            
            sample_aug = sample.clone()
            sample_aug[drop_idx] = 0.0
            return sample_aug
        return sample


class TimeShift(EEGAugmentation):
    """Randomly shift the signal in time."""
    def __init__(self, max_shift_ratio=0.1):
        self.max_shift_ratio = max_shift_ratio
        
    def __call__(self, sample):
        # sample shape: [41, 1536]
        max_shift = int(sample.shape[1] * self.max_shift_ratio)
        if max_shift == 0:
            return sample
            
        shift = torch.randint(-max_shift, max_shift+1, (1,)).item()
        sample_aug = torch.zeros_like(sample)
        
        if shift > 0:
            sample_aug[:, shift:] = sample[:, :-shift]
        elif shift < 0:
            sample_aug[:, :shift] = sample[:, -shift:]
        else:
            sample_aug = sample
            
        return sample_aug
