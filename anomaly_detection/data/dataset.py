import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class TimeSeriesDataset(Dataset):
    """
    Time series dataset for PyTorch DataLoader
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sliding_windows(X, y, lookback=15):
    """
    Create sliding window dataset: input is lookback steps, target is the next step
    
    Args:
        X: Features array
        y: Labels array
        lookback: Number of time steps to look back
        
    Returns:
        X_windows: Tensor of shape (n_samples, lookback, n_features)
        y_windows: Tensor of target values
    """
    print(f"Creating sliding windows with lookback={lookback}...")
    X_windows, y_windows = [], []
    
    for i in tqdm(range(len(X) - lookback), desc="Creating windows"):
        X_windows.append(X[i:i+lookback])
        y_windows.append(y[i+lookback])
    
    return torch.tensor(np.array(X_windows), dtype=torch.float32), torch.tensor(np.array(y_windows), dtype=torch.float32)


class DataProcessor:
    """
    Class for processing and preparing datasets for anomaly detection
    """
    def __init__(self, lookback=15, val_ratio=0.1, batch_size=16, random_state=42):
        """
        Initialize data processor
        
        Args:
            lookback: Number of time steps to look back
            val_ratio: Ratio of validation data
            batch_size: Batch size for DataLoader
            random_state: Random state for reproducibility
        """
        self.lookback = lookback
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def prepare_data(self, X_train, y_train, X_test=None, y_test=None, normalize=True):
        """
        Prepare data for training and testing
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Testing features (optional)
            y_test: Testing labels (optional)
            normalize: Whether to normalize data
            
        Returns:
            Dictionary containing training and validation data loaders, 
            and test data loader if test data is provided
        """
        # Normalize data if needed
        if normalize:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test) if X_test is not None else None
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Create sliding window datasets
        X_train_windows, y_train_windows = create_sliding_windows(X_train_scaled, y_train, self.lookback)
        
        # Create training dataset
        train_dataset = TimeSeriesDataset(X_train_windows, y_train_windows)
        
        # Split training and validation sets
        train_indices, val_indices = train_test_split(
            range(len(train_dataset)), 
            test_size=self.val_ratio, 
            random_state=self.random_state
        )
        
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            sampler=train_sampler
        )
        
        val_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=val_sampler
        )
        
        result = {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'feature_dim': X_train_windows.shape[2]
        }
        
        # Prepare test data if provided
        if X_test is not None and y_test is not None:
            X_test_windows, y_test_windows = create_sliding_windows(X_test_scaled, y_test, self.lookback)
            test_dataset = TimeSeriesDataset(X_test_windows, y_test_windows)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            
            result['test_loader'] = test_loader
            result['X_test_windows'] = X_test_windows
            result['y_test_windows'] = y_test_windows
        
        return result

    def inverse_transform(self, X_scaled):
        """
        Transform scaled data back to original scale
        
        Args:
            X_scaled: Scaled data
            
        Returns:
            X: Data in original scale
        """
        return self.scaler.inverse_transform(X_scaled)