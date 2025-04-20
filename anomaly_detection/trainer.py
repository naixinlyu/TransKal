import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from anomaly_detection.models.transformer import TransformerPredictor
from anomaly_detection.models.kalman_filter import KalmanFilter
from anomaly_detection.utils.visualization import plot_training_history, plot_anomaly_detection_results, calculate_metrics

class AnomalyDetector:
    """
    Neural network-based anomaly detector with Transformer model and Kalman filtering
    """
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1, Q=1e-4, R=0.1, 
                 device=None, checkpoint_dir='checkpoints'):
        """
        Initialize detector
        
        Args:
            input_dim: Input feature dimension
            d_model: Hidden dimension for transformer
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
            Q: Process noise for Kalman filter
            R: Measurement noise for Kalman filter
            device: Device to run the model on (cpu or cuda)
            checkpoint_dir: Directory to save checkpoints
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.transformer = TransformerPredictor(input_dim, d_model, nhead, num_layers, dropout)
        self.kalman_filter = KalmanFilter(1, Q, R)
        self.transformer.to(self.device)
        
        # Training related variables
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.current_epoch = 0
        
        # Checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def save_checkpoint(self, model_name, epoch, optimizer, loss, filename=None):
        """
        Save model checkpoint
        
        Args:
            model_name: Name of the model
            epoch: Current epoch
            optimizer: Optimizer state
            loss: Current loss value
            filename: Name of the checkpoint file
        """
        if filename is None:
            filename = f'{model_name}_checkpoint.pth'
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'model_name': model_name,
            'epoch': epoch,
            'model_state_dict': self.transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'epochs': self.epochs,
            'current_epoch': self.current_epoch
        }, path)
        print(f"Checkpoint saved at {path}")
        
    def load_checkpoint(self, model_name, optimizer, filename=None):
        """
        Load model checkpoint
        
        Args:
            model_name: Name of the model
            optimizer: Optimizer to load state into
            filename: Name of the checkpoint file
            
        Returns:
            epoch: Epoch of the checkpoint
            loss: Loss value of the checkpoint
        """
        if filename is None:
            filename = f'{model_name}_checkpoint.pth'
        path = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.transformer.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.epochs = checkpoint['epochs']
            self.current_epoch = checkpoint['current_epoch']
            print(f"Checkpoint for model {model_name} loaded from {path}, resuming from epoch {self.current_epoch}")
            return checkpoint['epoch'], checkpoint['loss']
        else:
            print(f"No checkpoint found for model {model_name} at {path}, starting from scratch")
            return 0, float('inf')
        
    def validate(self, val_loader):
        """
        Validate model on validation set
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            avg_val_loss: Average validation loss
        """
        self.transformer.eval()
        criterion = nn.BCEWithLogitsLoss()
        total_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.transformer(X_batch).squeeze()
                
                # Handle case when batch size is 1
                if outputs.ndim == 0:
                    outputs = outputs.unsqueeze(0)
                    
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()
                
        avg_val_loss = total_loss / len(val_loader)
        return avg_val_loss
        
    def train_model(self, model_name, train_loader, val_loader=None, num_epochs=30, lr=1e-3, 
                   resume=True, save_interval=5, plot_interval=5, verbose=True):
        """
        Train the model with checkpoint support and real-time plotting
        
        Args:
            model_name: Name of the model for saving checkpoints
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            lr: Learning rate
            resume: Whether to resume from checkpoint
            save_interval: Interval to save checkpoints
            plot_interval: Interval to plot training progress
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training history
        """
        # Record current model name
        self.model_name = model_name
        
        # Use BCE loss for anomaly detection (binary classification)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.transformer.parameters(), lr=lr)
        
        # Try to load checkpoint if resume is True
        start_epoch = 0
        best_loss = float('inf')
        if resume:
            start_epoch, best_loss = self.load_checkpoint(model_name, optimizer)
            self.current_epoch = start_epoch
        
        # Clear previous training records (if starting from scratch)
        if start_epoch == 0:
            self.train_losses = []
            self.val_losses = []
            self.epochs = []
        
        self.transformer.train()
        
        if verbose:
            print(f"Starting training for model {model_name}...")
            
        try:
            for epoch in range(start_epoch, start_epoch + num_epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                total_loss = 0.0
                
                if verbose:
                    batch_pbar = tqdm(train_loader, desc=f"Model {model_name} - Epoch {epoch+1}/{start_epoch + num_epochs}", leave=False)
                else:
                    batch_pbar = train_loader
                
                for X_batch, y_batch in batch_pbar:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.transformer(X_batch).squeeze()
                    
                    # Handle case when batch size is 1
                    if outputs.ndim == 0:
                        outputs = outputs.unsqueeze(0)
                        
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    if verbose and hasattr(batch_pbar, 'set_postfix'):
                        batch_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
                
                # Calculate average training loss
                avg_train_loss = total_loss / len(train_loader)
                
                # Calculate validation loss if validation loader is provided
                avg_val_loss = None
                if val_loader:
                    avg_val_loss = self.validate(val_loader)
                    self.val_losses.append(avg_val_loss)
                
                # Record losses and epoch
                self.train_losses.append(avg_train_loss)
                self.epochs.append(epoch + 1)
                
                # Print epoch summary
                if verbose:
                    epoch_time = time.time() - epoch_start_time
                    status = f"Model {model_name} - Epoch {epoch+1}/{start_epoch + num_epochs}, Train Loss: {avg_train_loss:.4f}"
                    if avg_val_loss:
                        status += f", Val Loss: {avg_val_loss:.4f}"
                    status += f", Time: {epoch_time:.2f}s"
                    print(status)
                
                # Visualize training progress
                if plot_interval > 0 and (epoch + 1) % plot_interval == 0:
                    plot_training_history(
                        self.epochs, 
                        self.train_losses, 
                        self.val_losses if self.val_losses else None,
                        title=f'Model {model_name} - Training and Validation Loss',
                        save_path=f'{model_name}_loss_epoch_{epoch+1}.png'
                    )
                
                # Save checkpoint
                if save_interval > 0 and (epoch + 1) % save_interval == 0:
                    self.save_checkpoint(
                        model_name, 
                        epoch + 1, 
                        optimizer, 
                        avg_train_loss, 
                        f'{model_name}_checkpoint_epoch_{epoch+1}.pth'
                    )
                
                # Save best model
                current_loss = avg_val_loss if avg_val_loss else avg_train_loss
                if current_loss < best_loss:
                    best_loss = current_loss
                    self.save_checkpoint(model_name, epoch + 1, optimizer, current_loss, f'{model_name}_best_model.pth')
                    if verbose:
                        print(f"New best model for {model_name} saved with loss: {current_loss:.4f}")
                
        except KeyboardInterrupt:
            if verbose:
                print(f"Training for model {model_name} interrupted! Saving checkpoint...")
            self.save_checkpoint(model_name, epoch + 1, optimizer, avg_train_loss, f'{model_name}_interrupted_checkpoint.pth')
        
        # Save final model
        self.save_checkpoint(model_name, start_epoch + num_epochs, optimizer, self.train_losses[-1], f'{model_name}_final_model.pth')
        if verbose:
            print(f"Training for model {model_name} complete!")
            
        # Return training history
        return {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses if val_loader else None,
            'best_loss': best_loss
        }
    
    def predict(self, X, threshold=0.5):
        """
        Make predictions using the neural network and apply Kalman filtering
        
        Args:
            X: Input data tensor or numpy array
            threshold: Threshold for binary classification
            
        Returns:
            anomalies: Binary anomaly predictions
            filtered_scores: Continuous anomaly scores after Kalman filtering
        """
        self.transformer.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            else:
                X_tensor = X.to(self.device)
            
            outputs = self.transformer(X_tensor).squeeze()
            
            # Handle case when batch size is 1
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)
                
            raw_scores = torch.sigmoid(outputs).cpu().numpy()
            
            # Apply Kalman filtering to smooth predictions
            filtered_scores = self.kalman_filter.filter(raw_scores)
            
            # Determine anomalies based on threshold
            anomalies = filtered_scores > threshold
            
            return anomalies, filtered_scores
            
    def evaluate(self, X_test, y_test, threshold=0.5, fault_start=None, title=None, save_path=None):
        """
        Evaluate model on test data
        
        Args:
            X_test: Test data
            y_test: Test labels
            threshold: Threshold for anomaly detection
            fault_start: Index where fault was introduced (optional)
            title: Title for plots (optional)
            save_path: Path to save evaluation results (optional)
            
        Returns:
            Dictionary with evaluation metrics and predictions
        """
        # Get predictions
        anomalies, scores = self.predict(X_test, threshold)
        
        # Convert tensors to numpy arrays if needed
        if isinstance(y_test, torch.Tensor):
            y_test_np = y_test.cpu().numpy()
        else:
            y_test_np = y_test
            
        # Calculate metrics
        metrics = calculate_metrics(y_test_np, anomalies)
        
        print(f"Evaluation results - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, " +
              f"F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        
        # Plot results
        if save_path is not None:
            plot_anomaly_detection_results(
                y_test_np, 
                scores, 
                anomalies, 
                fault_start=fault_start,
                threshold=threshold,
                title=title,
                save_path=save_path
            )
        
        return {
            'metrics': metrics,
            'anomalies': anomalies,
            'scores': scores
        }