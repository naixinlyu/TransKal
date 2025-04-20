import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def plot_training_history(epochs, train_losses, val_losses=None, title=None, save_path=None):
    """
    Plot training and validation loss history
    
    Args:
        epochs: List of epoch numbers
        train_losses: List of training losses
        val_losses: List of validation losses (optional)
        title: Plot title (optional)
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    if title:
        plt.title(title)
    else:
        plt.title('Training and Validation Loss')
        
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.close()

def plot_anomaly_detection_results(y_true, scores, anomalies, fault_start=None, threshold=0.5, title=None, save_path=None):
    """
    Plot anomaly detection results
    
    Args:
        y_true: True labels
        scores: Anomaly scores
        anomalies: Detected anomalies
        fault_start: Index where fault was introduced (optional)
        threshold: Threshold for anomaly detection
        title: Plot title (optional)
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 8))
    
    # Top plot: anomaly scores and detected anomalies
    plt.subplot(2, 1, 1)
    plt.plot(scores, label='Anomaly Score')
    plt.scatter(np.where(anomalies)[0], scores[anomalies], color='r', label='Detected Anomalies')
    
    if fault_start:
        plt.axvline(fault_start, c='g', linestyle='--', label='Fault Introduction')
        
    plt.axhline(threshold, c='k', linestyle='--', label='Threshold')
    
    if title:
        plt.title(f'{title} - Anomaly Detection Results')
    else:
        plt.title('Anomaly Detection Results')
        
    plt.legend()
    
    # Bottom plot: true labels vs predicted labels
    plt.subplot(2, 1, 2)
    plt.plot(y_true, 'g-', label='True Labels')
    plt.plot(anomalies, 'r--', label='Predicted Labels')
    
    if fault_start:
        plt.axvline(fault_start, c='g', linestyle='--', label='Fault Introduction')
        
    if title:
        plt.title(f'{title} - True vs Predicted Labels')
    else:
        plt.title('True vs Predicted Labels')
        
    plt.xlabel('Sample')
    plt.ylabel('Label (0=Normal, 1=Fault)')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.close()

def plot_metrics_comparison(labels, precision_values, recall_values, f1_values, title=None, save_path=None):
    """
    Plot comparison of precision, recall, and F1 score for different categories
    
    Args:
        labels: Labels for x-axis
        precision_values: List of precision values
        recall_values: List of recall values
        f1_values: List of F1 scores
        title: Plot title (optional)
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(15, 7))
    
    x = np.arange(len(labels))
    width = 0.25
    
    plt.bar(x - width, precision_values, width, label='Precision')
    plt.bar(x, recall_values, width, label='Recall')
    plt.bar(x + width, f1_values, width, label='F1 Score')
    
    if title:
        plt.title(title)
    else:
        plt.title('Precision, Recall, and F1 Score Comparison')
        
    plt.xlabel('Category')
    plt.ylabel('Score')
    plt.xticks(x, labels)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.close()

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics for anomaly detection
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary containing precision, recall, F1 score, and accuracy
    """
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    accuracy = accuracy_score(y_true, y_pred)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }