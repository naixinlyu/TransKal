#!/usr/bin/env python3
"""
A simple example of using the anomaly detection package with a custom dataset.
This script demonstrates how to use the package with a simple time series dataset.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from anomaly_detection import AnomalyDetector, DataProcessor

def generate_synthetic_data(n_samples=1000, anomaly_start=700, noise_level=0.1):
    """
    Generate synthetic time series data with anomalies
    
    Args:
        n_samples: Number of data points
        anomaly_start: Index where anomalies start
        noise_level: Level of noise to add
        
    Returns:
        X: Features (time series data)
        y: Labels (0: normal, 1: anomaly)
    """
    # Generate time and normal pattern
    t = np.linspace(0, 10 * np.pi, n_samples)
    normal = np.sin(t)
    
    # Add some noise
    noise = np.random.normal(0, noise_level, n_samples)
    normal = normal + noise
    
    # Create anomalies
    anomaly = np.copy(normal)
    anomaly[anomaly_start:] = anomaly[anomaly_start:] + 0.5 * np.sin(3 * t[anomaly_start:])
    
    # Create features (you can add more features as needed)
    X = np.column_stack((
        anomaly,                        # Original signal with anomalies
        np.sin(2 * t) + noise * 0.5,    # Related signal 1
        np.cos(0.5 * t) + noise * 0.5,  # Related signal 2
    ))
    
    # Create labels (0 for normal, 1 for anomaly)
    y = np.zeros(n_samples)
    y[anomaly_start:] = 1
    
    return X, y

def plot_data(X, y, title="Synthetic Time Series Data with Anomalies"):
    """
    Plot the synthetic data
    
    Args:
        X: Features
        y: Labels
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Plot the main signal
    plt.plot(X[:, 0], label='Signal 1')
    plt.plot(X[:, 1], alpha=0.5, label='Signal 2')
    plt.plot(X[:, 2], alpha=0.5, label='Signal 3')
    
    # Highlight anomalies
    anomaly_indices = np.where(y == 1)[0]
    plt.axvspan(anomaly_indices[0], anomaly_indices[-1], alpha=0.2, color='red', label='Anomaly Region')
    
    plt.legend()
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.savefig('synthetic_data.png')
    plt.close()
    
    print(f"Data visualization saved as 'synthetic_data.png'")

def main():
    print("Generating synthetic time series data...")
    X, y = generate_synthetic_data(n_samples=2000, anomaly_start=1400, noise_level=0.1)
    
    # Plot the data
    plot_data(X, y)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False  # No shuffle to maintain time series order
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Save the data to CSV for future use (optional)
    df = pd.DataFrame(
        np.column_stack([X, y]), 
        columns=['feature1', 'feature2', 'feature3', 'anomaly']
    )
    df.to_csv('synthetic_data.csv', index=False)
    print("Data saved to 'synthetic_data.csv'")
    
    # Initialize data processor
    processor = DataProcessor(lookback=20, batch_size=32, val_ratio=0.2)
    
    # Prepare data
    data = processor.prepare_data(X_train, y_train, X_test, y_test)
    
    # Initialize model
    model = AnomalyDetector(
        input_dim=data['feature_dim'],
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        Q=1e-4,  # Kalman filter process noise
        R=0.1     # Kalman filter measurement noise
    )
    
    # Train model
    print("\nTraining model...")
    history = model.train_model(
        model_name="synthetic_data_model",
        train_loader=data['train_loader'],
        val_loader=data['val_loader'],
        num_epochs=30,
        lr=1e-3,
        plot_interval=5
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    eval_results = model.evaluate(
        data['X_test_windows'],
        data['y_test_windows'],
        threshold=0.5,
        title="Synthetic Data",
        save_path='synthetic_data_evaluation.png'
    )
    
    print("\nEvaluation results:")
    for metric, value in eval_results['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nExperiment completed!")
    print("- Model saved as 'synthetic_data_model_final_model.pth'")
    print("- Evaluation results saved as 'synthetic_data_evaluation.png'")

if __name__ == "__main__":
    main()