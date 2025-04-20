#!/usr/bin/env python3
"""
Example script for anomaly detection on custom time series datasets
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from anomaly_detection import AnomalyDetector, DataProcessor

def load_data(data_file, label_col='label', time_col=None, feature_cols=None, train_ratio=0.7, 
             fault_start=None, time_format=None, sep=','):
    """
    Load and split time series data from a single CSV file
    
    Args:
        data_file: Path to data file
        label_col: Name of the column containing labels (if None, labels will be created based on fault_start)
        time_col: Name of the time column (optional)
        feature_cols: List of feature column names (if None, all columns except label and time will be used)
        train_ratio: Ratio of training data
        fault_start: Index or time where fault starts (used to create labels if label_col is None)
        time_format: Format string for parsing time column (if time_col is provided)
        sep: Separator used in the CSV file
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test, feature_names)
    """
    # Load data
    df = pd.read_csv(data_file, sep=sep)
    print(f"Loaded data with shape: {df.shape}")
    
    # Parse time column if provided
    if time_col and time_col in df.columns:
        if time_format:
            df[time_col] = pd.to_datetime(df[time_col], format=time_format)
        else:
            df[time_col] = pd.to_datetime(df[time_col])
    
    # Determine feature columns
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != label_col and col != time_col]
    
    # Extract features
    X = df[feature_cols].values
    feature_names = feature_cols
    
    # Extract or create labels
    if label_col and label_col in df.columns:
        y = df[label_col].values
    elif fault_start is not None:
        # Create labels based on fault_start
        y = np.zeros(len(df))
        
        if isinstance(fault_start, str) and time_col:
            # Convert time string to index
            fault_time = pd.to_datetime(fault_start)
            fault_index = df[df[time_col] >= fault_time].index[0]
        else:
            # Use fault_start as index directly
            fault_index = int(fault_start)
            
        y[fault_index:] = 1
        print(f"Created labels with fault starting at index {fault_index}")
    else:
        raise ValueError("Either label_col or fault_start must be provided")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_ratio, shuffle=False
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test, feature_names

def main():
    """
    Main function to train and evaluate anomaly detection model on a custom dataset
    """
    parser = argparse.ArgumentParser(description='Train and evaluate anomaly detection model on custom time series data')
    
    # Input data arguments
    parser.add_argument('--data', type=str, required=True, help='Path to data CSV file')
    parser.add_argument('--label-col', type=str, default='label', help='Name of the label column')
    parser.add_argument('--time-col', type=str, help='Name of the time column')
    parser.add_argument('--feature-cols', type=str, help='Comma-separated list of feature columns')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Ratio of training data')
    parser.add_argument('--fault-start', type=str, help='Index or time where fault starts (used to create labels if label_col is None)')
    parser.add_argument('--time-format', type=str, help='Format string for parsing time column')
    parser.add_argument('--sep', type=str, default=',', help='Separator used in the CSV file')
    
    # Model configuration arguments
    parser.add_argument('--model-name', type=str, default='custom_anomaly_detector', help='Name of the model')
    parser.add_argument('--lookback', type=int, default=15, help='Number of time steps to look back')
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    # Transformer model arguments
    parser.add_argument('--d-model', type=int, default=64, help='Hidden dimension for transformer model')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    
    # Kalman filter arguments
    parser.add_argument('--Q', type=float, default=1e-4, help='Process noise for Kalman filter')
    parser.add_argument('--R', type=float, default=0.1, help='Measurement noise for Kalman filter')
    
    args = parser.parse_args()
    
    # Process feature columns if provided
    feature_cols = None
    if args.feature_cols:
        feature_cols = args.feature_cols.split(',')
    
    # Load and split data
    X_train, y_train, X_test, y_test, feature_names = load_data(
        args.data,
        args.label_col,
        args.time_col,
        feature_cols,
        args.train_ratio,
        args.fault_start,
        args.time_format,
        args.sep
    )
    
    # Initialize data processor
    processor = DataProcessor(
        lookback=args.lookback,
        batch_size=args.batch_size
    )
    
    # Prepare data
    data = processor.prepare_data(X_train, y_train, X_test, y_test)
    
    # Initialize model
    model = AnomalyDetector(
        input_dim=data['feature_dim'],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        Q=args.Q,
        R=args.R
    )
    
    # Train model
    print(f"Training model: {args.model_name}")
    history = model.train_model(
        model_name=args.model_name,
        train_loader=data['train_loader'],
        val_loader=data['val_loader'],
        num_epochs=args.epochs,
        lr=args.lr
    )
    
    # Evaluate model
    print("\nEvaluating model on test data...")
    
    # Determine fault start in test data for visualization
    fault_index = None
    if args.fault_start is not None:
        # Calculate approximate fault start in test windows
        if isinstance(args.fault_start, str) and args.time_col:
            # This is just an approximation for visualization
            test_size = len(y_test)
            fault_ratio = sum(y_test) / test_size
            fault_index = int((1 - fault_ratio) * test_size) - args.lookback
        else:
            # Rough estimate
            train_size = len(y_train)
            fault_start_original = int(args.fault_start)
            if fault_start_original > train_size:
                fault_index = fault_start_original - train_size - args.lookback
    
    eval_results = model.evaluate(
        data['X_test_windows'],
        data['y_test_windows'],
        threshold=0.5,
        fault_start=fault_index,
        title=args.model_name,
        save_path=f'{args.model_name}_evaluation.png'
    )
    
    print("\nFinal evaluation results:")
    for metric, value in eval_results['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model
    final_model_path = f'{args.model_name}_final_model.pth'
    print(f"\nModel saved at: {final_model_path}")
    print(f"Evaluation plot saved at: {args.model_name}_evaluation.png")
    
    print("\nDone!")

if __name__ == "__main__":
    main()