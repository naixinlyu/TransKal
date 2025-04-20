#!/usr/bin/env python3
"""
Main entry point for anomaly detection using transformer-based model.
This script can train and evaluate the model on custom datasets.
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from anomaly_detection import AnomalyDetector, DataProcessor

def load_data(train_file, test_file=None, label_col='label', feature_cols=None, sep=','):
    """
    Load data from CSV files
    
    Args:
        train_file: Path to training data file
        test_file: Path to testing data file (optional)
        label_col: Name of the column containing labels
        feature_cols: List of feature column names (if None, all columns except label_col)
        sep: Separator used in CSV files
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    # Load training data
    df_train = pd.read_csv(train_file, sep=sep)
    
    # Determine feature columns if not specified
    if feature_cols is None:
        feature_cols = [col for col in df_train.columns if col != label_col]
        
    # Extract features and labels
    X_train = df_train[feature_cols].values
    y_train = df_train[label_col].values
    
    # Load testing data if provided
    X_test, y_test = None, None
    if test_file:
        df_test = pd.read_csv(test_file, sep=sep)
        X_test = df_test[feature_cols].values
        y_test = df_test[label_col].values
        
    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train, X_test=None, y_test=None, model_name='anomaly_detector', 
                lookback=15, batch_size=16, num_epochs=30, learning_rate=1e-3, 
                d_model=64, nhead=4, num_layers=2, dropout=0.1, Q=1e-4, R=0.1):
    """
    Train anomaly detection model
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features (optional)
        y_test: Testing labels (optional)
        model_name: Name of the model
        lookback: Number of time steps to look back
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        d_model: Hidden dimension for transformer model
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout probability
        Q: Process noise for Kalman filter
        R: Measurement noise for Kalman filter
        
    Returns:
        Trained model and evaluation results if test data is provided
    """
    # Initialize data processor
    processor = DataProcessor(lookback=lookback, batch_size=batch_size)
    
    # Prepare data
    data = processor.prepare_data(X_train, y_train, X_test, y_test)
    
    # Initialize model
    model = AnomalyDetector(
        input_dim=data['feature_dim'],
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        Q=Q,
        R=R
    )
    
    # Train model
    history = model.train_model(
        model_name=model_name,
        train_loader=data['train_loader'],
        val_loader=data['val_loader'],
        num_epochs=num_epochs,
        lr=learning_rate
    )
    
    # Evaluate model if test data is provided
    eval_results = None
    if X_test is not None and y_test is not None:
        print("\nEvaluating model on test data...")
        
        eval_results = model.evaluate(
            data['X_test_windows'],
            data['y_test_windows'],
            threshold=0.5,
            title=model_name,
            save_path=f'{model_name}_evaluation.png'
        )
        
    return model, history, eval_results

def evaluate_saved_model(model_path, X_test, y_test, lookback=15, batch_size=16, 
                        d_model=64, nhead=4, num_layers=2, dropout=0.1, Q=1e-4, R=0.1):
    """
    Evaluate a saved model on test data
    
    Args:
        model_path: Path to saved model checkpoint
        X_test: Test features
        y_test: Test labels
        lookback: Number of time steps to look back
        batch_size: Batch size
        d_model, nhead, num_layers, dropout, Q, R: Model parameters
        
    Returns:
        Evaluation results
    """
    # Load checkpoint to get model parameters
    checkpoint = torch.load(model_path, map_location='cpu')
    model_name = checkpoint.get('model_name', 'loaded_model')
    
    # Initialize data processor
    processor = DataProcessor(lookback=lookback, batch_size=batch_size)
    
    # Prepare test data
    X_test_scaled = processor.scaler.fit_transform(X_test)  # Note: In real use, fit on train data first
    X_test_windows, y_test_windows = processor.create_sliding_windows(X_test_scaled, y_test, lookback)
    
    # Initialize model
    feature_dim = X_test_windows.shape[2]
    model = AnomalyDetector(
        input_dim=feature_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        Q=Q,
        R=R
    )
    
    # Load model weights
    model.transformer.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    eval_results = model.evaluate(
        X_test_windows,
        y_test_windows,
        threshold=0.5,
        title=f'Loaded Model: {model_name}',
        save_path=f'{model_name}_evaluation.png'
    )
    
    return eval_results

def main():
    """
    Main function to parse command line arguments and run the anomaly detection pipeline
    """
    parser = argparse.ArgumentParser(description='Train and evaluate transformer-based anomaly detection model')
    
    # Input data arguments
    parser.add_argument('--train', type=str, required=True, help='Path to training data CSV file')
    parser.add_argument('--test', type=str, help='Path to testing data CSV file')
    parser.add_argument('--label-col', type=str, default='label', help='Name of the label column')
    parser.add_argument('--feature-cols', type=str, help='Comma-separated list of feature columns (if not provided, all columns except label will be used)')
    parser.add_argument('--sep', type=str, default=',', help='Separator used in CSV files')
    
    # Model configuration arguments
    parser.add_argument('--model-name', type=str, default='anomaly_detector', help='Name of the model')
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
    
    # Evaluation arguments
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for anomaly detection')
    parser.add_argument('--fault-start', type=int, help='Index where fault was introduced (for visualization)')
    
    # Load model mode
    parser.add_argument('--load-model', type=str, help='Path to saved model for evaluation only (skip training)')
    
    args = parser.parse_args()
    
    # Process feature columns if provided
    feature_cols = None
    if args.feature_cols:
        feature_cols = args.feature_cols.split(',')
    
    # Load data
    print(f"Loading data from {args.train}" + (f" and {args.test}" if args.test else ""))
    X_train, y_train, X_test, y_test = load_data(
        args.train, 
        args.test, 
        args.label_col, 
        feature_cols, 
        args.sep
    )
    
    print(f"Training data shape: {X_train.shape}")
    if X_test is not None:
        print(f"Testing data shape: {X_test.shape}")
    
    # Either load and evaluate a saved model or train a new one
    if args.load_model:
        print(f"Loading model from {args.load_model} and evaluating...")
        if X_test is None or y_test is None:
            print("Error: Test data is required for evaluation mode")
            return
            
        eval_results = evaluate_saved_model(
            args.load_model,
            X_test,
            y_test,
            args.lookback,
            args.batch_size,
            args.d_model,
            args.nhead,
            args.num_layers,
            args.dropout,
            args.Q,
            args.R
        )
        
        print("Evaluation results:")
        for metric, value in eval_results['metrics'].items():
            print(f"  {metric}: {value:.4f}")
    else:
        print(f"Training new model: {args.model_name}")
        model, history, eval_results = train_model(
            X_train,
            y_train,
            X_test,
            y_test,
            args.model_name,
            args.lookback,
            args.batch_size,
            args.epochs,
            args.lr,
            args.d_model,
            args.nhead,
            args.num_layers,
            args.dropout,
            args.Q,
            args.R
        )
        
        if eval_results:
            print("\nFinal evaluation results:")
            for metric, value in eval_results['metrics'].items():
                print(f"  {metric}: {value:.4f}")
    
    print("\nDone!")
    
if __name__ == "__main__":
    main()