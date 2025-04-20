#!/usr/bin/env python3
"""
Example script for anomaly detection on the Tennessee Eastman Process (TEP) dataset
"""
import os
import numpy as np
import pandas as pd
import pyreadr
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from anomaly_detection import AnomalyDetector, DataProcessor

def load_tep_data(fault_free_train_path, faulty_train_path, faulty_test_path):
    """
    Load TEP dataset from R data files
    
    Args:
        fault_free_train_path: Path to fault-free training data (RData file)
        faulty_train_path: Path to faulty training data (RData file)
        faulty_test_path: Path to faulty testing data (RData file)
        
    Returns:
        Tuple of DataFrames (df_train_faultfree, df_train_faulty, df_test_faulty)
    """
    print("Loading TEP datasets...")
    df_train_faultfree = pyreadr.read_r(fault_free_train_path)['fault_free_training']
    df_train_faulty = pyreadr.read_r(faulty_train_path)['faulty_training']
    df_test_faulty = pyreadr.read_r(faulty_test_path)['faulty_testing']
    
    print(f"Training fault-free data shape: {df_train_faultfree.shape}")
    print(f"Training faulty data shape: {df_train_faulty.shape}")
    print(f"Testing faulty data shape: {df_test_faulty.shape}")
    
    return df_train_faultfree, df_train_faulty, df_test_faulty

def train_and_evaluate_fault_model(fault_type, df_train_faultfree, df_train_faulty, df_test_faulty, 
                                   lookback=15, num_epochs=30, batch_size=16):
    """
    Train and evaluate a model for a specific fault type
    
    Args:
        fault_type: Type of fault to train for
        df_train_faultfree: DataFrame with fault-free training data
        df_train_faulty: DataFrame with faulty training data
        df_test_faulty: DataFrame with faulty testing data
        lookback: Number of time steps to look back
        num_epochs: Number of epochs to train
        batch_size: Batch size for training
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*80}")
    print(f"Processing Fault Type {fault_type}")
    print(f"{'='*80}")
    
    # Prepare training data
    # Extract data for current fault type from training set
    train_fault_data = df_train_faulty[df_train_faulty['faultNumber'] == fault_type]
    
    if len(train_fault_data) == 0:
        print(f"No training data found for fault type {fault_type}. Skipping...")
        return None
    
    # Prepare feature data (normal data and current fault type data)
    X_train_normal = df_train_faultfree.iloc[:, 3:].values
    X_train_fault = train_fault_data.iloc[:, 3:].values
    
    # Create labels: normal=0, fault=1
    # Note: In training set, fault is introduced after sample 20 (1 hour, one sample every 3 minutes)
    y_train_normal = np.zeros(len(X_train_normal))
    y_train_fault = np.ones(len(X_train_fault))
    y_train_fault[:20] = 0  # First 20 samples are normal
    
    # Combine training data
    X_train = np.vstack([X_train_normal, X_train_fault])
    y_train = np.concatenate([y_train_normal, y_train_fault])
    
    # Prepare test data
    # Extract data for current fault type from test set
    test_fault_data = df_test_faulty[df_test_faulty['faultNumber'] == fault_type]
    
    if len(test_fault_data) == 0:
        print(f"No test data found for fault type {fault_type}. Skipping evaluation...")
        return None
    
    # Prepare test features and labels
    X_test = test_fault_data.iloc[:, 3:].values
    
    # Note: In test set, fault is introduced after sample 160 (8 hours, one sample every 3 minutes)
    y_test = np.ones(len(X_test))
    y_test[:160] = 0  # First 160 samples are normal
    
    # Initialize data processor
    processor = DataProcessor(lookback=lookback, batch_size=batch_size)
    
    # Prepare data
    data = processor.prepare_data(X_train, y_train, X_test, y_test)
    
    # Initialize model
    model_name = f"fault_{fault_type}"
    print(f"Initializing model for {model_name}...")
    
    model = AnomalyDetector(
        input_dim=data['feature_dim'],
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        Q=1e-5,
        R=0.1
    )
    
    # Train model
    history = model.train_model(
        model_name=model_name,
        train_loader=data['train_loader'],
        val_loader=data['val_loader'],
        num_epochs=num_epochs,
        lr=1e-3
    )
    
    # Evaluate model
    print(f"Evaluating model for fault type {fault_type}...")
    
    # Calculate fault start index in test windows
    fault_start = 160 - lookback
    
    eval_results = model.evaluate(
        data['X_test_windows'],
        data['y_test_windows'],
        threshold=0.5,
        fault_start=fault_start,
        title=f'Fault Type {fault_type}',
        save_path=f'fault_{fault_type}_evaluation.png'
    )
    
    # Add fault type to results
    eval_results['metrics']['fault_type'] = fault_type
    
    return eval_results['metrics']

def main():
    """
    Main function to train and evaluate models for all fault types in TEP dataset
    """
    # Paths to TEP dataset files
    fault_free_train_path = "TEP_FaultFree_Training.RData"
    faulty_train_path = "TEP_Faulty_Training.RData"
    faulty_test_path = "TEP_Faulty_Testing.RData"
    
    # Check if files exist
    for file_path in [fault_free_train_path, faulty_train_path, faulty_test_path]:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found.")
            return
    
    # Load TEP data
    df_train_faultfree, df_train_faulty, df_test_faulty = load_tep_data(
        fault_free_train_path, 
        faulty_train_path, 
        faulty_test_path
    )
    
    # Set parameters
    lookback = 15
    num_epochs = 30
    batch_size = 16
    
    # Get all fault types
    fault_types = sorted(df_train_faulty['faultNumber'].unique())
    print(f"Found fault types: {fault_types}")
    
    # Store results for all fault types
    all_results = {}
    
    # Train and evaluate model for each fault type individually
    for fault_type in fault_types:
        print(f"\nProcessing fault type {fault_type}...")
        metrics = train_and_evaluate_fault_model(
            fault_type, 
            df_train_faultfree, 
            df_train_faulty, 
            df_test_faulty,
            lookback=lookback,
            num_epochs=num_epochs,
            batch_size=batch_size
        )
        
        if metrics:
            all_results[fault_type] = metrics
    
    # Summarize results
    print("\n" + "="*80)
    print("Summary of Results for All Fault Types")
    print("="*80)
    
    f1_scores = []
    accuracies = []
    precisions = []
    recalls = []
    
    for fault_type, metrics in all_results.items():
        print(f"Fault {fault_type} - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, " +
              f"F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        f1_scores.append(metrics['f1'])
        accuracies.append(metrics['accuracy'])
    
    if f1_scores:
        print("\nOverall Performance:")
        print(f"Average Precision: {np.mean(precisions):.4f}")
        print(f"Average Recall: {np.mean(recalls):.4f}")
        print(f"Average F1 Score: {np.mean(f1_scores):.4f}")
        print(f"Average Accuracy: {np.mean(accuracies):.4f}")
        
        # Visualize metrics by fault type
        fault_types = list(all_results.keys())
        precision_values = [all_results[ft]['precision'] for ft in fault_types]
        recall_values = [all_results[ft]['recall'] for ft in fault_types]
        f1_values = [all_results[ft]['f1'] for ft in fault_types]
        
        # Create plot
        plt.figure(figsize=(15, 7))
        x = np.arange(len(fault_types))
        width = 0.25
        
        plt.bar(x - width, precision_values, width, label='Precision')
        plt.bar(x, recall_values, width, label='Recall')
        plt.bar(x + width, f1_values, width, label='F1 Score')
        
        plt.title('Anomaly Detection Performance by Fault Type')
        plt.xlabel('Fault Type')
        plt.ylabel('Score')
        plt.xticks(x, fault_types)
        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.savefig('tep_fault_comparison.png')
        plt.close()
    
    return all_results

if __name__ == "__main__":
    main()