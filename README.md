# TransKal

![Image Description](transkal-model-architecture.jpg)

A Python package for time series anomaly detection using Transformer neural networks and Kalman filtering. This package is designed for general-purpose anomaly detection across various time series data. It has been tested on the Tennessee Eastman Process (TEP) dataset and has also demonstrated strong performance on other public benchmarks, such as the Server Machine Dataset (SMD), where it achieved an F1-score of 88.10% and accuracy of 98.49%.

## Features

- Transformer-based model with positional encoding
- Kalman filtering for smoothing predictions
- Easy training and evaluation API
- Support for custom datasets
- Visualization tools for model performance

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/naixinlyu/TransKal.git
   cd TransKal
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Quick Start with Command Line

The simplest way to train and evaluate an anomaly detection model on your dataset is using the `run.py` script:

```bash
python run.py --train path/to/training_data.csv --test path/to/testing_data.csv --label-col label
```

For a dataset that doesn't have labels but has a known fault start time:

```bash
python run.py --data path/to/data.csv --fault-start 500 --train-ratio 0.7
```

### Custom Example with Command Line

For more advanced usage with custom time series data:

```bash
python examples/custom_example.py --data path/to/data.csv --label-col anomaly --time-col timestamp --feature-cols "feature1,feature2,feature3" --lookback 20 --epochs 50
```

### TEP Dataset Example

If you have the Tennessee Eastman Process dataset in RData format:

```bash
python examples/tep_example.py
```

For detailed implementation and analysis, refer to:

- [TransKal - Unsupervised Anomaly Detection on TEP](TransKal-Unsupervised_Anomaly_Detection_on_TEP.ipynb): training with normal data.
- [Fault Detection with Transformer and Kalman Filter for TEP](Fault_Detection_with_Transformer_and_Kalman_Filter_for_TEP.ipynb): training with fault-specific data.

### Python API Example

You can also use the package directly in your Python code:

```python
from anomaly_detection import AnomalyDetector, DataProcessor
import numpy as np

# Prepare data
X_train = ...  # Training features
y_train = ...  # Training labels
X_test = ...   # Testing features
y_test = ...   # Testing labels

# Initialize data processor
processor = DataProcessor(lookback=15, batch_size=16)
data = processor.prepare_data(X_train, y_train, X_test, y_test)

# Initialize and train the model
model = AnomalyDetector(input_dim=data['feature_dim'])
model.train_model(
    model_name='my_model',
    train_loader=data['train_loader'],
    val_loader=data['val_loader'],
    num_epochs=30
)

# Evaluate model
results = model.evaluate(
    data['X_test_windows'],
    data['y_test_windows'],
    threshold=0.5
)

print(f"F1 Score: {results['metrics']['f1']:.4f}")
```

## File Structure

```
TransKal/
├── README.md
├── requirements.txt
├── anomaly_detection/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── transformer.py
│   │   └── kalman_filter.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── visualization.py
│   └── trainer.py
├── examples/
│   ├── tep_example.py
│   └── custom_example.py
├── run.py
├── TransKal-Unsupervised_Anomaly_Detection_on_TEP.ipynb
└── Fault_Detection_with_Transformer_and_Kalman_Filter_for_TEP.ipynb
```

## Model Architecture

The anomaly detection model consists of two main components:

1. **Transformer Model**: Uses self-attention mechanisms to capture temporal dependencies in time series data
2. **Kalman Filter**: Applied to smooth the model predictions and reduce false positives

## Parameters

The model accepts several parameters that can be tuned for your specific dataset:

- `lookback`: Number of time steps to look back (window size)
- `d_model`: Hidden dimension of the transformer model
- `nhead`: Number of attention heads
- `num_layers`: Number of transformer encoder layers
- `dropout`: Dropout probability for regularization
- `Q`: Process noise for Kalman filter
- `R`: Measurement noise for Kalman filter

## Input Data Format

For CSV files, the expected format is:

- One row per time step
- Columns for features
- A column for labels (0 for normal, 1 for anomaly)
- Optional time column for time series data

## Acknowledgments

This project uses the Tennessee Eastman Process (TEP) dataset, a widely recognized benchmark in process monitoring and fault detection research. The TEP dataset simulates a chemical manufacturing process with 22 different fault scenarios, making it ideal for evaluating anomaly detection algorithms.

The dataset includes:
- Normal operating conditions (fault-free)
- 21 different fault scenarios
- 52 process variables (41 measured + 11 manipulated)
- Time-series data with faults introduced at specific points

Data is sourced from [Tennessee Eastman Process Simulation Dataset on Kaggle](https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset).

The Tennessee Eastman Process was originally developed by Downs and Vogel (1993) as a realistic industrial challenge problem, and has since become a standard benchmark for evaluating process monitoring and fault detection methodologies.
