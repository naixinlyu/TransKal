import numpy as np
from tqdm import tqdm

class KalmanFilter:
    """
    Simple implementation of Kalman filter for 1D signal smoothing
    """
    def __init__(self, dim_x=1, Q=1e-4, R=0.1):
        """
        Initialize Kalman filter
        
        Args:
            dim_x: State dimension
            Q: Process noise covariance
            R: Measurement noise covariance
        """
        self.dim_x = dim_x
        self.Q = Q
        self.R = R
        
    def filter(self, y_noisy):
        """
        Apply Kalman filtering to noisy observations
        
        Args:
            y_noisy: Noisy observations
            
        Returns:
            x_est: Filtered (smoothed) state estimates
        """
        n = len(y_noisy)
        x_est = np.zeros(n)  # State estimates
        P = np.zeros(n)      # Error covariance
        
        # Initial state and covariance
        x_est[0] = y_noisy[0]
        P[0] = 1.0
        
        for k in tqdm(range(1, n), desc="Kalman filtering", leave=False):
            # Prediction step
            x_pred = x_est[k-1]      # State prediction
            P_pred = P[k-1] + self.Q  # Covariance prediction
            
            # Update step with measurement
            K = P_pred / (P_pred + self.R)  # Kalman gain
            x_est[k] = x_pred + K * (y_noisy[k] - x_pred)  # State update
            P[k] = (1 - K) * P_pred   # Covariance update
            
        return x_est