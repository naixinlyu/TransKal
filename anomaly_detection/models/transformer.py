import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer model
    Adds information about position of tokens in the sequence
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to input
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerPredictor(nn.Module):
    """
    Transformer model for anomaly detection
    """
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        """
        Initialize Transformer model
        
        Args:
            input_dim: Input feature dimension
            d_model: Hidden dimension for the model
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dropout: Dropout probability
        """
        super().__init__()
        
        # Input embedding
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=256, 
            dropout=dropout, 
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer for anomaly detection (binary classification)
        self.fc = nn.Linear(d_model, 1)
        
    def forward(self, x):
        """
        Forward pass of the model
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        batch_size, seq_len, _ = x.size()
        
        # Embedding and scale
        x = self.embedding(x) * np.sqrt(self.embedding.out_features)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create causal mask (lower triangular)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x, mask=mask)
        
        # Get the output for the last position
        output = self.fc(x[:, -1, :])
        
        return output