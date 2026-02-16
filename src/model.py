import torch
import torch.nn as nn
import math

class RNAFoldingModel(nn.Module):
    def __init__(self, vocab_size=6, embed_dim=128, num_heads=4, num_layers=2):
        super(RNAFoldingModel, self).__init__()
        
        # 1. Embedding Layer
        # Converts integer tokens (0-5) into dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. Positional Encoding
        # Transformers have no sense of order, so we add position info
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        # 3. Transformer Encoder
        # The "Brain" that figures out the structure
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Output Head (Regressor)
        # Maps the hidden state to 3 coordinates (x, y, z)
        self.fc_out = nn.Linear(embed_dim, 3)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Seq_Len)
        Returns:
            coords: Output tensor of shape (Batch, Seq_Len, 3)
        """
        # Embed and add position info
        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        # Pass through Transformer
        # (Batch, Seq_Len, Embed_Dim)
        x = self.transformer_encoder(x)
        
        # Project to 3D coordinates
        # (Batch, Seq_Len, 3)
        coords = self.fc_out(x)
        
        return coords

class PositionalEncoding(nn.Module):
    """
    Standard Positional Encoding (Sine/Cosine)
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to embedding
        # x shape: (Batch, Seq_Len, Dim)
        return x + self.pe[:x.size(1), :].unsqueeze(0)