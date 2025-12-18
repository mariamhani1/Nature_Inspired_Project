import torch
import torch.nn as nn
import math

# Define Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_length, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Define the Transformer model architecture
class TextTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers,
                 dim_feedforward, num_classes, max_length=200, dropout=0.1):
        super(TextTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embedding_dim, max_length, dropout)

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)

        self.embedding_dim = embedding_dim

    def forward(self, x):
        # x shape: (batch_size, seq_length)

        # Create padding mask
        padding_mask = (x == 0)  # True for padding tokens

        # Embedding
        embedded = self.embedding(x) * math.sqrt(self.embedding_dim)  # (batch_size, seq_length, embedding_dim)
        embedded = self.pos_encoder(embedded)

        # Transformer encoding
        transformer_out = self.transformer_encoder(
            embedded,
            src_key_padding_mask=padding_mask
        )  # (batch_size, seq_length, embedding_dim)

        # Global average pooling over sequence dimension
        # Mask out padding tokens before pooling
        mask_expanded = (~padding_mask).unsqueeze(-1).float()  # (batch_size, seq_length, 1)
        summed = (transformer_out * mask_expanded).sum(dim=1)  # (batch_size, embedding_dim)
        counts = mask_expanded.sum(dim=1)  # (batch_size, 1)
        pooled = summed / (counts + 1e-9)  # (batch_size, embedding_dim)

        # Classification
        pooled = self.dropout(pooled)
        output = self.fc(pooled)  # (batch_size, num_classes)

        return output
