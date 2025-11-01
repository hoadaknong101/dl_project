import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        """
        Khởi tạo mô hình LSTM
        Args:
            vocab_size (int): Kích thước từ điển (số lượng từ duy nhất)
            embedding_dim (int): Kích thước vector embedding
            hidden_dim (int): Kích thước của lớp ẩn LSTM
            output_dim (int): Kích thước đầu ra (số lượng lớp, ở đây là 3)
            n_layers (int): Số lớp LSTM (ví dụ 1, 2)
            bidirectional (bool): Sử dụng LSTM 2 chiều hay không
            dropout (float): Tỷ lệ dropout
            pad_idx (int): Chỉ số của token <pad> trong từ điển
        """
        super().__init__()
        
        # 1. Lớp Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # 2. Lớp LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # 3. Lớp MLP (Linear + ReLU + Dropout + Linear)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * (2 if bidirectional else 1), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, text, lengths):
        """
        Quá trình forward
        Args:
            text (Tensor): (batch_size, seq_len)
            lengths (Tensor): (batch_size) - độ dài thực của từng sequence
        
        Returns:
            Tensor: (batch_size, output_dim) - Logits
        """
        embedded = self.embedding(text)
        packed_embedded = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed_embedded)
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        return self.fc(hidden)