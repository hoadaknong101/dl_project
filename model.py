import torch
import torch.nn as nn

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
        # padding_idx=pad_idx: báo cho lớp Embedding bỏ qua token padding khi tính toán
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
        
        # 3. Lớp Linear (Fully Connected)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        # 4. Lớp Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, lengths):
        """
        Quá trình forward
        Args:
            text (Tensor): (batch_size, seq_len)
            lengths (Tensor): (batch_size) - độ dài thực của từng sequence
        
        Returns:
            Tensor: (batch_size, output_dim) - Logits
        """
        
        # 1. Embedding
        embedded = self.embedding(text)
        
        # 2. LSTM
        output, (hidden, cell) = self.lstm(embedded)
        
        # 3. Lấy hidden state cuối cùng
        if self.lstm.bidirectional:
            # Nối hidden state của chiều thuận (lớp cuối) và chiều ngược (lớp cuối)
            # hidden[-2,:,:] là forward của lớp cuối
            # hidden[-1,:,:] là backward của lớp cuối
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
                    
        # 4. Qua lớp Dropout và Linear
        dropped_hidden = self.dropout(hidden)
        prediction = self.fc(dropped_hidden)
                
        return prediction