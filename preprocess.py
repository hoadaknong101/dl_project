import re
import torch
import pandas as pd
from collections import Counter
from underthesea import word_tokenize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from config import LABEL_MAP

def clean_text(text):
    """
    Làm sạch văn bản: bỏ ký tự đặc biệt, link, và chuyển về chữ thường
    Args:
        text (str): Văn bản đầu vào
    Returns:
        str: Văn bản đã được làm sạch
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()  # Chuyển về chữ thường
    text = re.sub(r'http\S+', '', text)  # Xóa URL
    text = re.sub(r'[^\w\s]', '', text)  # Xóa dấu câu và ký tự đặc biệt
    text = re.sub(r'\d+', '', text)  # Xóa số
    text = text.strip()  # Xóa khoảng trắng thừa
    
    return text

def tokenize_vietnamese(text):
    """
    Sử dụng underthesea để token hóa văn bản
    Args:
        text (str): Văn bản đầu vào
    Returns:
        list: Danh sách các token
    """
    return word_tokenize(text)

def build_vocab(tokenized_texts, max_vocab_size=10000, min_freq=2):
    """
    Xây dựng từ điển từ các văn bản đã token hóa
    Args:
        tokenized_texts (list of list of str): Danh sách các văn bản đã được token hóa
        max_vocab_size (int): Kích thước tối đa của từ điển
        min_freq (int): Tần suất xuất hiện tối thiểu để một từ được đưa vào từ điển
    Returns:
        dict: Từ điển ánh xạ từ sang chỉ số
    """
    word_counts = Counter()
    for text in tokenized_texts:
        word_counts.update(text)
    
    # Lọc các từ theo tần suất xuất hiện
    words = [word for word, count in word_counts.items() if count >= min_freq]
    
    # Giới hạn kích thước vocab
    if len(words) > max_vocab_size - 2: # Dành chỗ cho <pad> và <unk>
        words = [word for word, count in word_counts.most_common(max_vocab_size - 2)]
        
    # Tạo từ điển: <pad> (padding) và <unk> (unknown) là 2 token đặc biệt
    vocab = {'<pad>': 0, '<unk>': 1}
    vocab.update({word: i+2 for i, word in enumerate(words)})
    return vocab

def text_to_sequence(tokenized_text, vocab):
    """
    Chuyển đổi tokenized text sang sequence số dựa trên vocab
    Args:
        tokenized_text (list of str): Danh sách các token
        vocab (dict): Từ điển ánh xạ từ sang chỉ số
    Returns:
        list: Danh sách các chỉ số tương ứng với tokenized_text
    """
    return [vocab.get(word, vocab['<unk>']) for word in tokenized_text]

class VietnameseTextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_seq_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize và chuyển sang sequence
        tokens = tokenize_vietnamese(clean_text(text))
        sequence = text_to_sequence(tokens, self.vocab)
        
        # Cắt bớt nếu dài hơn max_seq_len
        if len(sequence) > self.max_seq_len:
            sequence = sequence[:self.max_seq_len]
            
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def create_collate_fn(pad_idx):
    def collate_fn(batch):
        """
        Xử lý padding cho từng batch.
        Nó sẽ pad các sequence trong batch về độ dài của sequence dài nhất TRONG BATCH ĐÓ.
        Args:
            batch (list of tuples): Mỗi tuple là (sequence_tensor, label_tensor)
        Returns:
            padded_sequences (Tensor): Tensor các sequence đã được pad
        """
        sequences, labels = zip(*batch)
        
        # Lấy độ dài của từng sequence
        lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
        
        # Pad các sequence
        # batch_first=True -> output shape (batch_size, max_seq_len_in_batch)
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=pad_idx)
        
        # Sắp xếp batch theo độ dài (cần thiết cho pack_padded_sequence nếu dùng)
        # Ở đây chúng ta không dùng pack_padded_sequence nhưng vẫn trả về lengths
        # để tiện cho việc theo dõi hoặc tối ưu sau này.
        labels = torch.stack(labels)
        
        return padded_sequences, labels, lengths
    
    return collate_fn

def get_dataloaders(file_path, vocab, batch_size, max_seq_len, test_size=0.2):
    """
    Tải dữ liệu, chia train/test và tạo DataLoaders
    Args:
        file_path (str): Đường dẫn tới file CSV chứa dữ liệu
        vocab (dict): Từ điển ánh xạ từ sang chỉ số
        batch_size (int): Kích thước batch
        max_seq_len (int): Độ dài tối đa của sequence
        test_size (float): Tỷ lệ dữ liệu dùng cho test
    Returns:
        train_loader (DataLoader): DataLoader cho train
        test_loader (DataLoader): DataLoader cho test
    """
    df = pd.read_csv(file_path)
    
    # Chỉ lấy các cột cần thiết và map nhãn
    df = df[['comment', 'label']].dropna()
    df['label_id'] = df['label'].map(LABEL_MAP)
    
    # Bỏ qua các dòng có nhãn không xác định
    df = df.dropna(subset=['label_id'])
    df['label_id'] = df['label_id'].astype(int)
    
    # Chia train/test (ví dụ, 80% train, 20% test/val)
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=42, stratify=df['label_id'])

    train_dataset = VietnameseTextDataset(
        df_train['comment'].tolist(), 
        df_train['label_id'].tolist(), 
        vocab, 
        max_seq_len
    )
    
    test_dataset = VietnameseTextDataset(
        df_test['comment'].tolist(), 
        df_test['label_id'].tolist(), 
        vocab, 
        max_seq_len
    )
    
    pad_idx = vocab['<pad>']
    collate_batch = create_collate_fn(pad_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    
    return train_loader, test_loader

def prepare_vocab(file_path, max_vocab_size=10000, min_freq=2):
    """
    Chuẩn bị từ điển từ file dữ liệu
    Args:
        file_path (str): Đường dẫn tới file CSV chứa dữ liệu
        max_vocab_size (int): Kích thước tối đa của từ điển
        min_freq (int): Tần suất xuất hiện tối thiểu để một từ được đưa vào từ điển
    Returns:
        dict: Từ điển ánh xạ từ sang chỉ số
    """
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['comment'])
    
    print("Đang token hóa và xây dựng từ điển...")
    tokenized_texts = [tokenize_vietnamese(clean_text(text)) for text in df['comment'].tolist()]
    vocab = build_vocab(tokenized_texts, max_vocab_size, min_freq)
    print(f"Xây dựng từ điển hoàn tất với {len(vocab)} từ.")

    return vocab