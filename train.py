import os
import torch
import pickle
from tqdm import tqdm
import torch.nn as nn
from model import LSTM
import torch.optim as optim
from metrics import evaluate_model
from preprocess import prepare_vocab, get_dataloaders
from config import (EMBEDDING_DIM, 
                    HIDDEN_DIM,
                    OUTPUT_DIM, 
                    N_LAYERS,
                    BIDIRECTIONAL,
                    DROPOUT, 
                    MAX_SEQ_LEN,
                    MODEL_SAVE_PATH,
                    VOCAB_SAVE_PATH,
                    DATA_FILE,
                    MAX_VOCAB_SIZE,
                    MIN_FREQ,
                    BATCH_SIZE,
                    LEARNING_RATE,
                    N_EPOCHS,
                    LOG_FILE_PATH)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Thiết bị sử dụng: {device}")

print("Bắt đầu chuẩn bị từ điển...")

if os.path.exists(VOCAB_SAVE_PATH):
    print("Đang tải từ điển đã lưu...")
    with open(VOCAB_SAVE_PATH, 'rb') as f:
        vocab = pickle.load(f)
else:
    print("Tạo từ điển mới...")
    vocab = prepare_vocab(DATA_FILE, MAX_VOCAB_SIZE, MIN_FREQ)
    with open(VOCAB_SAVE_PATH, 'wb') as f:
        pickle.dump(vocab, f)

VOCAB_SIZE = len(vocab)
PAD_IDX = vocab['<pad>']

print(f"Kích thước từ điển: {VOCAB_SIZE}")

print("Chuẩn bị DataLoaders...")
train_loader, test_loader = get_dataloaders(
    DATA_FILE, 
    vocab, 
    BATCH_SIZE, 
    MAX_SEQ_LEN,
    test_size=0.2
)
print("DataLoaders đã sẵn sàng.")

model = LSTM(
    VOCAB_SIZE,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    OUTPUT_DIM,
    N_LAYERS,
    BIDIRECTIONAL,
    DROPOUT,
    PAD_IDX
)

model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

def train_epoch(epoch, model, iterator, optimizer, criterion, device):
    """
    Hàm huấn luyện cho 1 epoch
    Args:
        model: Mô hình cần huấn luyện
        iterator: DataLoader cho tập huấn luyện
        optimizer: Bộ tối ưu
        criterion: Hàm mất mát
        device: Thiết bị (CPU/GPU)
    Returns:
        float: Loss trung bình trên epoch
    """
    epoch_loss = 0
    model.train()
    
    pbar = tqdm(iterator, desc=f"[Epoch {epoch+1:02}/{N_EPOCHS}]", leave=True)
    
    for (texts, labels, lengths) in pbar:
        texts = texts.to(device)
        labels = labels.to(device)
        
        # 1. Reset gradients
        optimizer.zero_grad()
        
        # 2. Forward pass
        # predictions shape: (batch_size, output_dim)
        predictions = model(texts, lengths)
        
        # 3. Tính loss
        loss = criterion(predictions, labels)
        
        # 4. Backward pass (tính gradient)
        loss.backward()
        
        # 5. Cập nhật trọng số
        optimizer.step()
        
        epoch_loss += loss.item()
        
        pbar.set_postfix({'loss': loss.item()})
        
    return epoch_loss / len(iterator)

print("Bắt đầu quá trình huấn luyện...")
best_test_loss = float('inf')

for epoch in range(N_EPOCHS):
    with open(LOG_FILE_PATH, "a") as log_file:
        log_file.write(f"[Epoch {epoch+1:02}/{N_EPOCHS}]: ")

    train_loss = train_epoch(epoch, model, train_loader, optimizer, criterion, device)
    
    test_loss, _, _ = evaluate_model(model, test_loader, criterion, device)

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Test Loss:  {test_loss:.4f}")
    print("-"*30)
    
    with open(LOG_FILE_PATH, "a") as log_file:
        log_file.write(f", Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}\n")

print("Huấn luyện hoàn tất.")

print("Đang tải mô hình tốt nhất để đánh giá cuối cùng...")
model.load_state_dict(torch.load(MODEL_SAVE_PATH))

evaluate_model(model, test_loader, criterion, device)