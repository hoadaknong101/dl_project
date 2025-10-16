import torch
import torch.nn as nn
import torch.optim as optim
from model import LSTM
import json
from utils import process_data

# Hyperparameters
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
LAYER_DIM = 2
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
BATCH_SIZE = 64
SEQUENCE_LENGTH = 100
DATA_PATH = 'datasets/data.csv'

def train():
    # 1. Load and process data
    train_loader, val_loader, vocab_size, output_dim, vocab_to_int = process_data(
        DATA_PATH,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH
    )

    # 2. Instantiate the model
    OUTPUT_DIM = output_dim
    model = LSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, LAYER_DIM, OUTPUT_DIM)

    # 3. Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Training on {device}")

    # 4. Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs, _, _ = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # 5. Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _, _ = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {100 * correct / total:.2f}%')

    # 6. Save the model and vocabulary
    torch.save(model.state_dict(), 'lstm_model.pth')
    with open('vocab_to_int.json', 'w') as f:
        json.dump(vocab_to_int, f)
    print("Finished Training and model saved to lstm_model.pth")
    print("Vocabulary saved to vocab_to_int.json")

if __name__ == '__main__':
    train()
