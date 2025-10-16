import torch
import pandas as pd
import numpy as np
import re
from torch.utils.data import TensorDataset, DataLoader, random_split
from collections import Counter

def process_data(file_path, batch_size=32, sequence_length=100):
    """
    Processes the raw CSV data for LSTM model training.

    Args:
        file_path (str): The path to the CSV file.
        batch_size (int): The batch size for the DataLoader.
        sequence_length (int): The fixed length for input sequences.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training set.
            - val_loader (DataLoader): DataLoader for the validation set.
            - vocab_size (int): The size of the vocabulary.
            - output_dim (int): The number of unique labels.
    """
    # 1. Load Data
    df = pd.read_csv(file_path)

    # 2. Clean Text and Tokenize
    def clean_text(text):
        text = str(text)
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    df['tokenized'] = df['content'].apply(clean_text)

    # 3. Build Vocabulary
    all_words = [word for tokens in df['tokenized'] for word in tokens]
    word_counts = Counter(all_words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    vocab_to_int = {word: i + 1 for i, word in enumerate(sorted_vocab)}
    vocab_size = len(vocab_to_int) + 1  # +1 for padding

    # 4. Numericalize
    df['numerical'] = df['tokenized'].apply(lambda x: [vocab_to_int.get(word, 0) for word in x])

    # 5. Pad Sequences
    def pad_features(reviews_int, seq_length):
        features = np.zeros((len(reviews_int), seq_length), dtype=int)
        for i, row in enumerate(reviews_int):
            if len(row) != 0:
                features[i, -len(row):] = np.array(row)[:seq_length]
        return features

    features = pad_features(df['numerical'].tolist(), sequence_length)

    # 6. Label Encoding
    labels = df['label'].astype('category').cat.codes.to_numpy()
    output_dim = len(df['label'].unique())

    # 7. Create TensorDataset
    data = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels).long())

    # 8. Split Data
    val_size = int(len(data) * 0.2)
    train_size = len(data) - val_size
    train_dataset, val_dataset = random_split(data, [train_size, val_size])

    # 9. Create DataLoaders
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)

    return train_loader, val_loader, vocab_size, output_dim, vocab_to_int
