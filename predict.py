import torch
import json
import re
import numpy as np
from model import LSTM
import pandas as pd

# --- Configuration ---
MODEL_PATH = 'lstm_model.pth'
VOCAB_PATH = 'vocab_to_int.json'
DATA_PATH = 'datasets/data.csv' # Needed to map predicted index back to label name

# --- Hyperparameters (must match the training configuration) ---
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
LAYER_DIM = 2
SEQUENCE_LENGTH = 100

def predict(text):
    """
    Loads the trained model and predicts the category for a given text.

    Args:
        text (str): The input text string for prediction.

    Returns:
        str: The predicted label name.
    """
    # --- 1. Load Artifacts ---
    # Load vocabulary
    with open(VOCAB_PATH, 'r') as f:
        vocab_to_int = json.load(f)
    vocab_size = len(vocab_to_int) + 1 # +1 for padding token

    # Load label mapping
    df = pd.read_csv(DATA_PATH)
    # Create a mapping from category codes (0, 1, ...) to label names
    label_map = dict(enumerate(df['label'].astype('category').cat.categories))
    output_dim = len(label_map)

    # --- 2. Instantiate the Model ---
    model = LSTM(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        layer_dim=LAYER_DIM,
        output_dim=output_dim
    )

    # --- 3. Load the Trained State ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval() # Set model to evaluation mode

    # --- 4. Preprocess the Input Text ---
    # Same cleaning and tokenization as in utils.py
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokenized = text.split()

    # Numericalize
    numericalized = [vocab_to_int.get(word, 0) for word in tokenized]

    # Pad/Truncate
    features = np.zeros((1, SEQUENCE_LENGTH), dtype=int)
    if len(numericalized) != 0:
        features[0, -len(numericalized):] = np.array(numericalized)[:SEQUENCE_LENGTH]

    # Convert to tensor
    input_tensor = torch.from_numpy(features).to(device)

    # --- 5. Make Prediction ---
    with torch.no_grad():
        output, _, _ = model(input_tensor)
        _, predicted_idx = torch.max(output.data, 1)

    # --- 6. Map Prediction to Label ---
    predicted_label = label_map[predicted_idx.item()]

    return predicted_label

if __name__ == '__main__':
    # --- Example Usage ---
    # Replace this with any text you want to classify
    sample_text_1 = "This movie was absolutely fantastic! The acting was superb and the plot was gripping."
    sample_text_2 = "I was really disappointed. The story was boring and it felt way too long."

    print(f"Input Text: '{sample_text_1}'")
    prediction_1 = predict(sample_text_1)
    print(f"Predicted Category: {prediction_1}\n")

    print(f"Input Text: '{sample_text_2}'")
    prediction_2 = predict(sample_text_2)
    print(f"Predicted Category: {prediction_2}")