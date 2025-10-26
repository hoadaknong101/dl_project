import torch
import pickle
from model import LSTM
from flask import Flask, request, jsonify, render_template
from preprocess import clean_text, tokenize_vietnamese, text_to_sequence
from config import (EMBEDDING_DIM, 
                    HIDDEN_DIM,
                    OUTPUT_DIM, 
                    N_LAYERS,
                    BIDIRECTIONAL,
                    DROPOUT, 
                    MAX_SEQ_LEN,
                    MODEL_PATH,
                    VOCAB_PATH,
                    LABEL_INV_MAP,
                    SENTIMENT_MAP)

print("Đang tải từ điển (vocab)...")
with open(VOCAB_PATH, 'rb') as f:
    vocab = pickle.load(f)
print("Tải từ điển hoàn tất.")

VOCAB_SIZE = len(vocab)
PAD_IDX = vocab['<pad>']

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

print("Đang tải trọng số mô hình...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("Mô hình đã sẵn sàng.")

def predict_sentiment(text):
    """
    Hàm nhận văn bản thô và trả về nhãn dự đoán (POS, NEG, NEU)
    """
    model.eval()
    
    # 1. Tiền xử lý văn bản
    cleaned_text = clean_text(text)
    tokens = tokenize_vietnamese(cleaned_text)
    sequence = text_to_sequence(tokens, vocab)
    
    # 2. Cắt/Pad (mặc dù chỉ có 1 câu, vẫn cần chuẩn hóa độ dài)
    if len(sequence) > MAX_SEQ_LEN:
        sequence = sequence[:MAX_SEQ_LEN]
    
    # 3. Chuyển sang Tensor
    sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(device) # Thêm batch dimension [1, seq_len]
    lengths_tensor = torch.tensor([len(sequence)], dtype=torch.long)
    
    # 4. Dự đoán
    with torch.no_grad():
        predictions = model(sequence_tensor, lengths_tensor)
        
    # 5. Lấy kết quả
    pred_idx = torch.argmax(predictions, dim=1).item()
    pred_label = LABEL_INV_MAP[pred_idx]
    
    sentiment_info = SENTIMENT_MAP[pred_label]
    score = torch.softmax(predictions, dim=1)[0][pred_idx].item()
    
    return {
        'language': "Tiếng Việt",
        'sentiment': sentiment_info["label"],
        'label': sentiment_info["label"],
        'score': round(score * 100, 2),
        'emoji': sentiment_info["emoji"],
        'color': sentiment_info["color"]
    }

app = Flask(__name__)

@app.route('/')
def index():
    """
    Render trang chủ của ứng dụng.
    """
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def handle_predict():
    """
    Endpoint API nhận văn bản và trả về kết quả phân tích tình cảm.
    """
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text.strip():
            return jsonify({'error': 'Vui lòng nhập văn bản để phân tích.'}), 400
        
        prediction = predict_sentiment(text)
        print(prediction)
        return jsonify(prediction)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)