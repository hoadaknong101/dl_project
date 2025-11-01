import os
from datetime import datetime

# --- Cấu hình mô hình LSTM ---
EMBEDDING_DIM = 300
"""Kích thước của vector embedding cho mỗi từ."""

HIDDEN_DIM = 128
"""Kích thước của lớp ẩn trong mô hình LSTM."""

OUTPUT_DIM = 3
"""Số lượng lớp đầu ra (tương ứng với 3 loại cảm xúc: POS, NEG, NEU)."""

N_LAYERS = 3
"""Số lượng lớp LSTM xếp chồng lên nhau."""

BIDIRECTIONAL = False
"""Cờ xác định có sử dụng LSTM hai chiều (bidirectional) hay không."""

DROPOUT = 0.5
"""Tỷ lệ dropout được áp dụng giữa các lớp LSTM để tránh overfitting."""

MAX_SEQ_LEN = 100
"""Độ dài tối đa của một câu. Các câu dài hơn sẽ bị cắt ngắn."""

# --- Cấu hình quá trình huấn luyện ---
BATCH_SIZE = 2048
"""Số lượng mẫu dữ liệu được xử lý trong một lần lặp huấn luyện."""

N_EPOCHS = 50
"""Tổng số lần lặp qua toàn bộ tập dữ liệu huấn luyện."""

LEARNING_RATE = 1e-3
"""Tốc độ học của optimizer (Adam)."""

# --- Cấu hình tiền xử lý dữ liệu ---
MAX_VOCAB_SIZE = 20000
"""Kích thước tối đa của từ điển. Các từ ít phổ biến hơn sẽ bị loại bỏ."""

MIN_FREQ = 1
"""Tần suất xuất hiện tối thiểu của một từ để được đưa vào từ điển."""

# --- Cấu hình đường dẫn file ---
SAVE_DIR = 'saved_model'
"""Thư mục để lưu các file của mô hình (từ điển, trọng số)."""

DATA_FILE = 'datasets/data - data.csv'
"""Đường dẫn đến file CSV chứa dữ liệu huấn luyện và kiểm thử."""

VOCAB_PATH = os.path.join(SAVE_DIR, 'vocab.pkl')
"""Đường dẫn đầy đủ để lưu/tải file từ điển (.pkl)."""

MODEL_PATH = os.path.join(SAVE_DIR, 'lstm_model.pth')
"""Đường dẫn đầy đủ để lưu/tải file trọng số mô hình (.pth)."""

VOCAB_SAVE_PATH = os.path.join(SAVE_DIR, 'vocab.pkl')
"""Đường dẫn lưu file từ điển (sử dụng trong train.py)."""

MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'lstm_model.pth')
"""Đường dẫn lưu file mô hình (sử dụng trong train.py)."""

# --- Cấu hình nhãn ---
LABEL_MAP = {'POS': 0, 'NEG': 1, 'NEU': 2}
"""Ánh xạ từ nhãn dạng chuỗi (string) sang nhãn dạng số (integer)."""

LABEL_INV_MAP = {v: k for k, v in LABEL_MAP.items()}
"""Ánh xạ ngược từ nhãn dạng số về dạng chuỗi, hữu ích cho việc diễn giải kết quả."""

# --- Cấu hình checkpoint ---
RUN_CHECKPOINT_PATH = os.path.join("runs", datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
"""Thư mục gốc để lưu các checkpoint của các lần chạy (runs)."""

LOG_FILE_PATH = os.path.join(RUN_CHECKPOINT_PATH, "training_log.txt")
"""Đường dẫn đến file log ghi lại quá trình huấn luyện."""

# Đảm bảo thư mục lưu trữ tồn tại
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RUN_CHECKPOINT_PATH, exist_ok=True)

SENTIMENT_MAP = {
    "POS": {"label": "Tích cực", "emoji": "😊", "color": "bg-green-100 text-green-800 border-green-400"},
    "NEG": {"label": "Tiêu cực", "emoji": "😠", "color": "bg-red-100 text-red-800 border-red-400"},
    "NEU": {"label": "Trung tính", "emoji": "😐", "color": "bg-blue-100 text-blue-800 border-blue-400"},
}
"""Bản đồ ánh xạ nhãn cảm xúc sang nhãn hiển thị, biểu tượng cảm xúc và màu sắc tương ứng."""