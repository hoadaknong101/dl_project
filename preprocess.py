import re
import torch
import pandas as pd
from collections import Counter
from underthesea import word_tokenize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from config import LABEL_MAP

emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"
                u"\U0001F300-\U0001F5FF" 
                u"\U0001F680-\U0001F6FF"  
                u"\U0001F1E0-\U0001F1FF"  
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u'\U00010000-\U0010ffff'
                u"\u200d"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\u3030"
                u"\ufe0f"
    "]+", flags=re.UNICODE)

def clean_text(text):
    """
    Làm sạch văn bản:
        - Đưa các ký tự về chữ thường.
        - Loại bỏ các số
        - Loại bỏ các dấu câu
        - Loại bỏ khoảng trắng thừa
        - Bỏ bớt các chữ cái giống nhau liên tiếp (Vd: Quaaaa -> qua )
        - Tách từ tiếng việt sử dụng thư viện underthesea có sẵn
        - Chuẩn hóa dữ liệu
    Args:
        text (str): Văn bản đầu vào
    Returns:
        str: Văn bản đã được làm sạch
    """
    if not isinstance(text, str):
        return ""
    
    text = text_lowercase(text)
    text = re.sub(emoji_pattern, " ", text)
    text = remove_similarletter(text)
    text = remove_number(text)
    text = remove_punctuation(text)
    text = remove_whitespace(text)
    text = remove_stopwords(text)
    text = norm_sentence(text)
    
    return text

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
        self.unk_idx = vocab.get('<unk>', 1)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 1️⃣ Làm sạch & token hóa
        cleaned = clean_text(text)
        tokens = tokenize_vietnamese(cleaned)

        # 2️⃣ Nếu sau khi làm sạch mà trống, thêm token giả
        if not tokens:
            tokens = ['<unk>']

        # 3️⃣ Chuyển token thành id
        sequence = text_to_sequence(tokens, self.vocab)

        # 4️⃣ Nếu sequence vẫn rỗng (edge case)
        if len(sequence) == 0:
            sequence = [self.unk_idx]

        # 5️⃣ Cắt bớt nếu quá dài
        if len(sequence) > self.max_seq_len:
            sequence = sequence[:self.max_seq_len]

        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def create_collate_fn(pad_idx):
    def collate_fn(batch):
        texts, labels = zip(*batch)
        lengths = torch.tensor([len(x) for x in texts])

        mask = lengths > 0
        texts = [t for t, m in zip(texts, mask) if m]
        labels = [l for l, m in zip(labels, mask) if m]
        lengths = lengths[mask]

        # Nếu batch trống, thêm 1 mẫu giả để tránh crash
        if len(texts) == 0:
            texts = [torch.tensor([pad_idx])]
            lengths = torch.tensor([1])
            labels = torch.tensor([0])
        
        texts = pad_sequence(texts, batch_first=True, padding_value=pad_idx)
        labels = torch.tensor(labels)
        
        return texts, lengths, labels

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
    
    # Chia train/test
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

    tokenized_texts = []
    for text in df['comment'].tolist():
        cleaned = clean_text(text)
        tokens = tokenize_vietnamese(cleaned)
        if len(tokens) > 0: 
            tokenized_texts.append(tokens)

    vocab = build_vocab(tokenized_texts, max_vocab_size, min_freq)
    print(f"Xây dựng từ điển hoàn tất với {len(vocab)} từ.")

    return vocab

def text_lowercase(text):
    """
    Chuyển văn bản về chữ thường.
    Args:
        text (str): Văn bản đầu vào
    Returns:
        str: Văn bản đã được chuyển về chữ thường
    """
    return text.lower()

def remove_number(text):
    """
    Xóa tất cả các chữ số khỏi văn bản.
    Args:
        text (str): Văn bản đầu vào
    Returns:
        str: Văn bản đã được xóa số
    """    
    return re.sub(r'\d+', '', text)

def remove_punctuation(text):
    """
    Xóa tất cả các dấu câu khỏi văn bản.
    Args:
        text (str): Văn bản đầu vào
    Returns:
        str: Văn bản đã được xóa dấu câu
    """
    text = text.replace(",", " ").replace(".", " ") \
    .replace(";", " ").replace("“", " ") \
    .replace(":", " ").replace("”", " ") \
    .replace('"', " ").replace("'", " ") \
    .replace("!", " ").replace("?", " ") \
    .replace("-", " ").replace("?", " ")
    return text

def remove_whitespace(text):
    """
    Xóa khoảng trắng thừa trong văn bản.
    Args:
        text (str): Văn bản đầu vào
    Returns:
        str: Văn bản đã được xóa khoảng trắng thừa
    """
    return  " ".join(text.split())

def remove_similarletter(text):
    """
    Xóa các chữ cái lặp lại liên tiếp trong văn bản.
    Args:
        text (str): Văn bản đầu vào
    Returns:
        str: Văn bản đã được xóa chữ cái lặp lại
    """
    text = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), text, flags=re.IGNORECASE)
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

def remove_stopwords(text):
    """
    Xóa các từ dừng trong văn bản tiếng Việt.
    Args:
        text (str): Văn bản đầu vào
    Returns:
        str: Văn bản đã được xóa từ dừng
    """
    file_stopwords = pd.read_csv('datasets/vietnamese-stopwords.txt', encoding = 'UTF-8')
    file_stopwords.columns = ["Stop_words"]

    stopword_list = []
    for i in file_stopwords["Stop_words"]:
        stopword_list.append(i)

    text_token = tokenize_vietnamese(text)
    result = [word for word in text_token if word not in stopword_list]

    return " ".join(result)

def norm_sentence(text):
    """
    Chuẩn hóa câu bằng cách thay thế các từ viết tắt, từ lóng
    Args:
        text (str): Câu đầu vào
    Returns:
        str: Câu đã được chuẩn hóa
    """
    replace_list = {
       ':v':'hihi', '<3':'yêu', '♥️':'yêu','❤':'yêu','a':'anh','ac':'anh chị','ace':'anh chị em','ad':'quản lý',
       'ae':'anh em','ah':'à','ak':'à','amin':'quản lý','androir':'android','app':'ứng dụng','auto ':'tự động',
       'ây':'vậy','b nào':'bạn nào','bằg':'bằng','băng':'bằng','băp':'bắp','băt':'bắt','battery':'pin','bể':'vỡ',
       'been':'bên','best':'nhất','best':'tốt nhất','bgqafy ':'ngày','bh':'bao giờ','bh':'bây giờ','bhx':'bảo hành',
       'bi':'bị','big':'lớn','bik':'biết','bin':'pin','bit':'biết','bít':'biết','bn':'bạn','bông tróc':'bong tróc', 'k': 'không', 'ok': 'được',
       'bro':'anh em','bt':'bình thường','bt':'biết','bth':'bình thường','bthg':'bình thường','bua':'bữa','bùn':'buồn',
       'buonc':'buồn','bx':'hộp','bye':'tạm biệt','c':'chị','cac':'các','cam':'máy ảnh','card':'thẻ','châu':'khỏe',
       'chiệu':'triệu','chíp':'chip','chội':'trội','chs':'chơi','chửa':'chữa','chug ':'chung','chup':'chụp','chuq':'chung',
       'clip':'đoạn phim','cmt':'bình luận','co':'có','cở':'cỡ','cọc':'cột','cpu':'chíp xử lý','cty':'công ty',
       'cua':'của','cũg':'cũng','cug ':'cũng','cuh':'cũng','cùi':'tệ','củng':'cũng','cụt':'cục','cv':'công việc',
       'cx':'cũng','đ':' đồng','dag':'đang','dăng':'văng','dấp':'lỗi','dất':'rất','đay':'đấy','đâỳ':'đầy','đc':'được',
       'dè':'rè','dể':'dễ','delay':'trễ','dêm':'đêm','đén':'đến','deplay ':'chậm','deu':'đều','diem':'điểm','dien':'diện',
       'đien':'điển','điễn':'điển','dienmayxanh':'điện máy xanh','dín':'dính','dis':'văng','diss':'văng','dk':'được',
       'dmx':'điện máy xanh','dô':'vào','dõ':'rõ','dỡ':'dở','đỗi':'đổi','download':'tải','drop':'tụt','dt':'điện thoại',
       'đt':'điện thoại','đth':'điện thoại','đthoai':'điện thoại','du':'dù','dùg':'dùng','dừg':'dừng','đứg':'đứng',
       'dụg ':'dụng','dung':'dùng','đụng':'chạm','đươc':'được','đuọc ':'được','đưowjc':'được','dựt ':'giật','dx':'được'
       ,'đx':'được','đy':'đi','e':'em','ế':'không bán được','êm':'tốt','f':'facebook','fabook':'facebook',
       'face':'facebook','fast':'nhanh','fb':'facebook','fim':'phim','fix':'sửa','flash sale':'giảm giá','fm':'đài',
       'for what':'vì sao','fps':'tốc độ khung hình','full':'đầy','future':'tương lai','game':'trò chơi','gem':'trò chơi',
       'geme':'trò chơi','gia tiên':'giá tiền','giât':'giật','giốg ':'giống','giử':'dữ','giùm':'dùm','gmae':'trò chơi',
       'gởi':'gửi','gold':'vàng','gơn':'hơn','good':'tốt','good jup':'tốt','gop':'góp','gửa':'gửi','gủng':'cái','h':'giờ',
       'haiz':'thở dài','hẵn ':'hẳn','hành':'hành','hazzz':'haizz','hc':'học','hcm':'hồ chí minh','hd':'chất lượng cao',
       'hdh':'hệ điều hành','hđh':'hệ điều hành','headphone':'tai nghe','hên':'may mắn','hẻo':'yếu','hẹo':'yếu','het':'hết',
       'hét':'hết','hic':'khóc','hieu':'hiểu','high-tech':'công nghệ cao','hít':'sử dụng','hiu':'hiểu','hỉu':'hiểu',
       'hk':'không','hn':'hà nội','hnay':'hôm nay','hoài':'nhiều lần','hoi':'hơi','hới':'hơi','hời':'tốt',
       'hoi han':'hối hận','hok':'không','hong':'không','hông':'không','hot':'nổi bật','hqua':'hôm qua','hs':'học sinh',
       'hssv':'học sinh sinh viên','hut':'hút','huway ':'huawei','huwei ':'huawei','í':'ý','I like it':'tôi thích nó',
       'ik':'đi','ip':'iphone','j':'gì','k':'không','kàm':'làm','kb':'không biết','kg':'không','kh ':'khách hàng',
       'khach':'khách hàng','khát phục':'khắc phục','khj':'khi','khoá ':'khóa','khóai ':'thích','khoẻ':'khỏe',
       'khoẽ':'khỏe','khôg':'không','khoi đong':'khởi động','khong':'không','khoong ':'không','khuân':'khuôn',
       'khủg':'khủng','kím':'kiếm','kipo':'tiêu cực','ko':'không','kt':'kiểm tra','ktra':'kiểm tra','la':'là',
       'lác':'lỗi','lắc':'lỗi','lag':'lỗi','laii':'lại','lak':'giật','lan':'lần','lãng':'giật','lap':'máy tính',
       'laptop':'máy tính','lay':'này','len toi':'lên tới','les':'led','lg':'lượng','lí':'lý','lien':'liên',
       'like':'thích','liti':'nhỏ','live stream':'phát sóng trực tiếp','lm':'làm','ln':'luôn','loadd':'tải ',
       'lôi':'lỗi','lổi':'lỗi','LOL ':'trò chơi','lởm':'kém chất lượng','lỏng lẽo':'lỏng lẻo','luc':'lúc','lun':'luôn',
       'luong':'lượng','luot':'lướt','lưot ':'lượt','m':'mình','mạ':'trời','mắc công':'mất công','macseger':'messenger',
       'mag':'màn','main':'chính','mak':'mà','man':'màn','màng':'màn','màng hình':'màn hình','mao ':'mau','mẩu':'mẫu',
       'mầu ':'màu','max':'lớn nhất','may':'máy','mèn':'màn','méo gì':'làm gì','mih':'mình','mìk':'mình','min':'nhỏ nhât',
       'mìn':'mình','mjh':'mình','mjk':'mình','mjnh':'minh','mk':'mình','mn':'mọi người','mng ':'mọi người','mo':'đâu',
       'mò':'tìm','mobile':'điện thoại','mog':'mong','moi':'mới','mơi':'mới','ms':'mới','mún':'muốn','mước':'mức',
       'mược':'mượt','muot':'mượt','mỷ':'mỹ','n':'nó','n':'nói chuyện','nãn':'nản','nayd':'này','nc':'nói chuyện',
       'nch':'nói chuyện','nch':'nói chung','nếo ':'nếu','ng':'người','ngan':'ngang','nge':'nghe','nghiêm':'nghiệm',
       'ngĩ':'nghĩ','ngốn':'sử dụng','nguon':'nguồn','nhah':'nhanh','nhan vien':'nhân viên','nhay':'nhạy','nhe':'nhé',
       'nhèo':'nhòe','nhiet':'nhiệt','nhiểu':'nhiều','nhiu':'nhiều','nhìu':'nhiều','nhoè':'nhòe','như v':'như vậy',
       'nhug':'nhưng','nhưg':'nhưng','nhữg':'những','nhung':'nhưng','nhuoc':'nhược','nhượt':'nhược','nock ao':'hạ gục',
       'noi':'nói','nống':'nóng','not':'lưu ý','ns ':'nói','nsx':'ngày sản xuất','nt':'nhắn tin','ntin':'nhắn tin',
       'ntn':'như thế nào','nũa':'nữa','nut ':'nút','nv':'nhân viên','nz':'như vậy','ô xi':'oxy','ofice':'văn phòng',
       'ok':'được','ôk':'được','oke':'được','okee':'được','oki':'được','okie':'được','onl':'sử dụng',
       'ộp ẹp':'không chắc chắn','option':'tùy chọn','or':'hoặc','out':'thoát','oỳ':'rồi','pải':'phải','phảm':'phẩm',
       'phẩn':'phẩm','phan van':'phân vân','phèo':'vậy','phut ':'phút','pít':'biết','pro':'chất lượng cao','pùn':'buồn',
       'pv':'giới thiệu','qá':'quá','qc':'quảng cáo','qtv':'quản trị viên','qua ve':'qua vẻ','quang trọng':'quan trọng',
       'qus':'quá','r ':'rồi','rat':'rất','rát':'rất','rắt':'rất','rata':'rất','rễ':'dễ','rep':'trả lời',
       'research':'nghiên cứu','reset':'cài đặt lại','restart':'khởi động lại','review':'đánh giá','rì':'gì',
       'rinh':'mua','rỏ':'rõ','rùi':'rồi','rùng':'dùng','s':'sao','sac':'sạc','sài':'xài','sài':'dùng','sale':'giảm giá',
       'sale off':'giảm giá','sâng':'sáng','sạt':'sạc','saving':'tiết kiệm','sd':'sử dụng','sdt':'số điện thoại',
       'seal':'mới','search':'tìm kiếm','sefil':'chụp ảnh','selfie':'chụp ảnh','setting':'cài đặt','setup':'cài đặt',
       'sexy':'quyến rũ','shiper':'nhân viên giao hàng','shop':'cửa hàng','skill':'kỹ năng','smooth':'mượt',
       'so good':'rất tốt','sp':'sản phẩm','sphẩm':'sản phẩm','stars':'sao','sử':'xử','suất':'xuất','sưj':'sự',
       'sước':'xước','super':'siêu','support':'hỗ trợ','sụt':'tụt','sv':'sinh viên','sx':'sản xuất','t':'tôi',
       'T G D Đ':'thế giới di động','tằm ':'tầm','tes':'kiểm tra','test':'kiểm tra','tet':'tết','teung':'trung',
       'tg':'thời gian','tgdd':'thế giới di động','tgdđ':'thế giới di động','thag':'tháng','thág':'tháng','ship':'giao','Ship':'giao',
    }
    text = text.split()
    len_ = len(text)

    for i in range(0, len_):
        for k, v in replace_list.items():
            if (text[i]==k):
                text[i] = v
    
    return " ".join(text)