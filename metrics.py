import torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from config import LABEL_INV_MAP, LOG_FILE_PATH
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def calculate_metrics(y_true, y_pred, target_names):
    """
    Tính toán và in ra accuracy, precision, recall, F1-score.
    Args:
        y_true (list): Danh sách nhãn thực tế
        y_pred (list): Danh sách nhãn dự đoán
        target_names (list): Tên các lớp
    """
    accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true, 
        y_pred, 
        target_names=target_names, 
        zero_division=0,
        output_dict=True
    )

    macro = report["macro avg"]
    weighted = report["weighted avg"]
    acc = report["accuracy"]

    # Ghép lại thành một chuỗi
    line = (
        f"accuracy: {acc:.4f}, "
        f"macro_precision: {macro['precision']:.4f}, "
        f"macro_recall: {macro['recall']:.4f}, "
        f"macro_f1: {macro['f1-score']:.4f}, "
        f"weighted_precision: {weighted['precision']:.4f}, "
        f"weighted_recall: {weighted['recall']:.4f}, "
        f"weighted_f1: {weighted['f1-score']:.4f}"
    )

    print(report)
    print(f"Accuracy: {accuracy:.4f}")
    
    with open(LOG_FILE_PATH, "a") as log_file:
        log_file.write(line)

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Vẽ và hiển thị confusion matrix.
    Args:
        y_true (list): Danh sách nhãn thực tế
        y_pred (list): Danh sách nhãn dự đoán
        class_names (list): Tên các lớp
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('Nhãn thực tế (True Label)')
    plt.xlabel('Nhãn dự đoán (Predicted Label)')
    
    plt.savefig('saved_model/confusion_matrix.png')


def evaluate_model(model, iterator, criterion, device):
    """
    Hàm tổng quát để đánh giá mô hình trên tập test/validation.
    Args:
        model: Mô hình cần đánh giá
        iterator: DataLoader cho tập test/validation
        criterion: Hàm mất mát
        device: Thiết bị (CPU/GPU)
    Returns:
        float: Loss trung bình trên tập đánh giá
        list: Danh sách các nhãn dự đoán
        list: Danh sách các nhãn thực tế
    """
    epoch_loss = 0
    
    all_preds = []
    all_labels = []
    
    model.eval()
    
    with torch.no_grad():
        pbar = tqdm(iterator, desc="[Evaluating]", leave=False)

        for texts, labels, lengths in pbar:
            texts = texts.to(device)
            labels = labels.to(device)
            
            # Forward pass
            predictions = model(texts, lengths)
            
            # Tính loss
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()
            
            preds = torch.argmax(predictions, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': loss.item()})
            
    avg_loss = epoch_loss / len(iterator)
    
    target_names = [LABEL_INV_MAP[i] for i in range(len(LABEL_INV_MAP))]
    
    calculate_metrics(all_labels, all_preds, target_names)
    
    plot_confusion_matrix(all_labels, all_preds, target_names)
    
    return avg_loss, all_preds, all_labels