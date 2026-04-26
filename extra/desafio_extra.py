import os
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

DATA_DIR = r'C:\Users\jefer\Documents\3_POS\extra'
BASE_DIR = os.path.join(DATA_DIR, 'CBIS-DDSM')
CSV_DIR  = os.path.join(BASE_DIR, 'csv')

IMG_SIZE   = (224, 224)
BATCH_SIZE = 16
EPOCHS     = 20

#verificar se tem gpu, se tiver usar, se não usar cpu
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(file_path):
    """carrega arquivo CSV; return um DataFrame."""
    return pd.read_csv(file_path)


def preprocess_image(image_path):
    """Carrega e resize image; return array RGB ou None se flahar."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, IMG_SIZE)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def build_label_image_mapping():
    """Vincule arquivos CSV de descrição de casos a caminhos JPEG por meio da SeriesInstanceUID.
        Os caminhos de descrição de casos codificam o UID da série como o penúltimo componente do caminho; o arquivo dicom_info.csv registra o mesmo UID em SeriesInstanceUID.
    """
    dicom_info = pd.read_csv(os.path.join(CSV_DIR, 'dicom_info.csv'))
    # 'Imagens recortadas' são recortes da região de interesse (ROI) que contêm a lesão.
    recortada = dicom_info[dicom_info['SeriesDescription'] == 'cropped images']
    series_to_jpeg = dict(zip(recortada['SeriesInstanceUID'], recortada['image_path']))

    case_csvs = [
        ('mass_case_description_train_set.csv', 'train'),
        ('mass_case_description_test_set.csv',  'test'),
        ('calc_case_description_train_set.csv', 'train'),
        ('calc_case_description_test_set.csv',  'test'),
    ]

    records = []
    for csv_name, split in case_csvs:
        df = pd.read_csv(os.path.join(CSV_DIR, csv_name))
        for _, row in df.iterrows():
            path_str  = str(row.get('cropped image file path', '')).strip().replace('\\', '/')
            parts     = [p for p in path_str.split('/') if p]
            if len(parts) < 3:
                continue
            series_uid = parts[-2]
            if series_uid not in series_to_jpeg:
                continue
            full_path = os.path.join(DATA_DIR, series_to_jpeg[series_uid])
            label     = 1 if str(row['pathology']).strip() == 'MALIGNANT' else 0
            records.append({'image_path': full_path, 'label': label, 'split': split})

    df_out = pd.DataFrame(records).drop_duplicates(subset='image_path')
    print(f"Amostras mapeadas : {len(df_out)}")
    print(f"  Malignos         : {df_out['label'].sum()}")
    print(f"  Benignos         : {(df_out['label'] == 0).sum()}")
    return df_out


def load_images(df):
    images, labels = [], []
    failed = 0
    for _, row in df.iterrows():
        img = preprocess_image(row['image_path'])
        if img is not None:
            images.append(img)
            labels.append(row['label'])
        else:
            failed += 1
    if failed:
        print(f"Imagens não carregadas: {failed}")
    return images, np.array(labels, dtype=np.int64)


# ── Dataset ───────────────────────────────────────────────────────────────────

# As estatísticas do ImageNet foram usadas porque o EfficientNetB0 foi pré-treinado com base nelas.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

class MammographyDataset(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images = images
        self.labels = labels

        aug = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2),
            transforms.RandomResizedCrop(IMG_SIZE[0], scale=(0.9, 1.0)),
        ] if augment else []

        self.transform = transforms.Compose(
            aug + [transforms.ToTensor(),
                   transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        tensor = self.transform(Image.fromarray(self.images[idx]))
        label  = torch.tensor(self.labels[idx], dtype=torch.float32)
        return tensor, label


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model():
    """EfficientNetB0 com base congelada e cabeçalho de classificação binária personalizado.

Os logits (sem sigmoide) são retornados, permitindo o uso de BCEWithLogitsLoss, que
é numericamente mais estável que BCELoss + sigmoide.
    """
    net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    for param in net.parameters():
        param.requires_grad = False

    in_features = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),   # sigmoide aplicada no momento da inferência.
    )
    return net.to(DEVICE)


def _pos_weight(y):
    neg, pos = np.bincount(y)
    # para compensar o desequilíbrio de classes.
    return torch.tensor([neg / pos], dtype=torch.float32).to(DEVICE)


# ── Ajudantes de treinamento ──────────────────────────────────────────────────────────

def _train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X).squeeze(1), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


def _val_epoch(model, loader, criterion):
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X).squeeze(1)
            total_loss += criterion(logits, y).item() * len(y)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(y.cpu().numpy())
    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    return total_loss / len(loader.dataset), auc


def _run_phase(model, train_dl, val_dl, criterion, lr, max_epochs, patience, label):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, min_lr=1e-6)
    best_auc, no_improve = 0.0, 0

    print(f"\n── {label} ──")
    for epoch in range(max_epochs):
        train_loss = _train_epoch(model, train_dl, criterion, optimizer)
        val_loss, val_auc = _val_epoch(model, val_dl, criterion)
        scheduler.step(val_auc)
        print(f"Epoch {epoch+1:02d} | train={train_loss:.4f} | val={val_loss:.4f} | auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'best_model.pt')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load('best_model.pt', map_location=DEVICE))
    return model


def train_model(model, X_train, y_train, X_val, y_val):
    """Treinamento em duas fases: primeiro, classificação apenas -- ajuste fino dos últimos blocos de características."""
    criterion = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(y_train))

    train_dl = DataLoader(MammographyDataset(X_train, y_train, augment=True),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(MammographyDataset(X_val,   y_val,   augment=False),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Fase 1: apenas o novo cabeçalho de classificação é treinável.
    model = _run_phase(model, train_dl, val_dl, criterion,
                       lr=1e-3, max_epochs=10, patience=5,
                       label="Fase 1: treinando cabeça")

    # Fase 2: usar os últimos 3 blocos de características para ajuste fino.
    for block in list(model.features.children())[-3:]:
        for param in block.parameters():
            param.requires_grad = True

    model = _run_phase(model, train_dl, val_dl, criterion,
                       lr=1e-5, max_epochs=EPOCHS, patience=5,
                       label="Fase 2: fine-tuning")
    return model


# ── Avaliação e inferência ────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test):
    test_dl = DataLoader(MammographyDataset(X_test, y_test, augment=False),
                         batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X, y in test_dl:
            probs = torch.sigmoid(model(X.to(DEVICE))).squeeze(1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels).astype(int)
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n=== Avaliação no conjunto de teste ===")
    print(classification_report(y_true, y_pred, target_names=['Benigno', 'Maligno']))
    print(f"AUC-ROC: {roc_auc_score(y_true, y_prob):.4f}")
    return y_prob


def predict_cancer(model, image_path):
    """Predict malignancy for a single mammography image.

    Returns a dict with probability, label, and confidence,
    or None if the image cannot be read.
    """
    raw = preprocess_image(image_path)
    if raw is None:
        print(f"Erro: imagem não encontrada → {image_path}")
        return None

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    tensor = tfm(Image.fromarray(raw)).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        prob = float(torch.sigmoid(model(tensor)).squeeze())

    label = 'MALIGNO' if prob >= 0.5 else 'BENIGNO'
    conf  = max(prob, 1 - prob) * 100

    result = {
        'image_path':            image_path,
        'probabilidade_maligno': prob,
        'classificacao':         label,
        'confianca':             f"{conf:.1f}%",
    }
    print(f"\n=== Predição ===")
    print(f"Imagem        : {os.path.basename(image_path)}")
    print(f"Classificação : {label}")
    print(f"P(maligno)    : {prob:.4f}")
    print(f"Confiança     : {result['confianca']}")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Dispositivo: {DEVICE}")

    print("Mapeando imagens e rótulos...")
    df = build_label_image_mapping()
    if df.empty:
        print("Nenhuma imagem mapeada. Verifique os caminhos.")
        return

    print("Carregando imagens...")
    images, labels = load_images(df)
    print(f"Dataset: {len(images)} imagens")

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.20, stratify=labels, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )
    print(f"Treino: {len(X_train)} | Val: {len(X_val)} | Teste: {len(X_test)}")

    model = build_model()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parâmetros treináveis (fase 1): {trainable:,}")

    model = train_model(model, X_train, y_train, X_val, y_val)

    evaluate_model(model, X_test, y_test)

    torch.save(model.state_dict(), 'breast_cancer_model.pt')
    print("\nModelo salvo em: breast_cancer_model.pt")

    # Exemplo de previsão de imagem única (substitua por um caminho real)
    # predict_cancer(model, r'C:\...\imagem.jpg')


if __name__ == "__main__":
    main()
