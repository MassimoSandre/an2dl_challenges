import os
import cv2
import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.cuda.amp import autocast, GradScaler

# ==========================================
# 1. CONFIGURAZIONE
# ==========================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True # Velocizza un po'

IMG_SIZE = 384
BATCH_SIZE = 16
NUM_CLASSES = 4
N_FOLDS = 5     # Numero di Fold per la Cross-Validation
EPOCHS = 20     # Epoche per fold (meno di prima perché facciamo fine tuning subito)
LR = 1e-4       # Learning rate medio

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.getcwd()
TRAIN_DIR = os.path.join(BASE_DIR, 'train_data')
TEST_DIR = os.path.join(BASE_DIR, 'test_data')
LABELS_FILE = os.path.join(BASE_DIR, 'train_labels.csv')

print(f"🔧 Configurazione: {N_FOLDS}-Fold CV su {DEVICE}")

# ==========================================
# 2. DATASET E UTILS
# ==========================================
def smart_crop_and_resize(img_path, mask_path, target_size=(384,384), margin=0.15):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None: return None
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    pts = cv2.findNonZero(binary_mask)
    h,w = img.shape[:2]
    if pts is not None:
        x,y,ww,hh = cv2.boundingRect(pts)
        # expand with margin
        pad_x = int(ww * margin)
        pad_y = int(hh * margin)
        x0 = max(0, x - pad_x); y0 = max(0, y - pad_y)
        x1 = min(w, x + ww + pad_x); y1 = min(h, y + hh + pad_y)
        crop = img[y0:y1, x0:x1]
        # if crop too small -> fallback to central crop of original
        if crop.shape[0] < 32 or crop.shape[1] < 32:
            min_side = min(h, w)
            startx = (w - min_side)//2; starty = (h - min_side)//2
            crop = img[starty:starty+min_side, startx:startx+min_side]
    else:
        # fallback to center crop
        min_side = min(h, w)
        startx = (w - min_side)//2; starty = (h - min_side)//2
        crop = img[starty:starty+min_side, startx:startx+min_side]
    resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return resized


class TissueDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform: img = self.transform(img)
        if self.labels is not None: return img, self.labels[idx]
        else: return img

def load_data_to_ram(csv_path, img_dir, is_test=False):
    if not is_test:
        df = pd.read_csv(csv_path)
        files = df['sample_index'].values
        labels_str = df['label'].values
    else:
        files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png') and 'mask' not in f])
        labels_str = None
    images_list = []
    valid_indices = []
    print(f"📥 Caricamento {'Test' if is_test else 'Train'} set...")
    for i, filename in enumerate(files):
        img_path = os.path.join(img_dir, filename)
        mask_name = filename.replace('img_', 'mask_')
        mask_path = os.path.join(img_dir, mask_name)
        try:
            processed_img = smart_crop_and_resize(img_path, mask_path, (IMG_SIZE, IMG_SIZE))
            if processed_img is not None:
                images_list.append(processed_img)
                valid_indices.append(i)
        except Exception: pass
    X = np.array(images_list, dtype='uint8')
    if not is_test: return X, labels_str[valid_indices], None
    else: return X, None, [files[i] for i in valid_indices]

# ==========================================
# 3. DEFINIZIONE MODELLO
# ==========================================
# Se non hai timm, usa ResNet50 che è standard in torchvision
def build_model(num_classes):
    # ResNet50 è un "trattore": robusta e ottima per le texture
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    
    # Sblocca subito gli ultimi layer (il Fine Tuning serve subito qui)
    for param in model.parameters():
        param.requires_grad = False
    
    # Sblocca l'ultimo blocco convoluzionale (layer4)
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    # Cambia la testa
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5), # Dropout alto perché abbiamo pochi dati
        nn.Linear(num_ftrs, num_classes)
    )
    return model
# ==========================================
# 4. TRAINING LOOP CON CLASS WEIGHTS
# ==========================================
scaler = GradScaler()

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE).long()
        
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total

def validate_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).long()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total

# ==========================================
# 5. DATA PREPARATION
# ==========================================
X_all, y_all_str, _ = load_data_to_ram(LABELS_FILE, TRAIN_DIR)
le = LabelEncoder()
y_all_enc = le.fit_transform(y_all_str)

# Calcolo pesi delle classi (IMPORTANTISSIMO PER BILANCIARE)
class_weights = compute_class_weight('balanced', classes=np.unique(y_all_enc), y=y_all_enc)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
print(f"⚖️ Class Weights: {class_weights}")

# Trasformazioni
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    #☺transforms.ColorJitter(brightness=0.1, contrast=0.1), # Leggero jitter colore
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 6. CROSS VALIDATION LOOP (5 FOLDS)
# ==========================================
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_results = []
models_list = [] # Salviamo i modelli per l'ensemble finale

print(f"\n🚀 Inizio Cross-Validation ({N_FOLDS} folds)...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all_enc)):
    print(f"\n--- FOLD {fold+1}/{N_FOLDS} ---")
    
    # Dataset per questo fold
    X_train_fold, X_val_fold = X_all[train_idx], X_all[val_idx]
    y_train_fold, y_val_fold = y_all_enc[train_idx], y_all_enc[val_idx]
    
    train_ds = TissueDataset(X_train_fold, y_train_fold, transform=train_transforms)
    val_ds = TissueDataset(X_val_fold, y_val_fold, transform=val_transforms)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Init Modello
    model = build_model(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor) # Loss pesata!
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4) # AdamW è meglio
    
    # Training
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    
    for epoch in range(EPOCHS):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc = validate_epoch(model, val_loader, criterion)
        
        # Salva se migliora
        if v_acc > best_acc:
            best_acc = v_acc
            best_weights = copy.deepcopy(model.state_dict())
        
        print(f"Ep {epoch+1:02d} | Train: {t_acc:.3f} | Val: {v_acc:.3f}")
        
    print(f"🏆 Best Val Acc Fold {fold+1}: {best_acc:.3f}")
    
    # Ricarica i pesi migliori e salva il modello in lista
    model.load_state_dict(best_weights)
    models_list.append(model)
    fold_results.append(best_acc)

print(f"\n📈 Media Accuracy Validation: {np.mean(fold_results):.4f}")

# ==========================================
# 7. ENSEMBLE PREDICTION (Test Time Augmentation)
# ==========================================
print("\n🔮 Generazione Ensemble Submission con TTA...")

X_test, _, test_filenames = load_data_to_ram(None, TEST_DIR, is_test=True)
test_ds = TissueDataset(X_test, labels=None, transform=val_transforms)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

final_probs = np.zeros((len(X_test), NUM_CLASSES))

with torch.no_grad():
    for model in models_list:
        model.eval()
        model_probs = []
        for inputs in test_loader:
            inputs = inputs.to(DEVICE)
            
            # TTA: Predici su Originale + Flip H + Flip V
            out1 = torch.softmax(model(inputs), dim=1)
            out2 = torch.softmax(model(transforms.functional.hflip(inputs)), dim=1)
            out3 = torch.softmax(model(transforms.functional.vflip(inputs)), dim=1)
            
            # Media delle 3 predizioni
            avg_out = (out1 + out2 + out3) / 3.0
            model_probs.extend(avg_out.cpu().numpy())
        
        # Somma le probabilità di questo modello al totale
        final_probs += np.array(model_probs)

# Media finale su tutti i modelli
final_probs /= N_FOLDS
final_preds = np.argmax(final_probs, axis=1)

# Submission
pred_labels_str = le.inverse_transform(final_preds)
submission_df = pd.DataFrame({'sample_index': test_filenames, 'label': pred_labels_str})
submission_df.to_csv('submission_ensemble_cv.csv', index=False)
print("🎉 Submission Ensemble salvata: submission_ensemble_cv.csv")