import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import os
import time
import json

# ------------------- НАСТРОЙКИ -------------------
DATA_DIR = "datasets/WIDER_train/images"
ANNOT_FILE = "datasets/WIDER_train/annotations/wider_face_train_bbx_gt.txt"
BATCH_SIZE = 1          # Уменьши до 2, если мало видеопамяти
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "models/face_detector.pth"

print(f"Устройство: {DEVICE}")
print(f"Обучение на {EPOCHS} эпох с batch_size = {BATCH_SIZE}")

# ------------------- ПАРСИНГ АННОТАЦИЙ -------------------
def parse_annotations():
    annotations = {}
    with open(ANNOT_FILE, 'r') as f:
        lines = f.readlines()
    i = 0
    total_faces = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.endswith('.jpg'):
            i += 1
            continue
        img_path = line
        i += 1
        num_faces = int(lines[i].strip())
        i += 1
        boxes = []
        for _ in range(num_faces):
            parts = lines[i].strip().split()
            if len(parts) >= 4:
                x, y, w, h = map(int, parts[:4])
                if w > 0 and h > 0:
                    boxes.append([x, y, x + w, y + h])  # [xmin, ymin, xmax, ymax]
            i += 1
        if boxes:
            annotations[img_path] = boxes
            total_faces += len(boxes)
    print(f"Загружено изображений: {len(annotations)}")
    print(f"Всего лиц в датасете: {total_faces}")
    return annotations

annotations = parse_annotations()
image_paths = list(annotations.keys())

# ------------------- ДАТАСЕТ -------------------
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((640, 640)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class WIDERDataset(Dataset):
    def __init__(self, img_paths, annotations, root_dir, transform=None):
        self.img_paths = img_paths
        self.annotations = annotations
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        rel_path = self.img_paths[idx]
        img_path = os.path.join(self.root_dir, rel_path)
        image = Image.open(img_path).convert("RGB")
        boxes = torch.tensor(self.annotations[rel_path], dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            image = self.transform(image)

        return image, target

dataset = WIDERDataset(image_paths, annotations, DATA_DIR, transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

# ------------------- МОДЕЛЬ -------------------
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)  # background + face

model.to(DEVICE)

optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                            lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

# ------------------- ОБУЧЕНИЕ -------------------
print("\n=== НАЧАЛО ОБУЧЕНИЯ ===")
start_time = time.time()

history = {"train_loss": [], "val_loss": []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for images, targets in train_loader:
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        train_loss += losses.item()

    train_loss /= len(train_loader)

    # Валидация
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()
    val_loss /= len(val_loader)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    print(f"Эпоха {epoch}/{EPOCHS} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

end_time = time.time()
training_time = end_time - start_time

# ------------------- РЕЗУЛЬТАТЫ -------------------
print(f"\n=== ОБУЧЕНИЕ ЗАВЕРШЕНО ===")
print(f"Общее время: {training_time:.2f} секунд")
print(f"В среднем на эпоху: {training_time / EPOCHS:.2f} секунд")
print(f"Финальный train loss: {history['train_loss'][-1]:.4f}")
print(f"Финальный val loss: {history['val_loss'][-1]:.4f}")

# Сохранение модели
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Модель сохранена: {MODEL_SAVE_PATH}")

# Сохранение информации
info = {
    "total_training_time_seconds": round(training_time, 2),
    "average_time_per_epoch": round(training_time / EPOCHS, 2),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "num_images": len(image_paths),
    "num_faces": sum(len(b) for b in annotations.values()),
    "device": str(DEVICE),
    "final_train_loss": round(history["train_loss"][-1], 4),
    "final_val_loss": round(history["val_loss"][-1], 4)
}

with open("training_info.json", "w", encoding="utf-8") as f:
    json.dump(info, f, indent=4, ensure_ascii=False)

print("Информация сохранена в training_info.json")