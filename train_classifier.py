import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
import os
import time
import json

DATA_DIR = r"datasets\UTKFace"
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "models/face_classifier.pth"

class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith('.jpg') or f.endswith('.jpg.chip.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        try:
            parts = img_name.split('_')
            age = int(parts[0])
            gender = int(parts[1])
        except:
            age = 30
            gender = 0

        if self.transform:
            image = self.transform(image)

        return image, (torch.tensor(age, dtype=torch.float32), torch.tensor(gender, dtype=torch.long))

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if __name__ == '__main__':
    print(f"Устройство: {DEVICE}")
    print("Запуск обучения классификатора (пол + возраст)")

    dataset = UTKFaceDataset(DATA_DIR, transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Модель
    model = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Identity()

    class ClassifierHead(nn.Module):
        def __init__(self, in_features):
            super().__init__()
            self.shared = nn.Linear(in_features, 128)
            self.age_head = nn.Linear(128, 1)
            self.gender_head = nn.Linear(128, 2)

        def forward(self, x):
            x = torch.relu(self.shared(x))
            age = self.age_head(x).squeeze(1)
            gender = self.gender_head(x)
            return age, gender

    model = nn.Sequential(model, ClassifierHead(num_ftrs))
    model.to(DEVICE)

    age_criterion = nn.MSELoss()
    gender_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("НАЧАЛО ОБУЧЕНИЯ КЛАССИФИКАТОРА")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for images, (ages, genders) in train_loader:
            images = images.to(DEVICE)
            ages = ages.to(DEVICE)
            genders = genders.to(DEVICE)

            optimizer.zero_grad()
            age_pred, gender_pred = model(images)
            loss = age_criterion(age_pred, ages) + gender_criterion(gender_pred, genders)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, (ages, genders) in val_loader:
                images = images.to(DEVICE)
                ages = ages.to(DEVICE)
                genders = genders.to(DEVICE)
                age_pred, gender_pred = model(images)
                loss = age_criterion(age_pred, ages) + gender_criterion(gender_pred, genders)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Эпоха {epoch}/{EPOCHS} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

    end_time = time.time()
    training_time = end_time - start_time

    print(f"\nОБУЧЕНИЕ ЗАВЕРШЕНО! Время: {training_time:.2f} сек")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Классификатор сохранён: {MODEL_SAVE_PATH}")

    info = {
        "task": "face_classification_age_gender",
        "epochs": EPOCHS,
        "total_time_sec": round(training_time, 2)
    }
    with open("classifier_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)