import os
import shutil
from tqdm import tqdm  # Для прогресс-бара (pip install tqdm если нет)

# Пути
RAW_DIR = "UTKFace"  # Папка после распаковки ZIP (или "crop_part1")
OUTPUT_DIR = "utkface_gender"  # Куда сохранять (по полу)
# Или "utkface_age" для возраста

TRAIN_RATIO = 0.8  # 80% train, 20% val

os.makedirs(os.path.join(OUTPUT_DIR, "train", "male"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "train", "female"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "val", "male"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "val", "female"), exist_ok=True)

files = [f for f in os.listdir(RAW_DIR) if f.endswith('.jpg') or f.endswith('.jpg.chip.jpg')]

print(f"Найдено файлов: {len(files)}")
print("Организация по полу (male/female)...")

for file in tqdm(files):
    try:
        parts = file.split('_')
        age = int(parts[0])
        gender = int(parts[1])  # 0 = male, 1 = female

        gender_folder = "male" if gender == 0 else "female"

        # Разделение train/val
        if hash(file) % 100 < TRAIN_RATIO * 100:
            split = "train"
        else:
            split = "val"

        src = os.path.join(RAW_DIR, file)
        dst = os.path.join(OUTPUT_DIR, split, gender_folder, file)
        shutil.copy(src, dst)
    except:
        print(f"Ошибка с файлом: {file}")

print("Готово! Датасет организован в", OUTPUT_DIR)
print("Структура:")
print("- train/male")
print("- train/female")
print("- val/male")
print("- val/female")