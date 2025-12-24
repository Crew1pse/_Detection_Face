import os
from PIL import Image

IMAGE_DIR = "datasets/WIDER_train/images"
ANNOT_FILE = "datasets/WIDER_train/annotations/wider_face_train_bbx_gt.txt"
LABEL_DIR = "datasets/WIDER_train/labels"

os.makedirs(LABEL_DIR, exist_ok=True)

print("Конвертация с сохранением структуры папок...")

with open(ANNOT_FILE, 'r') as f:
    lines = f.readlines()

i = 0
converted = 0
skipped = 0
while i < len(lines):
    line = lines[i].strip()
    if not line.endswith('.jpg'):
        i += 1
        continue

    img_rel_path = line  # например, 0--Parade/0_Parade_marchingband_1_449.jpg
    full_img_path = os.path.join(IMAGE_DIR, img_rel_path)

    if not os.path.exists(full_img_path):
        i += 1
        continue

    i += 1
    num_boxes = int(lines[i].strip())
    i += 1

    boxes = []
    for _ in range(num_boxes):
        parts = lines[i].strip().split()
        if len(parts) >= 4:
            x, y, w, h = map(int, parts[:4])
            if w > 0 and h > 0:
                boxes.append((x, y, w, h))
        i += 1

    if not boxes:
        skipped += 1
        continue

    # Создаём ту же структуру папок в labels
    label_dir = os.path.join(LABEL_DIR, os.path.dirname(img_rel_path))
    os.makedirs(label_dir, exist_ok=True)

    label_path = os.path.join(label_dir, os.path.basename(img_rel_path).replace('.jpg', '.txt'))

    img = Image.open(full_img_path)
    img_width, img_height = img.size

    with open(label_path, 'w') as f:
        for x, y, w, h in boxes:
            cx = (x + w / 2) / img_width
            cy = (y + h / 2) / img_height
            bw = w / img_width
            bh = h / img_height
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    converted += 1
    if converted % 1000 == 0:
        print(f"Обработано: {converted}")

print(f"Готово! Создано {converted} меток (пропущено пустых: {skipped})")