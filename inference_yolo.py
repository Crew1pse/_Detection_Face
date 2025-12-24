from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

model = YOLO("models/yolo_face_best.pt")

# Тест на фото
img_path = "images/10--People_Marching/10_People_Marching_People_Marching_10_People_Marching_People_Marching_10_299.jpg"  # замени на своё

results = model(img_path)[0]

# Визуализация
img = Image.open(img_path)
plt.figure(figsize=(12, 8))
plt.imshow(img)
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = box.conf[0]
    plt.rectangle(plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='red', facecolor='none'))
    plt.text(x1, y1-10, f"{conf:.2f}", color='red', fontsize=12)

plt.axis('off')
plt.title(f"Найдено лиц: {len(results.boxes)}")
plt.show()

print(f"Найдено лиц: {len(results.boxes)}")