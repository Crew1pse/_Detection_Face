import torch
from ultralytics import YOLO
import time
import json
import os

if __name__ == '__main__':
    EPOCHS = 30
    BATCH = 6
    IMG_SIZE = 320
    MODEL_NAME = "yolov8m.pt"
    DEVICE = "0" if torch.cuda.is_available() else "cpu"

    print(f"Устройство: {'GPU' if DEVICE == '0' else 'CPU'}")
    print("Запуск обучения YOLOv8 на WIDER FACE...")

    start_time = time.time()

    model = YOLO(MODEL_NAME)

    results = model.train(
        data="data.yaml",
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMG_SIZE,
        device=DEVICE,
        name="wider_face_yolo",
        patience=20,
        save=True,
        project="runs",
        workers=4,
        val=True,
        plots=True
    )

    end_time = time.time()
    training_time = end_time - start_time

    save_dir = results.save_dir
    best_model_path = os.path.join(save_dir, "weights", "best.pt")

    os.makedirs("models", exist_ok=True)
    final_path = "models/yolo_face_best.pt"

    os.system(f"copy \"{best_model_path}\" \"{final_path}\" > nul 2>&1")

    print(f"Лучшая модель скопирована в: {final_path}")

    print(f"\nОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"Общее время: {training_time:.2f} секунд ({training_time / 3600:.2f} часов)")
    print(f"Лучшая модель сохранена: {final_path}")
    print(f"mAP@0.5: {results.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {results.box.map:.4f}")

    info = {
        "model": "YOLOv8",
        "variant": MODEL_NAME,
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "img_size": IMG_SIZE,
        "total_time_sec": round(training_time, 2),
        "map50": round(results.box.map50, 4),
        "map": round(results.box.map, 4),
        "best_model_path": final_path,
        "device": "GPU" if DEVICE == "0" else "CPU"
    }

    with open("yolo_training_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)

    print("Информация сохранена в yolo_training_info.json")