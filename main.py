import io
import os
import base64
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from torchvision import transforms, models
from PIL import Image, ImageDraw
from ultralytics import YOLO
import uvicorn

app = FastAPI()

if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

YOLO_PATH = os.path.join("models", "best.pt")
model = YOLO(YOLO_PATH)

class ClassifierHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.shared = nn.Linear(in_features, 128)
        self.age_head = nn.Linear(128, 1)
        self.gender_head = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.shared(x))
        return self.age_head(x).squeeze(1), self.gender_head(x)


classifier = models.resnet18(weights=None)
classifier.fc = ClassifierHead(classifier.fc.in_features)

try:
    state_dict = torch.load("models/face_classifier.pth", map_location=DEVICE, weights_only=True)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("0."):
            new_state_dict[k[2:]] = v
        elif k.startswith("1."):
            new_state_dict["fc." + k[2:]] = v
        else:
            new_state_dict[k] = v
    classifier.load_state_dict(new_state_dict)
    classifier.to(DEVICE).eval()
except Exception as e:
    print(f"ВНИМАНИЕ: Не удалось загрузить веса классификатора: {e}")

cls_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

COLORS = ["#00f2ff", "#ff003c", "#00ff41", "#ffcc00", "#9d00ff", "#ffffff"]

@app.get("/")
async def read_index():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return "index.html not found"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    results = model(img_pil, conf=0.25)[0]

    faces_data = []
    draw = ImageDraw.Draw(img_pil)

    for i, box in enumerate(results.boxes):
        coords = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, coords)
        conf = float(box.conf[0])

        face_crop = img_pil.crop((x1, y1, x2, y2))
        face_tensor = cls_transform(face_crop).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            age_pred, gender_logit = classifier(face_tensor)
            gender_id = torch.argmax(gender_logit, dim=1).item()
            gender = "MALE" if gender_id == 0 else "FEMALE"
            age = int(abs(age_pred.item()))

        current_color = COLORS[i % len(COLORS)]
        draw.rectangle(coords, outline=current_color, width=3)

        faces_data.append({
            "id": i + 1,
            "coords": [x1, y1, x2, y2],
            "confidence": f"{conf:.2%}",
            "gender": gender,
            "age": age,
            "status": "VERIFIED" if conf > 0.4 else "UNCERTAIN",
            "color": current_color
        })

    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "image": f"data:image/jpeg;base64,{img_b64}",
        "faces": faces_data
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)