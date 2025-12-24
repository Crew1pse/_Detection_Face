import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from utils.config import DEVICE, MODEL_SAVE_PATH
from utils.visualization import show_image_with_boxes
import os

model = fasterrcnn_resnet50_fpn_v2(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torch.nn.Linear(in_features, 2)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(image_path, conf_threshold=0.5):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prediction = model(img_tensor)[0]

    boxes = prediction['boxes'][prediction['scores'] > conf_threshold].cpu().numpy()
    return boxes.astype(int)

# Пример использования
if __name__ == "__main__":
    test_img_path = "images/0--Parade/0_Parade_marchingband_1_449.jpg"  # Замени на любое
    pred_boxes = predict(test_img_path)
    show_image_with_boxes(test_img_path, predicted_boxes=pred_boxes)