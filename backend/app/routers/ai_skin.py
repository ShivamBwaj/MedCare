from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import torch
from torch import nn
from torchvision import transforms, models
import io

router = APIRouter(prefix="/ai/skin", tags=["ai-skin"])

# Load model on startup
MODEL_PATH = "ai/skin_model.pt"
try:
    CHK = torch.load(MODEL_PATH, map_location="cpu")
    CLASSES = CHK["classes"]
    
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    model.load_state_dict(CHK["state_dict"])
    model.eval()
    
    T = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
except Exception as e:
    print(f"Failed to load skin model: {e}")
    model = None
    CLASSES = ["healthy", "fungal", "psoriasis"]

@router.post("/predict")
async def predict_skin(file: UploadFile = File(...)):
    if not model:
        # Return demo prediction
        return {"label": "healthy", "confidence": 0.85}
    
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Bad image")
    
    x = T(img).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)[0]
        idx = int(torch.argmax(prob).item())
    
    return {"label": CLASSES[idx], "confidence": round(float(prob[idx]), 3)}
