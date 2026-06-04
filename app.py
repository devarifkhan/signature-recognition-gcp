import os
import sys
import shutil
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import load_object
from src.entity.config_entity import ModelTrainerConfig

load_dotenv()

ENV_MODE = os.getenv("ENV_MODE", "dev")
IS_DEV = ENV_MODE == "dev"

if IS_DEV:
    HOST = os.getenv("DEV_HOST", "127.0.0.1")
    PORT = int(os.getenv("DEV_PORT", "8000"))
else:
    HOST = os.getenv("PROD_HOST", "0.0.0.0")
    PORT = int(os.getenv("PROD_PORT", "8080"))

templates = Jinja2Templates(directory="templates")

app = FastAPI(
    title="Signature Recognition API",
    description="Genuine vs Forged signature detection using ResNet-34",
    version="1.0.0",
    docs_url="/docs" if IS_DEV else None,
    redoc_url="/redoc" if IS_DEV else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if IS_DEV else ["https://yourdomain.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
def home(request: Request):
    model_path = ModelTrainerConfig().MODEL_PATH
    model_ready = os.path.exists(model_path)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"env_mode": ENV_MODE, "model_ready": model_ready},
    )


@app.get("/health", tags=["Health"])
def health():
    model_path = ModelTrainerConfig().MODEL_PATH
    return {"status": "ok", "env": ENV_MODE, "model_ready": os.path.exists(model_path)}


@app.post("/train", tags=["Training"])
def train():
    try:
        from src.pipeline.training import TrainingPipeline
        logging.info(f"Training triggered via API — env: {ENV_MODE}")
        pipeline = TrainingPipeline()
        artifacts = pipeline.run_pipeline()
        return JSONResponse({
            "status": "success",
            "message": "Training complete",
            "val_accuracy": f"{artifacts.val_accuracy:.4f}",
            "model_path": artifacts.model_path,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(CustomException(e, sys)))


@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """Upload a signature image (PNG/JPG) — returns genuine or forged."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (PNG or JPG)")

    model_path = ModelTrainerConfig().MODEL_PATH
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not trained yet. Hit /train first.")

    import torch
    from PIL import Image
    from torchvision import transforms

    predict_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    tmp_path = f"tmp_{file.filename}"
    try:
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        model = load_object(model_path)
        model.eval()

        img = Image.open(tmp_path).convert("RGB")
        tensor = predict_transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            predicted = outputs.argmax(1).item()

        label = "genuine" if predicted == 0 else "forged"
        confidence = probs[predicted].item()

        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "prediction": label,
            "confidence": f"{confidence:.2%}",
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(CustomException(e, sys)))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=HOST,
        port=PORT,
        reload=IS_DEV,
    )
