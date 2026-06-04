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
from src.pipeline.training import TrainingPipeline

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
    return templates.TemplateResponse(request=request, name="index.html", context={"env_mode": ENV_MODE})


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "env": ENV_MODE}


@app.post("/train", tags=["Training"])
def train():
    try:
        logging.info(f"Training triggered via API — env: {ENV_MODE}")
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
        return JSONResponse({"status": "success", "message": "Training pipeline completed"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(CustomException(e, sys)))


@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """Upload a signature image (PNG/JPG) — returns genuine or forged."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (PNG or JPG)")

    tmp_path = f"tmp_{file.filename}"
    try:
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # TODO: load trained model and run inference
        # model = load_object(MODEL_PATH)
        # prediction = model.predict(tmp_path)

        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "prediction": "model not yet trained — run /train first",
        })
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
