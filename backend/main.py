from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import pickle
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globals
model = None
class_names = []
IMG_SIZE = (224, 224)

# Paths (adjust these if needed)
model_path = "../models/plant_disease_model_fixed.keras"
metadata_path = "../models/plant_disease_model_fixed_metadata.pkl"

# Create FastAPI app
app = FastAPI(
    title="Plant Disease Detection API",
    description="API for detecting plant diseases using machine learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Be careful with this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and metadata on startup
@app.on_event("startup")
async def load_model_and_metadata():
    global model, class_names
    try:
        logger.info("Loading model and metadata...")
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {os.path.abspath(model_path)}")
            model = None
        else:
            model = tf.keras.models.load_model(model_path)
            logger.info("Model loaded successfully")
        
        if not os.path.exists(metadata_path):
            logger.error(f"Metadata file not found: {os.path.abspath(metadata_path)}")
            class_names = ["Unknown"]
        else:
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            class_names = metadata.get("class_names", ["Unknown"])
            logger.info(f"Metadata loaded: {len(class_names)} classes")
    except Exception as e:
        logger.error(f"Error loading model/metadata: {e}")
        model = None
        class_names = ["Unknown"]

def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(IMG_SIZE)
        img_array = np.array(image)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=400, detail=f"Image processing error: {e}")

@app.get("/")
async def root():
    return {"message": "Plant Disease Detection API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classes_available": len(class_names),
        "model_input_size": IMG_SIZE
    }

@app.get("/classes")
async def get_classes():
    if not class_names or class_names == ["Unknown"]:
        raise HTTPException(status_code=503, detail="Model/classes not loaded")
    return {"classes": class_names, "total_classes": len(class_names)}

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")
    image_bytes = await file.read()
    img = preprocess_image(image_bytes)
    predictions = model.predict(img)
    pred_idx = np.argmax(predictions, axis=1)[0]
    pred_class = class_names[pred_idx]
    confidence = float(np.max(predictions))
    top3_idx = np.argsort(predictions[0])[-3:][::-1]
    top3 = [{"class": class_names[i], "confidence": float(predictions[0][i])} for i in top3_idx]
    return {
        "success": True,
        "predicted_class": pred_class,
        "confidence": confidence,
        "top_3_predictions": top3,
        "filename": file.filename
    }

@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Max 10 files allowed")
    results = []
    for file in files:
        if not file.content_type.startswith("image/"):
            results.append({"filename": file.filename, "success": False, "error": "File is not an image"})
            continue
        try:
            image_bytes = await file.read()
            img = preprocess_image(image_bytes)
            predictions = model.predict(img)
            pred_idx = np.argmax(predictions, axis=1)[0]
            pred_class = class_names[pred_idx]
            confidence = float(np.max(predictions))
            results.append({
                "filename": file.filename,
                "success": True,
                "predicted_class": pred_class,
                "confidence": confidence
            })
        except Exception as e:
            results.append({"filename": file.filename, "success": False, "error": str(e)})
    return {
        "batch_results": results,
        "total_processed": len(results),
        "successful_predictions": sum(r["success"] for r in results)
    }

@app.get("/model/info")
async def get_model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        return {
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
            "total_params": model.count_params(),
            "layers": len(model.layers),
            "classes": {"total": len(class_names), "names": class_names}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
