from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import uvicorn
import io
from PIL import Image

app = FastAPI()

# Load your model at startup
MODEL_PATH = "cow_model.h5"
model = load_model(MODEL_PATH)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Load image from request
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))  # adjust to your model input
        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Run prediction
        preds = model.predict(img_array)
        class_idx = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds))

        return JSONResponse({
            "class_index": class_idx,
            "confidence": confidence
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
