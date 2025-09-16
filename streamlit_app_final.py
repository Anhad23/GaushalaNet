# streamlit_app_final.py
import os
import re
import io
import traceback
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import requests

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# -------------------------
# Config (edit if required)
# -------------------------
IMG_SIZE = (224, 224)                 # model input size
LOCAL_MODEL_PATH = "models/cow_model.h5"
DATA_FILENAME = "Facial_Recognition_Data.xlsx"   # adjust if you renamed the Excel
CLASS_LABEL_COL = "Label"             # column in Excel that stores label/index (optional)
DISPLAY_NAME_COL = "Name"             # friendly name column (optional)

# -------------------------
# Helpers: Google Drive-safe downloader
# -------------------------
def _extract_gdrive_id(url_or_id: str) -> str:
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", url_or_id)
    if m:
        return m.group(1)
    m = re.search(r"id=([a-zA-Z0-9_-]+)", url_or_id)
    if m:
        return m.group(1)
    # assume it's already an id
    return url_or_id

def download_file_from_gdrive(url_or_id: str, out_path: str, chunk_size: int = 32768):
    file_id = _extract_gdrive_id(url_or_id)
    session = requests.Session()
    URL = "https://docs.google.com/uc?export=download"
    params = {"id": file_id}
    response = session.get(URL, params=params, stream=True)
    token = None
    # try cookies first
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break
    # fallback: try to parse confirm token from html
    if not token:
        m = re.search(r"confirm=([0-9A-Za-z_]+)&", response.text)
        if m:
            token = m.group(1)
    if token:
        params["confirm"] = token
        response = session.get(URL, params=params, stream=True)

    response.raise_for_status()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    return out_path

def download_file_generic(url: str, out_path: str, chunk_size: int = 8192):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    return out_path

def download_file(url_or_id: str, out_path: str):
    if "drive.google.com" in str(url_or_id) or re.match(r"^[a-zA-Z0-9_-]{10,}$", str(url_or_id)):
        return download_file_from_gdrive(url_or_id, out_path)
    return download_file_generic(url_or_id, out_path)

# -------------------------
# Preprocessing
# -------------------------
def pil_to_numpy(img: Image.Image, target_size=IMG_SIZE):
    img = img.convert("RGB")
    img = img.resize(target_size)
    arr = img_to_array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

# -------------------------
# Cached resources
# -------------------------
@st.cache_data(ttl=3600)
def load_excel_data(path: str):
    if not os.path.exists(path):
        return None
    return pd.read_excel(path, engine="openpyxl")

@st.cache_resource
def get_model_from_url(model_url: str, local_path: str):
    if not os.path.exists(local_path):
        download_file(model_url, local_path)
    # load model (compile=False to avoid optimizer mismatch)
    model = load_model(local_path, compile=False)
    return model

# -------------------------
# Startup and diagnostics
# -------------------------
st.set_page_config(page_title="GaushalaNet", layout="wide")
st.title("GaushalaNet — Cattle ID Demo")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
Path("models").mkdir(parents=True, exist_ok=True)
DATA_PATH = os.path.join(BASE_DIR, DATA_FILENAME)
MODEL_PATH = os.path.join(BASE_DIR, LOCAL_MODEL_PATH)

st.sidebar.markdown("### Environment")
st.sidebar.write("cwd:", os.getcwd())
st.sidebar.write("base_dir:", BASE_DIR)
st.sidebar.write("files:", sorted(os.listdir(BASE_DIR)))
st.sidebar.markdown("### Paths")
st.sidebar.write("DATA_PATH:", DATA_PATH)
st.sidebar.write("MODEL_PATH:", MODEL_PATH)
st.sidebar.write("Exists DATA:", os.path.exists(DATA_PATH))
st.sidebar.write("Exists MODEL:", os.path.exists(MODEL_PATH))

# Load Excel (optional)
df_cow = None
try:
    df_cow = load_excel_data(DATA_PATH)
    if df_cow is not None:
        st.sidebar.success("Excel data loaded")
    else:
        st.sidebar.info("Excel not found — continuing without label mapping.")
except Exception as e:
    st.sidebar.error(f"Failed to read Excel: {e}")
    st.sidebar.text(traceback.format_exc())

# Load model: prefer local file; else use MODEL_URL secret
model = None
MODEL_URL = None
try:
    MODEL_URL = st.secrets.get("MODEL_URL", None)
except Exception:
    MODEL_URL = None

if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH, compile=False)
        st.sidebar.success("Model loaded from repo file")
    except Exception as e:
        st.sidebar.error("Failed to load local model; will try MODEL_URL if set.")
        st.sidebar.text(str(e))

if model is None and MODEL_URL:
    try:
        with st.spinner("Downloading and loading model (first run)..."):
            model = get_model_from_url(MODEL_URL, MODEL_PATH)
        st.sidebar.success("Model downloaded & loaded")
    except Exception as e:
        st.sidebar.error(f"Model download/load failed: {e}")
        st.sidebar.text(traceback.format_exc())

if model is None:
    st.warning("No model loaded. Set MODEL_URL secret in Streamlit Cloud or add cow_model.h5 to repo.")

# -------------------------
# UI: upload and sample selection
# -------------------------
st.markdown("## Predict cattle from an image")
col1, col2 = st.columns([1, 2])

with col1:
    uploaded = st.file_uploader("Upload an image (png/jpg/jpeg)", type=["png", "jpg", "jpeg"])
    sample_files = [f for f in os.listdir(BASE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    sample_choice = None
    if sample_files:
        sample_choice = st.selectbox("Sample images in repo", ["-- none --"] + sample_files)
    if st.button("Run prediction") and uploaded is None and (not sample_choice or sample_choice == "-- none --"):
        st.info("Please upload an image or select a sample first.")

with col2:
    image = None
    if uploaded:
        try:
            image = Image.open(io.BytesIO(uploaded.read()))
            st.image(image, caption="Uploaded image", use_column_width=True)
        except Exception as e:
            st.error(f"Failed to open uploaded image: {e}")
    elif sample_choice and sample_choice != "-- none --":
        try:
            image = Image.open(os.path.join(BASE_DIR, sample_choice))
            st.image(image, caption=f"Sample: {sample_choice}", use_column_width=True)
        except Exception as e:
            st.error(f"Failed to open sample image: {e}")
    else:
        st.info("Upload an image to run prediction.")

# -------------------------
# Prediction logic
# -------------------------
def predict_and_show(img: Image.Image):
    if model is None:
        st.error("Model is not available.")
        return

    x = pil_to_numpy(img, target_size=IMG_SIZE)
    try:
        preds = model.predict(x)
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.text(traceback.format_exc())
        return

    probs = preds.flatten() if preds.ndim > 1 else preds
    top_idx = int(np.argmax(probs))
    top_prob = float(np.max(probs))

    st.markdown("### Prediction")
    st.write(f"Predicted class index: **{top_idx}** — probability: **{top_prob:.3f}**")

    # Try mapping to Excel row if available
    if df_cow is not None:
        try:
            if CLASS_LABEL_COL in df_cow.columns:
                # if labels are stored as ints
                if np.issubdtype(df_cow[CLASS_LABEL_COL].dtype, np.integer):
                    match = df_cow[df_cow[CLASS_LABEL_COL] == top_idx]
                else:
                    # fallback: map by position if index in range
                    match = df_cow.iloc[[top_idx]] if top_idx < len(df_cow) else pd.DataFrame()
                if not match.empty:
                    display = match.iloc[0].to_dict()
                    name = display.get(DISPLAY_NAME_COL, display)
                    st.write("Matched record from Excel:")
                    st.write(match)
                    st.success(f"Identified as: **{name}**")
                else:
                    st.info("No matching row found in Excel for predicted class/index.")
            else:
                st.info("Excel loaded but CLASS_LABEL_COL not found; showing top probabilities.")
                topk = np.argsort(probs)[::-1][:5]
                for i in topk:
                    st.write(f"class {int(i)} — prob {probs[int(i)]:.3f}")
        except Exception as e:
            st.error(f"Failed mapping to Excel: {e}")
            st.text(traceback.format_exc())
    else:
        st.info("No Excel data available to map labels to names. Showing top probabilities.")
        topk = np.argsort(probs)[::-1][:5]
        for i in topk:
            st.write(f"class {int(i)} — prob {probs[int(i)]:.3f}")

# Run prediction when image is present
if image is not None:
    predict_and_show(image)

# Footer
st.write("---")
st.info("Ensure you set the MODEL_URL secret in Streamlit Cloud to your Google Drive link (or model ID).")
st.markdown(
    """
**Notes**
- First run will download the model; this may take several minutes depending on file size and network speed.
- If your model expects different preprocessing (mean/std normalization, different size), adjust IMG_SIZE and pil_to_numpy accordingly.
"""
)
