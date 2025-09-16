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
# Config (edit if needed)
# -------------------------
IMG_SIZE = (224, 224)                          # change if your model expects different size
LOCAL_MODEL_PATH = "models/cow_model.h5"
# Use the exact filename you have in the repo (keeps compatibility)
DATA_FILENAME = "Facila Recongnition Data.xlsx"  # <-- matches your uploaded file
# Candidate columns in your Excel - code will try these automatically
PREFERRED_LABEL_COLS = ["Label", "label", "class", "Class", "index"]
PREFERRED_NAME_COLS = ["Name", "name", "CowName", "cow_name", "Cow", "cow"]
PREFERRED_ID_COLS = ["ID", "id", "CowID", "cow_id"]
PREFERRED_GENDER_COLS = ["Gender", "gender", "Sex", "sex"]
OTHER_INFO_COLS = []  # add more known column names if desired

# -------------------------
# Helpers: Google Drive-safe downloader
# -------------------------
def _extract_gdrive_id(url_or_id: str) -> str:
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", str(url_or_id))
    if m:
        return m.group(1)
    m = re.search(r"id=([a-zA-Z0-9_-]+)", str(url_or_id))
    if m:
        return m.group(1)
    return str(url_or_id)

def download_file_from_gdrive(url_or_id: str, out_path: str, chunk_size: int = 32768):
    file_id = _extract_gdrive_id(url_or_id)
    session = requests.Session()
    URL = "https://docs.google.com/uc?export=download"
    params = {"id": file_id}
    response = session.get(URL, params=params, stream=True)
    token = None
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break
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
    try:
        df = pd.read_excel(path, engine="openpyxl")
        # reset index to ensure predictable iloc mapping
        df = df.reset_index(drop=True)
        return df
    except Exception:
        # if read_excel fails, return None and print trace in main flow
        return None

@st.cache_resource
def get_model_from_url(model_url: str, local_path: str):
    if not os.path.exists(local_path):
        download_file(model_url, local_path)
    model = load_model(local_path, compile=False)
    return model

# -------------------------
# Startup & diagnostics
# -------------------------
st.set_page_config(page_title="GaushalaNet", layout="wide")
st.title("🐄 GaushalaNet — Cattle ID Demo")

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

# Load Excel
df_cow = None
try:
    df_cow = load_excel_data(DATA_PATH)
    if df_cow is not None:
        st.sidebar.success("Excel data loaded")
    else:
        st.sidebar.warning("Excel file not found or unreadable — predictions won't map to cow info.")
except Exception as e:
    st.sidebar.error(f"Excel load error: {e}")
    st.sidebar.text(traceback.format_exc())
    df_cow = None

# Load model
model = None
MODEL_URL = None
try:
    # reading secrets; safe even if not present
    MODEL_URL = st.secrets.get("MODEL_URL", None)
except Exception:
    MODEL_URL = None

if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH, compile=False)
        st.sidebar.success("Model loaded from repo file")
    except Exception as e:
        st.sidebar.error(f"Failed to load local model: {e}")
        st.sidebar.text(traceback.format_exc())

if model is None and MODEL_URL:
    try:
        with st.spinner("Downloading & loading model (first run)..."):
            model = get_model_from_url(MODEL_URL, MODEL_PATH)
        st.sidebar.success("Model downloaded & loaded")
    except Exception as e:
        st.sidebar.error(f"Model download/load failed: {e}")
        st.sidebar.text(traceback.format_exc())

if model is None:
    st.warning("No model loaded. Set MODEL_URL in Streamlit secrets or place cow_model.h5 in models/")

# -------------------------
# Helper: best-effort column discovery
# -------------------------
def find_column(df, candidates):
    if df is None:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None

# -------------------------
# UI: image upload / sample
# -------------------------
st.markdown("## 📷 Predict cattle from an image")
col1, col2 = st.columns([1, 2])

with col1:
    uploaded = st.file_uploader("Upload image (png/jpg/jpeg)", type=["png", "jpg", "jpeg"])
    sample_files = [f for f in os.listdir(BASE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    sample_choice = None
    if sample_files:
        sample_choice = st.selectbox("Or choose a sample image from repo", ["-- none --"] + sample_files)

    run_btn = st.button("Run prediction")

with col2:
    image = None
    if uploaded:
        try:
            image = Image.open(io.BytesIO(uploaded.read()))
            st.image(image, caption="Uploaded image", use_container_width=True)
        except Exception as e:
            st.error(f"Cannot open uploaded image: {e}")
    elif sample_choice and sample_choice != "-- none --":
        try:
            image = Image.open(os.path.join(BASE_DIR, sample_choice))
            st.image(image, caption=f"Sample: {sample_choice}", use_container_width=True)
        except Exception as e:
            st.error(f"Cannot open sample image: {e}")
    else:
        st.info("Upload or select an image to run prediction.")

# -------------------------
# Prediction & mapping
# -------------------------
def map_prediction_to_row(df, top_idx):
    """
    Try to map predicted top_idx to a row:
    1. If a numeric label column exists, match on that.
    2. Else fallback to iloc[top_idx] if in range.
    Returns dict row or None.
    """
    if df is None:
        return None

    label_col = find_column(df, PREFERRED_LABEL_COLS)
    if label_col is not None:
        # attempt numeric match
        try:
            # Try matching integer label
            match = df[df[label_col].astype(str).str.strip() == str(top_idx)]
            if not match.empty:
                return match.iloc[0].to_dict()
        except Exception:
            pass
        # try numeric cast
        try:
            match = df[df[label_col].astype(float) == float(top_idx)]
            if not match.empty:
                return match.iloc[0].to_dict()
        except Exception:
            pass

    # fallback: positional mapping
    if 0 <= top_idx < len(df):
        return df.iloc[top_idx].to_dict()

    return None

def predict_and_display(img: Image.Image):
    if model is None:
        st.error("Model not available.")
        return

    x = pil_to_numpy(img, target_size=IMG_SIZE)
    try:
        preds = model.predict(x)
    except Exception as e:
        st.error(f"Model prediction error: {e}")
        st.text(traceback.format_exc())
        return

    # handle output shapes
    probs = preds.flatten() if preds.ndim > 1 else preds
    top_idx = int(np.argmax(probs))
    top_prob = float(np.max(probs))

    st.markdown("### 🔮 Prediction")
    st.write(f"Predicted class index: **{top_idx}** — probability: **{top_prob:.3f}**")

    # Try mapping to Excel
    mapped = map_prediction_to_row(df_cow, top_idx)
    if mapped:
        # find name/gender columns if present
        name_col = next((c for c in PREFERRED_NAME_COLS if c in mapped), None)
        id_col = next((c for c in PREFERRED_ID_COLS if c in mapped), None)
        gender_col = next((c for c in PREFERRED_GENDER_COLS if c in mapped), None)

        display_lines = []
        if name_col:
            display_lines.append(f"**Name:** {mapped.get(name_col)}")
        if id_col:
            display_lines.append(f"**ID:** {mapped.get(id_col)}")
        if gender_col:
            display_lines.append(f"**Gender:** {mapped.get(gender_col)}")

        # other info
        for other in OTHER_INFO_COLS:
            if other in mapped:
                display_lines.append(f"**{other}:** {mapped.get(other)}")

        # show as success + detail table
        st.success("🐄 Matched record from Excel")
        for ln in display_lines:
            st.write(ln)
        # show full row as table
        st.write(pd.DataFrame([mapped]))
    else:
        st.info("No matching record found in Excel. Showing top probabilities:")
        topk = np.argsort(probs)[::-1][:5]
        for i in topk:
            st.write(f"class {int(i)} — prob {probs[int(i)]:.3f}")

# Run prediction when button pressed or when image uploaded and auto-run
if run_btn or (uploaded is not None) or (sample_choice and sample_choice != "-- none --"):
    if image is None:
        st.warning("Please upload or select an image first.")
    else:
        predict_and_display(image)

# -------------------------
# Footer / help
# -------------------------
st.write("---")
st.info("Make sure MODEL_URL is set in Streamlit secrets (if you want the app to download the model). "
        "If cow_model.h5 exists in models/ it will be used instead.")
st.caption("If mapping doesn't show the cow name, upload a small sample of your Excel or tell me column names and I will adapt the mapping.")
