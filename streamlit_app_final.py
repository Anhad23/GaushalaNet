# streamlit_app.py
import os
import io
import re
import json
import requests
from io import BytesIO
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

st.set_page_config(page_title="GaushalaNet — Local", layout="wide")

# ---------------- CONFIG ----------------
DATA_PATH = "Facila Recongnition Data.xlsx"   # Excel with cow details (your file)
MODEL_PATH = "cow_model.h5"                   # Local fallback model
CLASS_INDICES_PATH = "class_indices.json"     # optional mapping saved at training
TRAIN_DIR = "cow_nose_dataset/training_data"  # optional fallback mapping
IMAGE_SIZE = (224, 224)                       # model input size
SAMPLES_DIRS = ["samples", "."]               # search these for sample images (samples/ first)
MAX_SAMPLES = 3                               # show up to 3 sample thumbnails

# ---------------- HELPERS ----------------
@st.cache_data(ttl=600)
def load_excel(path=DATA_PATH):
    """Load Excel with openpyxl backend. Returns DataFrame or empty df."""
    try:
        df = pd.read_excel(path, engine="openpyxl")
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

def _extract_gdrive_id(url_or_id: str):
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", str(url_or_id))
    if m:
        return m.group(1)
    m = re.search(r"id=([a-zA-Z0-9_-]+)", str(url_or_id))
    if m:
        return m.group(1)
    return str(url_or_id)

def download_from_gdrive(url_or_id: str, out_path: str, chunk_size: int = 32768):
    """Download large file from Google Drive (handles confirm token)."""
    file_id = _extract_gdrive_id(url_or_id)
    session = requests.Session()
    URL = "https://docs.google.com/uc?export=download"
    params = {"id": file_id}
    resp = session.get(URL, params=params, stream=True)
    token = None
    for k, v in resp.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break
    if not token:
        m = re.search(r"confirm=([0-9A-Za-z_]+)&", resp.text)
        if m:
            token = m.group(1)
    if token:
        params["confirm"] = token
        resp = session.get(URL, params=params, stream=True)

    resp.raise_for_status()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    return out_path

@st.cache_resource
def load_model_cached(model_path=MODEL_PATH):
    """
    Load model from disk; if MODEL_URL secret is set and model file missing, download it first.
    Returns (model, error_msg_or_None).
    """
    try:
        import tensorflow as tf
    except Exception as e:
        return None, f"tensorflow import failed: {e}"

    MODEL_URL = None
    try:
        MODEL_URL = st.secrets.get("MODEL_URL", None)
    except Exception:
        MODEL_URL = None

    if MODEL_URL and not os.path.exists(model_path):
        try:
            download_from_gdrive(MODEL_URL, model_path)
        except Exception as e:
            return None, f"model download failed: {e}"

    try:
        model = tf.keras.models.load_model(model_path)
        return model, None
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=600)
def load_class_indices(path=CLASS_INDICES_PATH, train_dir=TRAIN_DIR, df=None):
    """
    Build inv_class_indices mapping index -> label.
    Priority:
      1) class_indices.json (folder_name -> idx)
      2) Excel 'Names' column (exact header 'Names' case-insensitive)
      3) 'name' fallback (if present)
      4) training folder names
    """
    inv = {}
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                class_idx = json.load(f)
            inv = {int(v): k for k, v in class_idx.items()}
            return inv, "loaded_json"
        except Exception:
            pass

    if df is not None and not df.empty:
        cols_lower = [c.lower() for c in df.columns]
        if "names" in cols_lower:
            names = df[df.columns[cols_lower.index("names")]].astype(str).tolist()
            inv = {i: n for i, n in enumerate(names)}
            return inv, "loaded_excel_Names"
        if "name" in cols_lower:
            names = df[df.columns[cols_lower.index("name")]].astype(str).tolist()
            inv = {i: n for i, n in enumerate(names)}
            return inv, "loaded_excel_name"
        if "id" in cols_lower:
            ids = df[df.columns[cols_lower.index("id")]].astype(str).tolist()
            inv = {i: n for i, n in enumerate(ids)}
            return inv, "loaded_excel_id"

    if os.path.exists(train_dir):
        folders = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        inv = {i: name for i, name in enumerate(folders)}
        return inv, "loaded_train_dir"

    return {}, "no_mapping_found"

def get_resample_filter():
    try:
        return Image.Resampling.LANCZOS
    except Exception:
        try:
            return Image.LANCZOS
        except Exception:
            return Image.BICUBIC

def preprocess_pil(img: Image.Image, target_size=IMAGE_SIZE):
    if img.mode != "RGB":
        img = img.convert("RGB")
    resample = get_resample_filter()
    img = ImageOps.fit(img, target_size, method=resample)
    arr = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_index(model, pil_image: Image.Image):
    x = preprocess_pil(pil_image)
    preds = model.predict(x)
    if preds is None:
        return None, None
    preds = np.asarray(preds)
    if preds.ndim == 2 and preds.shape[1] > 1:
        idx = int(np.argmax(preds[0]))
        conf = float(np.max(preds[0]))
        return idx, conf
    # handle weird outputs
    if preds.ndim == 1 and preds.size > 1:
        idx = int(np.argmax(preds))
        conf = float(np.max(preds))
        return idx, conf
    return None, None

def find_repo_samples(max_samples=MAX_SAMPLES):
    found = []
    for d in SAMPLES_DIRS:
        if not os.path.exists(d):
            continue
        for fname in sorted(os.listdir(d)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                found.append(("file", os.path.join(d, fname)))
                if len(found) >= max_samples:
                    return found
    return found

def get_sample_list(max_samples=MAX_SAMPLES):
    """
    Return list of sample items: tuples ("url", url) or ("file", path).
    Priority:
      1) st.secrets["SAMPLE_URLS"] (comma-separated)
      2) repo samples (samples/ then repo root)
    """
    samples = []
    # 1) secrets
    try:
        raw = st.secrets.get("SAMPLE_URLS", None)
    except Exception:
        raw = None
    if raw:
        for u in [u.strip() for u in raw.split(",") if u.strip()]:
            samples.append(("url", u))
            if len(samples) >= max_samples:
                return samples

    # 2) repo files
    samples.extend(find_repo_samples(max_samples - len(samples)))
    return samples[:max_samples]

def load_image_from_sample(item):
    kind, val = item
    if kind == "file":
        return Image.open(val)
    else:
        r = requests.get(val, timeout=10)
        r.raise_for_status()
        return Image.open(BytesIO(r.content))

def compute_health_score(row):
    score = 80
    try:
        if "health_status" in row and isinstance(row["health_status"], str):
            if "sick" in row["health_status"].lower():
                score -= 40
    except Exception:
        pass
    try:
        if "age" in row and pd.notna(row["age"]):
            if float(row["age"]) > 10:
                score -= 10
    except Exception:
        pass
    try:
        if "last_checkup" in row and pd.isna(row["last_checkup"]):
            score -= 10
    except Exception:
        pass
    return max(0, min(100, score))

# ---------------- STARTUP ----------------
st.title("🐄 GaushalaNet — Smart Gaushala Dashboard")

# create models dir if needed
Path("models").mkdir(parents=True, exist_ok=True)

# Diagnostics (sidebar)
st.sidebar.markdown("### Environment")
st.sidebar.write("cwd:", os.getcwd())
st.sidebar.write("files (repo root):", sorted(os.listdir(".")))
st.sidebar.write("DATA_PATH:", DATA_PATH)
st.sidebar.write("MODEL_PATH:", MODEL_PATH)

# load resources
df = load_excel(DATA_PATH)
model, model_err = load_model_cached(MODEL_PATH)
inv_class_indices, map_source = load_class_indices(CLASS_INDICES_PATH, TRAIN_DIR, df)
st.sidebar.write("Mapping source:", map_source)
st.sidebar.write("Excel loaded:", not df.empty)
st.sidebar.write("Model loaded:", model is not None)

# prepare label fallback list
labels = None
if inv_class_indices:
    max_idx = max(inv_class_indices.keys())
    labels = [inv_class_indices.get(i, "") for i in range(max_idx + 1)]
else:
    # prefer 'Names' col
    names_cols = [c for c in df.columns if c.lower() == "names"]
    if names_cols:
        labels = df[names_cols[0]].astype(str).tolist()
    elif "name" in [c.lower() for c in df.columns]:
        name_col = [c for c in df.columns if c.lower() == "name"][0]
        labels = df[name_col].astype(str).tolist()

# ---------------- SIDEBAR NAV ----------------
st.sidebar.title("GaushalaNet — Control")
page = st.sidebar.radio("Navigation", ["Home", "Predict (Image)", "Health Tracker", "Cow Profiles", "E-Learning", "Upload Data", "About"])

# ---------------- PAGES ----------------
if page == "Home":
    st.title("GaushalaNet — Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    total = len(df) if not df.empty else 0
    at_risk = int((df["health_score"] < 50).sum()) if ("health_score" in df.columns) else 0
    vacc_due = int(df["vaccination_due"].notna().sum()) if "vaccination_due" in df.columns else 0
    avg_h = round(df["health_score"].mean(), 1) if ("health_score" in df.columns) else 0
    c1.metric("Total cows", total)
    c2.metric("At-risk (score<50)", at_risk)
    c3.metric("Vaccination records", vacc_due)
    c4.metric("Avg health score", avg_h)

    st.markdown("### Health score distribution")
    if not df.empty and "health_score" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(df["health_score"].dropna(), bins=10)
        ax.set_xlabel("Health score")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    else:
        st.info("No dataset loaded. Upload data on Upload Data page.")

elif page == "Predict (Image)":
    st.title("Identify Cow from Image")

    # ensure session state
    if "selected_sample" not in st.session_state:
        st.session_state["selected_sample"] = None  # ("url"/"file", value)
    if "uploaded_image_bytes" not in st.session_state:
        st.session_state["uploaded_image_bytes"] = None

    # layout: left controls, right preview/outcome
    left, right = st.columns([1, 2])

    sample_items = get_sample_list(MAX_SAMPLES)

    with left:
        st.subheader("Input source")
        source_choice = st.radio("Choose input:", ("Sample", "Upload"))

        st.markdown("#### Sample images")
        if sample_items:
            cols = st.columns(len(sample_items))
            for i, item in enumerate(sample_items):
                kind, val = item
                try:
                    preview = load_image_from_sample(item)
                    cols[i].image(preview, use_column_width=True)
                except Exception:
                    cols[i].write(os.path.basename(val) if kind == "file" else val)
                if cols[i].button(f"Select sample {i+1}"):
                    st.session_state["selected_sample"] = item
            if st.session_state["selected_sample"]:
                s_kind, s_val = st.session_state["selected_sample"]
                st.caption(f"Selected: {os.path.basename(s_val) if s_kind=='file' else s_val}")
        else:
            st.info("No sample images found. Add up to 3 images to `samples/` or set SAMPLE_URLS secret.")

        st.markdown("---")
        st.subheader("Upload an image")
        uploaded = st.file_uploader("Upload (jpg/png)", type=["jpg", "jpeg", "png"], key="uploader")
        if uploaded is not None:
            st.session_state["uploaded_image_bytes"] = uploaded.read()
            st.success("Uploaded — select 'Upload' source to use this image.")

        auto_run = st.checkbox("Auto-run prediction when selecting a sample", value=False)
        predict_btn = st.button("Predict")

    with right:
        image_to_predict = None

        # preview image according to source
        if source_choice == "Upload":
            if st.session_state["uploaded_image_bytes"]:
                try:
                    image_to_predict = Image.open(io.BytesIO(st.session_state["uploaded_image_bytes"]))
                    st.image(image_to_predict, caption="Uploaded image (selected)", use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to open uploaded image: {e}")
            else:
                st.info("No uploaded image. Upload one on the left to use it.")
        else:  # Sample
            if st.session_state["selected_sample"]:
                try:
                    image_to_predict = load_image_from_sample(st.session_state["selected_sample"])
                    kind, val = st.session_state["selected_sample"]
                    st.image(image_to_predict, caption=f"Sample: {os.path.basename(val) if kind=='file' else val}", use_container_width=True)
                except Exception:
                    st.error("Failed to open selected sample image.")
            else:
                st.info("Select a sample image on the left to preview.")

        # auto-run
        if auto_run and source_choice == "Sample" and st.session_state["selected_sample"] and model is not None:
            try:
                idx, conf = predict_index(model, image_to_predict)
                if idx is not None:
                    # mapping priority: class_indices.json -> Excel 'Names'
                    if inv_class_indices and idx in inv_class_indices:
                        cow_name = inv_class_indices[idx]
                        st.success(f"🐄 Predicted Cow: **{cow_name}**")
                    else:
                        names_cols = [c for c in df.columns if c.lower() == "names"]
                        if names_cols and 0 <= idx < len(df):
                            cow_name = str(df.iloc[idx][names_cols[0]])
                            st.success(f"🐄 Predicted Cow: **{cow_name}**")
                        else:
                            st.error("Could not map prediction to a cow name — ensure Excel has a 'Names' column and matches training order.")
                else:
                    st.error("Model did not return a valid class index.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

        # manual Predict button
        if predict_btn:
            if image_to_predict is None:
                st.warning("Please select a sample or upload an image and pick the corresponding source.")
            else:
                if model is None:
                    st.error("No model loaded.")
                    if model_err:
                        st.code(model_err)
                else:
                    try:
                        idx, conf = predict_index(model, image_to_predict)
                        if idx is not None:
                            if inv_class_indices and idx in inv_class_indices:
                                cow_name = inv_class_indices[idx]
                                st.success(f"🐄 Predicted Cow: **{cow_name}**")
                            else:
                                names_cols = [c for c in df.columns if c.lower() == "names"]
                                if names_cols and 0 <= idx < len(df):
                                    cow_name = str(df.iloc[idx][names_cols[0]])
                                    st.success(f"🐄 Predicted Cow: **{cow_name}**")
                                else:
                                    st.error("Could not map prediction to a cow name — ensure Excel has a 'Names' column and matches training order.")
                        else:
                            st.error("Model did not return a valid class index.")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

elif page == "Health Tracker":
    st.title("Health Tracker")
    if df.empty:
        st.info("No dataset available.")
    else:
        st.markdown("#### Vaccinations due in next 30 days")
        if "vaccination_due" in df.columns:
            today = pd.Timestamp.now().normalize()
            vacc_dates = pd.to_datetime(df["vaccination_due"], errors="coerce")
            due = df[(vacc_dates - today).dt.days.between(0, 30)]
            if not due.empty:
                cols = [c for c in ["id", "Names", "breed", "vaccination_due"] if c in df.columns]
                st.dataframe(due[cols].head(50))
            else:
                st.success("No vaccinations due soon.")
        st.markdown("#### Low health score animals (score < 50)")
        low = df[df["health_score"] < 50] if "health_score" in df.columns else pd.DataFrame()
        if not low.empty:
            cols_to_show = [c for c in ["id", "Names", "breed", "age", "health_score", "health_status"] if c in df.columns]
            st.dataframe(low[cols_to_show].head(50))
        else:
            st.success("No animals below threshold.")

elif page == "Cow Profiles":
    st.title("Cow Profiles")
    if df.empty:
        st.info("No dataset available.")
    else:
        search = st.text_input("Search by ID or name")
        subset = df.copy()
        if search:
            mask = subset.astype(str).apply(lambda row: row.str.contains(search, case=False, na=False)).any(axis=1)
            subset = subset[mask]
        st.dataframe(subset.head(100))

elif page == "E-Learning":
    st.title("E-Learning")
    st.subheader("Feeding best practices")
    st.write("- Keep feeding times consistent\n- Provide clean water\n- Balance nutrition")
    st.subheader("Vaccination schedule")
    st.write("- Keep a vaccination calendar\n- Record boosters and doses\n- Consult vet for regional disease risks")
    st.markdown("---")
    st.subheader("Quick quiz")
    q1 = st.radio("How often review vaccination schedules?", ["Monthly", "Annually", "Only when sick"])
    q2 = st.radio("Is decreased appetite an early sign of illness?", ["Yes", "No"])
    if st.button("Submit Quiz"):
        score = 0
        if q1 == "Monthly":
            score += 1
        if q2 == "Yes":
            score += 1
        st.success(f"Score: {score}/2")

elif page == "Upload Data":
    st.title("Upload dataset (Excel)")
    uploaded = st.file_uploader("Upload Excel file (.xls/.xlsx)", type=["xls", "xlsx"])
    if uploaded is not None:
        try:
            new_df = pd.read_excel(uploaded, engine="openpyxl")
            st.dataframe(new_df.head())
            if st.button("Save uploaded dataset"):
                new_df.to_excel(DATA_PATH, index=False)
                st.success("Saved dataset. Restart app to load new data.")
        except Exception as e:
            st.error(f"Failed to read uploaded Excel: {e}")

else:
    st.title("About")
    st.write("GaushalaNet — Local demo")
    if model is None:
        st.warning("Model not loaded.")
        if model_err:
            st.code(model_err)
