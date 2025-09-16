# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image, ImageOps
import io
import os
import json
import re
import requests

st.set_page_config(page_title="GaushalaNet — Local", layout="wide")

# ---------------- CONFIG ----------------
DATA_PATH = "Facila Recongnition Data.xlsx"   # Excel with cow details
MODEL_PATH = "cow_model.h5"                   # Local fallback model
CLASS_INDICES_PATH = "class_indices.json"     # optional mapping saved at training
TRAIN_DIR = "cow_nose_dataset/training_data"  # optional fallback
IMAGE_SIZE = (224, 224)                       # adjust if needed

# ---------------- HELPERS ----------------
@st.cache_data(ttl=600)
def load_data(path=DATA_PATH):
    try:
        df = pd.read_excel(path)
        return df
    except Exception as e:
        st.warning(f"Could not read Excel file at {path}: {e}")
        return pd.DataFrame()

def _extract_gdrive_id(url_or_id: str):
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", str(url_or_id))
    if m:
        return m.group(1)
    m = re.search(r"id=([a-zA-Z0-9_-]+)", str(url_or_id))
    if m:
        return m.group(1)
    return str(url_or_id)

def download_model_from_gdrive(url_or_id: str, out_path: str, chunk_size: int = 32768):
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

@st.cache_resource
def try_load_model(model_path=MODEL_PATH):
    import tensorflow as tf
    # If MODEL_URL secret exists, prefer that
    MODEL_URL = st.secrets.get("MODEL_URL", None) if "MODEL_URL" in st.secrets else None
    if MODEL_URL and not os.path.exists(model_path):
        try:
            st.sidebar.info("Downloading model from secrets URL...")
            download_model_from_gdrive(MODEL_URL, model_path)
        except Exception as e:
            return None, f"Download failed: {e}"

    try:
        model = tf.keras.models.load_model(model_path)
        return model, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def load_class_indices(path=CLASS_INDICES_PATH, train_dir=TRAIN_DIR, df=None):
    """
    Returns inv_class_indices: dict mapping index -> folder/name used in training.
    Priority:
      1) class_indices.json
      2) Excel df with 'name' or 'id' column
      3) Training folder names
    """
    inv = {}
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                class_idx = json.load(f)
            inv = {int(v): k for k, v in class_idx.items()}
            return inv, "loaded_json"
        except Exception as e:
            return {}, f"failed_json:{e}"

    if df is not None and not df.empty:
        if "name" in [c.lower() for c in df.columns]:
            names = df["name"].astype(str).tolist()
            inv = {i: n for i, n in enumerate(names)}
            return inv, "loaded_excel_name_order"
        if "id" in [c.lower() for c in df.columns]:
            ids = df["id"].astype(str).tolist()
            inv = {i: n for i, n in enumerate(ids)}
            return inv, "loaded_excel_id_order"

    if os.path.exists(train_dir):
        folders = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        inv = {i: name for i, name in enumerate(folders)}
        return inv, "loaded_train_dir"

    return {}, "no_mapping_found"

def get_resample_filter():
    try:
        return Image.Resampling.LANCZOS   # Pillow >=10
    except AttributeError:
        try:
            return Image.LANCZOS
        except AttributeError:
            return Image.BICUBIC

def preprocess_image(image: Image.Image, target_size=IMAGE_SIZE):
    if image.mode != "RGB":
        image = image.convert("RGB")
    resample = get_resample_filter()
    image = ImageOps.fit(image, target_size, method=resample)
    arr = np.array(image).astype(np.float32) / 255.0
    return arr

def predict_index(model, pil_image: Image.Image):
    x = preprocess_image(pil_image)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    if preds.ndim == 2 and preds.shape[1] > 1:
        idx = int(np.argmax(preds[0]))
        conf = float(np.max(preds[0]))
        return idx, conf
    else:
        return None, None

def compute_health_score(row):
    score = 80
    if "health_status" in row and isinstance(row["health_status"], str):
        if "sick" in row["health_status"].lower():
            score -= 40
    if "age" in row and pd.notna(row["age"]):
        try:
            if float(row["age"]) > 10:
                score -= 10
        except Exception:
            pass
    if "last_checkup" in row and pd.isna(row["last_checkup"]):
        score -= 10
    return max(0, min(100, score))

# ---------------- LOAD DATA & MODEL ----------------
df = load_data()
if not df.empty:
    cols_lower = [c.lower() for c in df.columns]
    rename_map = {}
    if "cow_id" in cols_lower and "id" not in cols_lower:
        rename_map[df.columns[cols_lower.index("cow_id")]] = "id"
    if "cow_name" in cols_lower and "name" not in cols_lower:
        rename_map[df.columns[cols_lower.index("cow_name")]] = "name"
    for alt in ["Name", "names", "NAME", "CowName", "cow_name"]:
        if alt in df.columns and "name" not in rename_map.values():
            rename_map[alt] = "name"
            break
    if rename_map:
        df = df.rename(columns=rename_map)

    for c in df.columns:
        if any(x in c.lower() for x in ["date", "dob", "checkup", "vacc"]):
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
            except Exception:
                pass
    if "age" not in df.columns and "dob" in df.columns:
        df["age"] = ((pd.Timestamp.now() - df["dob"]).dt.days / 365).round(1)
    if "health_score" not in df.columns:
        df["health_score"] = df.apply(compute_health_score, axis=1)

model, model_err = try_load_model(MODEL_PATH)
inv_class_indices, map_source = load_class_indices(CLASS_INDICES_PATH, TRAIN_DIR, df)
st.sidebar.write(f"Mapping source: {map_source}")

labels = None
if inv_class_indices:
    max_idx = max(inv_class_indices.keys())
    labels = [inv_class_indices.get(i, "") for i in range(max_idx + 1)]
elif not df.empty and "name" in df.columns:
    labels = df["name"].astype(str).tolist()

# ---------------- SIDEBAR ----------------
st.sidebar.title("GaushalaNet — Control")
page = st.sidebar.radio("Navigation",
                       ["Home", "Predict (Image)", "Health Tracker", "Cow Profiles", "E-Learning", "Upload Data", "About"])

# ---------------- PAGES ----------------
if page == "Home":
    st.title("GaushalaNet — Smart Gaushala Dashboard")
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
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        pil_image = Image.open(io.BytesIO(uploaded_file.read()))
        st.image(pil_image, caption="Uploaded image", use_container_width=True)
        if model is None:
            st.error("No model loaded.")
            if model_err:
                st.code(model_err)
        else:
            if st.button("Predict"):
                try:
                    idx, conf = predict_index(model, pil_image)
                    st.write(f"Raw predicted index: {idx}, confidence: {conf}")
                    mapped_name = None
                    if inv_class_indices and idx is not None:
                        mapped_name = inv_class_indices.get(idx)
                    if mapped_name is None and labels and idx is not None and 0 <= idx < len(labels):
                        mapped_name = labels[idx]
                    if mapped_name:
                        st.success(f"Predicted Cow: **{mapped_name}** ({conf * 100:.2f}% confidence)")
                        profile = pd.DataFrame()
                        if "name" in df.columns:
                            profile = df[df["name"].astype(str).str.strip().str.lower() == mapped_name.strip().lower()]
                        if not profile.empty:
                            st.subheader("Cow Profile")
                            st.table(profile.head(1).T)
                    else:
                        st.warning("Could not map prediction to a cow name. Check mapping source or provide class_indices.json.")
                        st.info(f"Mapping source: {map_source}, inv_class_indices entries: {list(inv_class_indices.items())[:10]}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

elif page == "Health Tracker":
    st.title("Health Tracker — Alerts & Management")
    if df.empty:
        st.info("No dataset available.")
    else:
        st.markdown("#### Vaccinations due in next 30 days")
        if "vaccination_due" in df.columns:
            today = pd.Timestamp.now().normalize()
            vacc_dates = pd.to_datetime(df["vaccination_due"], errors="coerce")
            due = df[(vacc_dates - today).dt.days.between(0, 30)]
            if not due.empty:
                st.warning(f"{len(due)} animals have vaccination due soon")
                st.dataframe(due[["id", "name", "breed", "vaccination_due"]])
            else:
                st.success("No vaccinations due in next 30 days.")
        st.markdown("#### Low health score animals (score < 50)")
        low = df[df["health_score"] < 50] if "health_score" in df.columns else pd.DataFrame()
        if not low.empty:
            st.dataframe(low[["id", "name", "breed", "age", "health_score", "health_status"]].head(50))
        else:
            st.success("No animals below health threshold.")

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
    st.title("E-Learning — Knowledge Hub")
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
        new_df = pd.read_excel(uploaded)
        st.dataframe(new_df.head())
        if st.button("Save uploaded dataset"):
            new_df.to_excel(DATA_PATH, index=False)
            st.success("Saved dataset. Restart app to load new data.")

else:
    st.title("About")
    st.write("GaushalaNet — Local-only demo")
    if model is None:
        st.warning("Model not loaded.")
        if model_err:
            st.code(model_err)
