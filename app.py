import streamlit as st
from PIL import Image
import numpy as np
import joblib
from ultralytics import YOLO
from fitur_ekstraksi import ekstrak_fitur_gabungan, visualisasi_prediksi
import io
import pandas as pd
import base64
import seaborn as sns
import matplotlib.pyplot as plt
import tempfile
import zipfile
import os

# Set page config
st.set_page_config(page_title="Klasifikasi Pisang", page_icon="🍌", layout="wide")

# Load model dan preprocessing
model_svm = joblib.load("model/model_svm.pkl")
model_rf = joblib.load("model/model_rf.pkl")
scaler = joblib.load("model/scaler.pkl")
encoder = joblib.load("model/label_encoder.pkl")
yolo_model = YOLO("yolov8s.pt")

# CSS untuk gaya
st.markdown("""
<style>
    .main {
        background-color: #fffceb;
    }
    h1, h2, h3 {
        color: #4b3f00;
    }
    .stButton>button {
        background-color: #fdd835;
        color: black;
        border-radius: 8px;
        font-weight: bold;
    }
    .stDownloadButton>button {
        background-color: #aed581;
        color: black;
        border-radius: 8px;
    }
    .highlight {
        background-color: #fff176;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🍌 Klasifikasi Tingkat Kematangan Pisang")
st.write("Upload satu atau beberapa gambar berisi pisang. Aplikasi ini akan mendeteksi dan mengklasifikasikan tingkat kematangan setiap buah pisang dalam gambar.")

model_choice = st.radio("🧠 Pilih Model Klasifikasi:", ("SVM", "Random Forest"))
uploaded_files = st.file_uploader("📂 Upload Gambar atau .zip File", type=["jpg", "jpeg", "png", "zip"], accept_multiple_files=True)

image_files = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".zip"):
            with tempfile.TemporaryDirectory() as tmpdirname:
                zip_path = os.path.join(tmpdirname, uploaded_file.name)
                with open(zip_path, "wb") as f:
                    f.write(uploaded_file.read())
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdirname)
                for root, dirs, files in os.walk(tmpdirname):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_files.append(Image.open(os.path.join(root, file)).convert("RGB"))
        else:
            image_files.append(Image.open(uploaded_file).convert("RGB"))

if image_files:
    all_results = []
    for idx, img in enumerate(image_files):
        st.markdown(f"---\n### 📷 Gambar #{idx+1}")
        img_np = np.array(img)

        with st.expander("🔍 Lihat Gambar Asli"):
            st.image(img, caption="Gambar Asli", use_column_width=True)

        st.info("⏳ Memproses deteksi dan klasifikasi...")

        hasil_deteksi = yolo_model.predict(img, conf=0.4, verbose=False)
        boxes, labels, scores = [], [], []

        for hasil in hasil_deteksi:
            for box in hasil.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                crop = img.crop((x1, y1, x2, y2))
                fitur = ekstrak_fitur_gabungan(crop)
                fitur_scaled = scaler.transform([fitur])

                if model_choice == "Random Forest":
                    probs = model_rf.predict_proba(fitur_scaled)[0]
                    pred_idx = np.argmax(probs)
                    confidence = probs[pred_idx]
                else:
                    probs = model_svm.predict_proba(fitur_scaled)[0]
                    pred_idx = np.argmax(probs)
                    confidence = probs[pred_idx]

                label = encoder.inverse_transform([pred_idx])[0]
                boxes.append((x1, y1, x2, y2))
                labels.append(label)
                scores.append(confidence)

        hasil_gambar = visualisasi_prediksi(img_np, boxes, labels, scores)
        st.image(hasil_gambar, caption="🔍 Hasil Deteksi & Klasifikasi", use_column_width=True)
        st.success(f"🎉 Deteksi selesai. Total pisang terdeteksi: {len(boxes)}")

        df_prediksi = pd.DataFrame({
            "Label": labels,
            "Confidence": [round(s, 4) for s in scores],
            "Koordinat (x1, y1, x2, y2)": boxes
        })

        with st.expander("📊 Tabel Hasil Klasifikasi"):
            for idx, row in df_prediksi.iterrows():
                label_color = "🟢" if row["Label"] == "ripe" else ("🟡" if row["Label"] == "unripe" else "🔴")
                st.markdown(f"<span class='highlight'>{label_color} Label:</span> {row['Label']} | <span class='highlight'>Confidence:</span> {row['Confidence']*100:.1f}%",
                            unsafe_allow_html=True)

        df_prediksi.insert(0, "Gambar", f"Gambar_{idx+1}")
        all_results.append((hasil_gambar, df_prediksi))

    # Gabungkan semua dataframe dan tampilkan tombol unduh CSV
    df_all = pd.concat([r[1] for r in all_results], ignore_index=True)
    csv_bytes = df_all.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📄 Unduh Semua Tabel Hasil (.csv)",
        data=csv_bytes,
        file_name="hasil_klasifikasi_pisang_semua.csv",
        mime="text/csv"
    )

    # Tambahkan Histogram Confidence Score
    st.subheader("📉 Histogram Confidence Score")
    plt.figure(figsize=(8, 4))
    sns.histplot(df_all["Confidence"], bins=10, kde=True, color="orange")
    plt.xlabel("Confidence Score")
    plt.ylabel("Jumlah Prediksi")
    plt.grid(True)
    st.pyplot(plt.gcf())
