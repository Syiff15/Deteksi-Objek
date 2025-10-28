# -*- coding: utf-8 -*-
"""App_Syifa_Bilingual.py"""

#pip install streamlit
#pip install ultralytics

# === Import Library ===
import streamlit as st
import time
from PIL import Image
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# === LOAD MODELS ===
@st.cache_resource
def load_models():
    yolo_model = YOLO("Model/best.pt")  # Model deteksi Gambar
    classifier = tf.keras.models.load_model("Model/Syifa Salsabila_Laporan 2.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# === KONFIGURASI DASAR HALAMAN WEB ===
st.set_page_config(page_title="Ursidetect", page_icon="🐻🐼", layout="centered")

# === STATE MANAGEMENT ===
if "step" not in st.session_state:
    st.session_state.step = 0
if "language" not in st.session_state:
    st.session_state.language = "id"
if "name" not in st.session_state:
    st.session_state.name = ""

# === Fungsi Bantu Terjemahan ===
def t(id_text, en_text):
    return id_text if st.session_state.get("language") == "id" else en_text

# === STEP 0: PILIH BAHASA ===
if "step" not in st.session_state:
    st.session_state.step = 0

if st.session_state.step == 0:
    st.image("slide 1.jpg", use_container_width=True)

    # 🪄 Tambah jarak dari logo ke teks
    st.markdown("<div style='margin-top:30px;'></div>", unsafe_allow_html=True)

    # === Judul Pilihan Bahasa ===
    st.markdown(
        "<h3 style='text-align:center; color:#966543;'>🌐 Pilih Bahasa / Choose Language</h3>",
        unsafe_allow_html=True,
    )

    # === CSS Styling ===
    st.markdown("""
    <style>
    .lang-container {
        display: flex;
        justify-content: center;
        gap: 80px;
        margin-top: 10px;
    }
    .lang-option {
        background-color: #f2e6d6;
        width: 350px;
        height: 130px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        font-weight: 700;
        color: #2f2f2f;
        text-align: center;
        font-size: 20px;
        border: 2px solid transparent;
    }
    .lang-option:hover {
        transform: scale(1.05);
        background-color: #e9dcc6;
        border: 2px solid #d2b48c;
    }
    </style>
    """, unsafe_allow_html=True)

    # === Dua Tombol Bahasa ===
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🇮🇩  ID BAHASA INDONESIA", use_container_width=True):
            st.session_state.language = "id"
            st.session_state.step = 1
            st.rerun()

    with col2:
        if st.button("🇬🇧  EN ENGLISH", use_container_width=True):
            st.session_state.language = "en"
            st.session_state.step = 1
            st.rerun()

# === STEP 1 ===
elif st.session_state.step == 1:
    st.image("slide 1.jpg", use_container_width=True)

    # Judul & deskripsi
    st.markdown(f"""
    <h1 style='text-align:center; color:#1E1E1E;'>
        {t('Selamat datang di', 'Welcome to')} 
        <span style="color:#966543;">Ursidetect</span>
    </h1>
    <p style='text-align:center; font-size:18px; color:#1E1E1E;'>
        {t('Sebelum kita mulai berpetualang, kenalan dulu yuk dengan', 'Before we start our adventure, let’s get to know')}
        <b>Ursidetect</b>!<br>
        {t('Ursidetect adalah platform berbasis <b>kecerdasan buatan (AI)</b> yang dirancang untuk <b>mendeteksi</b> dan <b>mengklasifikasikan</b> hewan <b>panda</b> serta <b>beruang</b>.',
           'Ursidetect is an <b>AI-based</b> platform designed to <b>detect</b> and <b>classify</b> <b>pandas</b> and <b>bears</b>.')}
    </p>
    """, unsafe_allow_html=True)

    # Styling dua kotak fitur
    st.markdown("""
    <style>
    .feature-container {
        display: flex;
        justify-content: center;
        gap: 80px;
        margin-top: 40px;
    }

    .feature-box {
        background-color: #f2e6d6;
        width: 400px;
        height: 220px;
        border-radius: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        padding: 25px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        transition: all 0.3s ease;
    }

    .feature-box:hover {
        transform: scale(1.03);
        background-color: #e9dcc6;
        border: 2px solid #d2b48c;
    }

    .feature-title {
        font-size: 22px;
        font-weight: 700;
        color: #2f2f2f;
        margin-bottom: 10px;
    }

    .feature-text {
        font-size: 16px;
        color: #3b3b3b;
        line-height: 1.4;
    }
    </style>
    """, unsafe_allow_html=True)

    # Isi dua kotak fitur
    st.markdown(f"""
    <div class="feature-container">
        <div class="feature-box">
            <div class="feature-title">🐻‍❄️ {t('Deteksi Gambar','Image Detection')}</div>
            <div class="feature-text">
                {t('Ursidetect dapat menemukan dan menandai posisi panda atau beruang di dalam gambar menggunakan kotak pembatas (<i>bounding box</i>).',
                   'Ursidetect can locate and highlight the position of pandas or bears in an image using <i>bounding boxes</i>.')}
            </div>
        </div>
        <div class="feature-box">
            <div class="feature-title">🐼 {t('Klasifikasi Gambar','Image Classification')}</div>
            <div class="feature-text">
                {t('Ursidetect dapat menentukan apakah gambar tersebut termasuk panda atau beruang, lengkap dengan tingkat kepercayaan (<i>confidence score</i>).',
                   'Ursidetect can determine whether an image shows a panda or a bear, along with its <i>confidence score</i>.')}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Garis dan teks lanjut
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <p style='text-align:center; color:#282328; font-size:18px;'>
        {t('Yuk lanjut ke langkah berikutnya untuk mulai berpetualang!','Let’s continue to the next step to start the adventure!')}
    </p>
    """, unsafe_allow_html=True)

    # Tombol lanjut
    col1, col2, col3 = st.columns([4, 1, 1])
    with col3:
        if st.button(t("Lanjut 🐾", "Next 🐾")):
            st.session_state.step = 2
            st.rerun()

# === STEP 2 ===
elif st.session_state.step == 2:
    st.image("slide 3-1.png", use_container_width=True)
    st.write(t("Sekarang giliran kamu! Masukkan namamu supaya Ursidetect tahu siapa partner barunya.",
               "Now it’s your turn! Enter your name so Ursidetect knows its new partner."))

    name_input = st.text_input("", placeholder=t("Contoh: Ursi", "Example: Ursi"))
    col_kiri, col_kanan = st.columns([4, 1])
    with col_kanan:
        if st.button(t("Lanjutkan 🐾", "Continue 🐾")):
            if name_input.strip() != "":
                st.session_state.name = name_input.strip()
                st.session_state.step = 3
                st.rerun()
            else:
                st.info(t("Ups, sepertinya kamu lupa menulis namamu dulu nih 😊", "Oops, you forgot to enter your name 😊"))

# === STEP 3 ===
elif st.session_state.step == 3:
    st.title(f"Hai, {st.session_state.name.lower().split()[0]}! 👋")
    st.info("Selamat datang di markas petualangan Ursidetect!")
    st.info("Pilih mode favoritmu: mau jadi pemburu hewan (deteksi) atau peneliti hewan (klasifikasi)?")

    analysis_type = st.radio(
        "Pilih Mode Petualang:",
        ["Pemburu Hewan (Deteksi)", "Peneliti Hewan (Klasifikasi)"],
        horizontal=True,
        index=0)
    st.divider()

    st.markdown("#### Masukkan Gambar")
    st.caption(f"Untuk mulai petualangannya, {st.session_state.name.lower().split()[0]} harus memasukkan gambar berbentuk jpg, jpeg atau png yaa.")
    uploaded_file = st.file_uploader("Pilih gambar (jpg, jpeg, png):", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption=uploaded_file.name, use_container_width=True)
    
    st.divider()
    analyze_button = st.button("🔎 Mulai Petualangan", use_container_width=True)

    if analyze_button:
        st.markdown("### Hasil Petualangan")
        if not uploaded_file:
            st.warning("⚠️ Silakan masukkan gambar terlebih dahulu sebelum berpetualang.")
        else:
            with st.spinner("⏳ Sedang Berpetualang..."):
                time.sleep(2)
            st.success("⬇️ Hasil analisis muncul di sini!")

            # === Mulai: kode hasil deteksi/klasifikasi ===
            pil_img = Image.open(uploaded_file).convert("RGB")
            # Jalankan deteksi YOLO
            results = yolo_model(pil_img)

            # Ambil hasil dari frame pertama
            res = results[0]

            # Ekstraksi box / confidence / kelas dari hasil YOLO (kompatibilitas CPU/GPU)
            try:
                boxes = res.boxes.xyxy.cpu().numpy()
                scores = res.boxes.conf.cpu().numpy()
                det_classes = res.boxes.cls.cpu().numpy().astype(int)
            except Exception:
                boxes = res.boxes.xyxy.numpy()
                scores = res.boxes.conf.numpy()
                det_classes = res.boxes.cls.numpy().astype(int)

            # Salin image ke format OpenCV untuk menggambar
            cv_img = np.array(pil_img)[:, :, ::-1].copy()  # RGB -> BGR untuk cv2
            annotated = cv_img.copy()

            # Ambil nama kelas dari model deteksi jika tersedia
            try:
                det_names = yolo_model.names
            except Exception:
                det_names = {}

            # Siapkan nama kelas untuk model klasifikasi (ubah jika modelmu punya label berbeda)
            classifier_names = ["Panda", "Beruang"]

            # Cek ukuran input classifier
            try:
                in_shape = classifier.input_shape
                in_h = in_shape[1] or 224
                in_w = in_shape[2] or 224
            except Exception:
                in_h, in_w = 224, 224

            detections_summary = []
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                det_conf = float(scores[i]) if i < len(scores) else 0.0
                det_cls_idx = int(det_classes[i]) if i < len(det_classes) else -1
                det_label = det_names.get(det_cls_idx, str(det_cls_idx))

                # Gambar bounding box dan label deteksi
                color = (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                text = f"{det_label} {det_conf:.2f}"
                cv2.putText(annotated, text, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Crop untuk klasifikasi
                try:
                    crop_pil = pil_img.crop((x1, y1, x2, y2)).convert("RGB")
                    crop_resized = crop_pil.resize((in_w, in_h))
                    x = tf.keras.preprocessing.image.img_to_array(crop_resized)
                    x = x / 255.0
                    x = np.expand_dims(x, 0)
                    preds = classifier.predict(x)
                    cls_idx = int(np.argmax(preds[0]))
                    cls_conf = float(np.max(preds[0]))
                    cls_name = classifier_names[cls_idx] if cls_idx < len(classifier_names) else str(cls_idx)
                except Exception as e:
                    cls_name = "N/A"
                    cls_conf = 0.0

                # Tulis hasil klasifikasi pada gambar
                cls_text = f"{cls_name} {cls_conf:.2f}"
                cv2.putText(annotated, cls_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

                detections_summary.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "detected_label": det_label,
                    "detected_confidence": round(det_conf, 4),
                    "classified_label": cls_name,
                    "classification_confidence": round(cls_conf, 4)
                })

            # Tampilkan hasil terannotasi
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            annotated_pil = Image.fromarray(annotated_rgb)
            st.image(annotated_pil, caption="Hasil Deteksi & Klasifikasi", use_container_width=True)

            # Tampilkan ringkasan deteksi
            if len(detections_summary) == 0:
                st.info("Tidak ada objek terdeteksi.")
            else:
                st.markdown("### Ringkasan Deteksi")
                for idx, d in enumerate(detections_summary, start=1):
                    st.markdown(
                        f"- **Objek {idx}**: Deteksi = **{d['detected_label']}** ({d['detected_confidence']:.2f}) | "
                        f"Klasifikasi = **{d['classified_label']}** ({d['classification_confidence']:.2f}) | "
                        f"Box = {d['box']}"
                    )
    
    col_kiri, col_kanan = st.columns([4, 1])
    with col_kanan:
        if st.button("Lanjutkan 🐾"):
            st.session_state.step = 4
            st.rerun()

# === STEP 4 ===
elif st.session_state.step == 4:
    st.subheader(t("💬 Cerita Petualanganmu", "💬 Your Adventure Story"))
    st.info(t(
        "Petualanganmu bersama Ursidetect sudah selesai 🐾  Ceritakan pengalamanmu, ya!",
        "Your adventure with Ursidetect has ended 🐾  Tell us about your experience!"
    ))

    feedback_text = st.text_area(t("Bagaimana Petualanganmu?", "How was your adventure?"), placeholder=t("Kirimkan ceritamu di sini...", "Share your story here..."))

    if st.button(t("Kirim Cerita Petualanganku", "Send My Story")):
        if feedback_text.strip() == "":
            st.warning(t("Ceritamu sangat berarti bagi kami 😊", "Your story means a lot to us 😊"))
        else:
            st.success(f"✅ {t('Terima kasih atas ceritanya','Thank you for sharing your story')}, {st.session_state.name.lower().split()[0]}!")

    st.markdown("---")
    if st.button(t("🔁 Mau memulai lagi?", "🔁 Start again?")):
        st.session_state.step = 0
        st.session_state.name = ""
        st.rerun()
