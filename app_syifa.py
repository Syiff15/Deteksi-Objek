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

    # --- Judul Selamat Datang ---
    st.markdown(f"""
    <div style='
        background-color:#f2e6d6;
        padding:25px;
        border-radius:15px;
        box-shadow:0 4px 15px rgba(0,0,0,0.1);
        text-align:center;
        margin-bottom:25px;
    '>
        <h1 style='color:#966543; margin-bottom:10px;'>
            {t('Hai', 'Hi')}, <span style='text-transform:capitalize;'>{st.session_state.name.lower().split()[0]}</span>! 👋
        </h1>
        <p style='font-size:18px; color:#5b4636;'>
            {t('Selamat datang di markas petualangan <b>Ursidetect</b>!',
               'Welcome to the adventure base of <b>Ursidetect</b>!')}<br>
            {t('Pilih mode favoritmu — mau jadi <b>pemburu hewan</b> (deteksi) atau <b>peneliti hewan</b> (klasifikasi)?',
               'Choose your mode — be a <b>Wildlife Hunter</b> (detection) or <b>Animal Researcher</b> (classification)?')}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<h4 style='color:#966543; text-align:center;'>{t('Pilih Mode Petualang:','Choose Your Adventure Mode:')}</h4>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")

    # --- Kolom kiri: Deteksi ---
    with col1:
        deteksi_clicked = st.button("🐾 Pemburu Hewan", key="deteksi_mode", use_container_width=True)
        css_class = "mode-card selected" if st.session_state.get("mode") == "deteksi" else "mode-card"
        st.markdown(f"""
        <div class="{css_class}">
            <h4 style='color:#966543;'>🐾 {t('Pemburu Hewan','Wildlife Hunter')}</h4>
            <p style='color:#5b4636; font-size:14px;'>
                {t('Mode <b>Deteksi</b> untuk menemukan lokasi panda dan beruang di gambar.',
                   '<b>Detection</b> mode to find pandas and bears in an image.')}
            </p>
        </div>
        """, unsafe_allow_html=True)
        if deteksi_clicked:
            st.session_state.mode = "deteksi"
            st.rerun()

    # --- Kolom kanan: Klasifikasi ---
    with col2:
        klasifikasi_clicked = st.button("🔬 Peneliti Hewan", key="klasifikasi_mode", use_container_width=True)
        css_class = "mode-card selected" if st.session_state.get("mode") == "klasifikasi" else "mode-card"
        st.markdown(f"""
        <div class="{css_class}">
            <h4 style='color:#966543;'>🔬 {t('Peneliti Hewan','Animal Researcher')}</h4>
            <p style='color:#5b4636; font-size:14px;'>
                {t('Mode <b>Klasifikasi</b> untuk mengenali apakah itu panda atau beruang.',
                   '<b>Classification</b> mode to recognize whether it’s a panda or a bear.')}
            </p>
        </div>
        """, unsafe_allow_html=True)
        if klasifikasi_clicked:
            st.session_state.mode = "klasifikasi"
            st.rerun()

    st.divider()

    # =========================
    # UPLOAD GAMBAR
    # =========================
    st.markdown(f"<h4 style='color:#966543;'>{t('🖼️ Masukkan Gambar','🖼️ Upload Image')}</h4>", unsafe_allow_html=True)
    st.caption(t("Kamu bisa mengunggah satu atau beberapa gambar (jpg, jpeg, png).",
                 "You can upload one or more images (jpg, jpeg, png)."))
    uploaded_files = st.file_uploader("", type=["jpg","jpeg","png"], accept_multiple_files=True)

    if uploaded_files:

        # --- Load Model SEKALI ---
        yolo_model, classifier = load_models()  # sudah didefinisikan di awal dengan @st.cache_resource

        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")

            col1, col2 = st.columns(2)

            # =========================
            # MODE DETEKSI
            # =========================
            if st.session_state.mode == "deteksi":
                results = yolo_model(image)
                detected_img = results[0].plot()  # hasil deteksi digambar

                with col1:
                    st.image(image, caption=uploaded_file.name, use_column_width=True)
                    st.markdown("<p style='text-align:center; color:#7B4F27;'>Gambar yang diunggah</p>", unsafe_allow_html=True)

                with col2:
                    st.image(detected_img, caption="Hasil Deteksi", use_column_width=True)
                    # Tampilkan label
                    labels = [yolo_model.names[int(c)] for c in results[0].boxes.cls.numpy()] if len(results[0].boxes) > 0 else []
                    if labels:
                        st.success(f"🎯 Objek terdeteksi: {', '.join(labels)}")
                    else:
                        st.warning("⚠️ Tidak ada objek panda atau beruang yang terdeteksi.")

            # =========================
            # MODE KLASIFIKASI
            # =========================
            elif st.session_state.mode == "klasifikasi":
                img_array = np.array(image.resize((224,224)))/255.0
                img_array = np.expand_dims(img_array, axis=0)
                pred = classifier.predict(img_array)
                class_idx = np.argmax(pred, axis=1)[0]
                class_names = ["Panda", "Beruang"]  # sesuaikan dengan modelmu
                confidence = pred[0][class_idx]

                with col1:
                    st.image(image, caption=uploaded_file.name, use_column_width=True)
                    st.markdown("<p style='text-align:center; color:#7B4F27;'>Gambar yang diunggah</p>", unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div style='background-color:#f2e6d6; padding:20px; border-radius:15px;
                    box-shadow:0 4px 15px rgba(0,0,0,0.1); text-align:center;'>
                        <h4 style='color:#6B4226; margin-bottom:10px;'>🔬 Hasil Klasifikasi</h4>
                        <p style='color:#7B4F27; font-size:16px;'>
                            {class_names[class_idx]} ({confidence*100:.2f}%)
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

        # Tombol lanjut
        col_kiri, col_kanan = st.columns([4,1])
        with col_kanan:
            if st.button(t("Lanjutkan 🐾","Continue 🐾"), key="lanjutkan_step4"):
                st.session_state.step = 4
                st.rerun()

    else:
        st.info("⬆️ Silakan unggah gambar terlebih dahulu untuk memproses objek.")

# === STEP 3 ===
elif st.session_state.step == 3:
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
