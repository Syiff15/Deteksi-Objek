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
    # ====================================================
    # 🟫 PENYAMBUT & PILIH MODE PETUALANG
    # ====================================================
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

    # --- CSS interaktif untuk kotak mode ---
    st.markdown("""
    <style>
    .mode-card {
        background-color: #f2e6d6;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        cursor: pointer;
    }
    .mode-card:hover {
        transform: scale(1.03);
        box-shadow: 0 6px 15px rgba(0,0,0,0.15);
    }
    .selected {
        background-color: #e1c9aa !important;
        border: 3px solid #966543;
    }
    </style>
    """, unsafe_allow_html=True)

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

    # ====================================================
    # 🟤 UPLOAD GAMBAR
    # ====================================================
    st.markdown(f"<h4 style='color:#966543;'>{t('🖼️ Masukkan Gambar','🖼️ Upload Image')}</h4>", unsafe_allow_html=True)
    st.caption(t("Kamu bisa mengunggah satu atau beberapa gambar (jpg, jpeg, png).",
                 "You can upload one or more images (jpg, jpeg, png)."))
    uploaded_files = st.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # ====================================================
    # 🟠 HASIL DETEKSI / KLASIFIKASI
    # ====================================================
    if uploaded_files:
        st.markdown(f"<h4 style='color:#966543;'>{t('📸 Hasil Petualangan Kamu','📸 Your Adventure Results')}</h4>", unsafe_allow_html=True)
        mode = st.session_state.get("mode", "deteksi")
        cols = st.columns(2) if len(uploaded_files) > 1 else [st]

        for i, file in enumerate(uploaded_files):
            col = cols[i % len(cols)]
            with col:
                # --- Wrapper hasil ---
                st.markdown("""
                    <div style='
                        background-color:#FFF8E7;
                        border-radius:20px;
                        box-shadow:0 4px 12px rgba(0,0,0,0.1);
                        padding:20px;
                        margin-bottom:20px;
                        text-align:center;
                    '>
                """, unsafe_allow_html=True)

                img = Image.open(file).convert("RGB")
                st.image(img, caption=f"🖼️ {file.name}", use_container_width=True)

                # --- MODE DETEKSI ---
                if mode == "deteksi":
                    with st.spinner(t(f"🔍 Mendeteksi objek pada {file.name}...", f"🔍 Detecting objects in {file.name}...")):
                        results = yolo_model(img)
                        boxes = results[0].boxes
                        if boxes is not None and len(boxes) > 0:
                            result_img = results[0].plot()
                            st.markdown("""
                                <div style="
                                    background:white;
                                    border-left: 6px solid #966543;
                                    border-radius:14px;
                                    padding:22px;
                                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                                    margin-top:25px;
                                ">
                                    <h3 style="color:#966543; margin-bottom:10px;">🔍 Hasil Deteksi Objek</h3>
                                    <p style="color:#5b4636; font-size:15px;">
                                        Sistem AI Vision berhasil mendeteksi objek pada gambar berikut menggunakan model YOLO.
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)
                            st.image(result_img, caption="🖼️ Hasil Deteksi", use_container_width=True)
                            st.success(t("✅ Objek berhasil terdeteksi!", "✅ Object detected successfully!"))
                        else:
                            st.warning(t("🚫 Tidak ada objek yang terdeteksi.", "🚫 No objects detected."))

                # --- MODE KLASIFIKASI ---
                elif mode == "klasifikasi":
                    with st.spinner(t(f"🧠 Mengklasifikasi {file.name}...", f"🧠 Classifying {file.name}...")):
                        target_size = classifier.input_shape[1:3]
                        img_resized = img.resize(target_size)
                        img_array = image.img_to_array(img_resized)
                        img_array = np.expand_dims(img_array, axis=0).astype("float32") / 255.0

                        prediction = classifier.predict(img_array)
                        predicted_class = np.argmax(prediction, axis=1)[0]
                        confidence = np.max(prediction)

                        class_names = [
                            "AMERICAN GOLDFINCH",
                            "BARN OWL",
                            "CARMINE BEE-EATER",
                            "DOWNY WOODPECKER",
                            "EMPEROR PENGUIN",
                            "FLAMINGO"
                        ]
                        predicted_label = class_names[predicted_class]

                        st.image(img_resized, caption="📸 Gambar yang Dianalisis", use_column_width=True)
                        st.markdown(f"""
                            <div style="
                                background:white;
                                border-left: 6px solid #966543;
                                border-radius:14px;
                                padding:22px;
                                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                                margin-top:25px;
                            ">
                                <h3 style="color:#966543; margin-bottom:8px;">🧠 Hasil Klasifikasi Gambar</h3>
                                <p style="color:#5b4636; font-size:15px;">
                                    Model AI Vision mengenali gambar sebagai spesies:
                                    <b style="color:#966543;">{predicted_label}</b>
                                </p>
                                <p style="color:#5b4636; margin-top:6px;">
                                    Confidence Score: <b>{confidence:.2f}</b>
                                </p>
                            </div>
                        """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

        # ====================================================
        # 🟢 TOMBOL LANJUT
        # ====================================================
        col_kiri, col_kanan = st.columns([4, 1])
        with col_kanan:
            if st.button(t("Lanjutkan 🐾", "Continue 🐾")):
                st.session_state.step = 4
                st.rerun()
                
    col_kiri, col_kanan = st.columns([4, 1])
    with col_kanan:
        if st.button(t("Lanjutkan 🐾", "Continue 🐾")):
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
