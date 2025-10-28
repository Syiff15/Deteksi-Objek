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
    uploaded_files = st.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        st.markdown(f"<h4 style='color:#966543;'>{t('📸 Hasil Petualangan Kamu','📸 Your Adventure Results')}</h4>", unsafe_allow_html=True)
        mode = st.session_state.get("mode", "deteksi")

    multiple = len(uploaded_files) > 1
    cols = st.columns(2) if multiple else [None]

    for i, file in enumerate(uploaded_files):
        col = cols[i % len(cols)] if multiple else None
        img_container = col if col else st  # <- ini penting

        # tidak perlu pakai "with context" lagi
        img_container.markdown(
            "<div style='background-color:#FFF8E7; border-radius:20px; "
            "box-shadow:0 4px 12px rgba(0,0,0,0.1); padding:20px; "
            "margin-bottom:20px; text-align:center;'>",
            unsafe_allow_html=True
        )

        img = Image.open(file).convert("RGB")
        img_container.image(img, caption=f"🖼️ {file.name}", use_container_width=True)

        # --- Statistik & Tabel Hasil ---
        dets = []
        if mode == "deteksi" and boxes is not None and len(boxes) > 0:
            boxes_array = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            names = results[0].names if hasattr(results[0], "names") else {}
            for b, s, c in zip(boxes_array, scores, classes):
                class_name = names.get(int(c), str(c))
                dets.append({
                    "Kelas": class_name,
                    "Confidence": float(s),
                    "Bounding Box": f"({int(b[0])},{int(b[1])},{int(b[2])},{int(b[3])})",
                    "Akurasi": f"{float(s):.1%}"
                })
        elif mode == "klasifikasi":
            dets.append({
                "Kelas": predicted_label,
                "Confidence": float(confidence),
                "Bounding Box": "-",
                "Akurasi": f"{confidence:.1%}"
            })

        img_container.markdown("<hr>", unsafe_allow_html=True)

        # Statistik ringkas
        panda_count = sum(1 for d in dets if "panda" in d["Kelas"].lower())
        bear_count = sum(1 for d in dets if "beruang" in d["Kelas"].lower())
        avg_conf = np.mean([d["Confidence"] for d in dets]) if dets else 0

        c1, c2, c3, c4 = img_container.columns(4)
        c1.metric(t("Akurasi Rata-rata","Avg Confidence"), f"{avg_conf:.1%}")
        c2.metric(t("Jumlah Panda","Panda Count"), panda_count)
        c3.metric(t("Jumlah Beruang","Bear Count"), bear_count)
        c4.metric(t("Waktu Inferensi (s)","Processing Time (s)"), f"{st.session_state.get('process_time',0):.2f}")

        # Tabel hasil
        if dets:
            df = pd.DataFrame(dets)
            img_container.markdown(f"<h5 style='color:#966543;'>{t('📋 Detail Hasil','📋 Detection/Classification Details')}</h5>", unsafe_allow_html=True)
            img_container.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            img_container.download_button(
                t("📥 Download CSV","📥 Download CSV"),
                data=csv,
                file_name=f"detections_{file.name}.csv",
                mime="text/csv"
            )

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
