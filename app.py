import os
import cv2
import mediapipe as mp
import streamlit as st
import tempfile
import numpy as np

# Streamlit başlıq
st.title("Face Detection & Blur App (MediaPipe ilə)")
st.sidebar.title("Rejim seçin")
mode = st.sidebar.selectbox("Rejim:", ["Webcam", "Image", "Video"])

# MediaPipe setup
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Blur funksiyası
def process_img(img):
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            bbox = detection.location_data.relative_bounding_box
            x1 = int(bbox.xmin * W)
            y1 = int(bbox.ymin * H)
            w = int(bbox.width * W)
            h = int(bbox.height * H)
            img[y1:y1+h, x1:x1+w, :] = cv2.blur(img[y1:y1+h, x1:x1+w, :], (30, 30))
    return img

# 1️⃣ IMAGE Mode
if mode == "Image":
    file = st.sidebar.file_uploader("Şəkil yüklə", type=['jpg', 'jpeg', 'png'])

    if file is not None:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Orijinal", use_column_width=True)

        result = process_img(img.copy())
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Blur tətbiq olunmuş", use_column_width=True)

# 2️⃣ VIDEO Mode
elif mode == "Video":
    file = st.sidebar.file_uploader("Video yüklə", type=["mp4", "mov", "avi"])
    
    if file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            result = process_img(frame)
            stframe.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()

# 3️⃣ WEBCAM Mode
elif mode == "Webcam":
    st.warning("Streamlit-in native webcam dəstəyi yoxdur, amma 'streamlit-webrtc' ilə əlavə etmək mümkündür. İstəyirsənsə onu da izah edim.")

