import os
# Lazımi kitabxanalar daxil edilir
import os
import argparse  # Komanda sətri arqumentləri üçün
import cv2       # OpenCV – görüntü və video işləmək üçün
import mediapipe as mp  # Google-un MediaPipe üz tanıma kitabxanası

# Üzə blur tətbiq edən funksiyanı tərif edirik
def process_img(img, face_detection):
    H, W, _ = img.shape  # Şəkilin hündürlüyü və eni

    # BGR formatından RGB formatına keçirik (MediaPipe üçün lazım)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)  # Üz aşkarlanması

    # Əgər üzlər tapılıbsa:
    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box  # Nisbi koordinatlar (0-1 arası)

            # Koordinatları orijinal ölçülərə çevirmək
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # Blur tətbiq edirik: üz sahəsinə gaussian blur
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (30, 30))

    return img


# Komanda sətrindən istifadəçi tərəfindən veriləcək parametrləri oxuyuruq
args = argparse.ArgumentParser()

# Mode seçimi: webcam, image, video
args.add_argument("--mode", default='webcam')
# Fayl yolu (əgər image və ya video rejimindədirsə)
args.add_argument("--filePath", default=None)

# Parametrləri parse edirik
args = args.parse_args()

# Nəticələrin saxlanacağı qovluğu yaradırıq
output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# MediaPipe üz tanıma modulu yüklənir
mp_face_detection = mp.solutions.face_detection

# Üz aşkarlama obyektini istifadə edərək aşağıdakı rejimlərdən birini işlədirik
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    # ------------------ ŞƏKİL REJİMİ ------------------
    if args.mode in ["image"]:
        img = cv2.imread(args.filePath)  # şəkil oxunur

        img = process_img(img, face_detection)  # üzlərə blur tətbiq edilir

        # Nəticəni fayla yazırıq
        cv2.imwrite(os.path.join(output_dir, 'output.png'), img)

    # ------------------ VİDEO REJİMİ ------------------
    elif args.mode in ['video']:
        cap = cv2.VideoCapture(args.filePath)  # video oxunur
        ret, frame = cap.read()

        # Yeni video faylı üçün yazıcı obyekt yaradılır
        output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'),
                                       cv2.VideoWriter_fourcc(*'MP4V'),
                                       25,  # FPS
                                       (frame.shape[1], frame.shape[0]))  # ölçü

        while ret:
            frame = process_img(frame, face_detection)  # blur tətbiq edilir
            output_video.write(frame)  # nəticə yazılır
            ret, frame = cap.read()  # növbəti kadr

        cap.release()
        output_video.release()  # faylı bağlayırıq

    # ------------------ WEBCAM REJİMİ ------------------
    elif args.mode in ['webcam']:
        cap = cv2.VideoCapture(2)  # 0 və ya 1 edə bilərsən — cihazdan asılı olaraq dəyişir

        ret, frame = cap.read()
        while ret:
            frame = process_img(frame, face_detection)  # blur tətbiq edilir

            # Canlı görüntünü göstərmək
            cv2.imshow('frame', frame)
            cv2.waitKey(25)  # 25ms gözləmə

            ret, frame = cap.read()  # növbəti kadr

        cap.release()  # webcam-i azad edirik
