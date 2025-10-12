import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import winsound
import datetime

print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)
print("TensorFlow version:", tf.__version__)

# Charger le modèle entraîné
model = load_model(r"C:\Users\IMANE\Desktop\real-time-drowsiness-detection\ModelDrowsiness.h5")

# Classes du dataset
classes = ['Drowsy', 'Non Drowsy']

# Haar Cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Webcam
cap = cv2.VideoCapture(0)

# Drapeaux de capture
captured_drowsy = False
captured_nondrowsy = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            # Sélectionner les deux yeux les plus grands
            eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]

            # Calculer la région englobant les deux yeux
            ex1, ey1, ew1, eh1 = eyes[0]
            ex2, ey2, ew2, eh2 = eyes[1]

            x_min = min(ex1, ex2)
            y_min = min(ey1, ey2)
            x_max = max(ex1+ew1, ex2+ew2)
            y_max = max(ey1+eh1, ey2+eh2)

            # Extraire et préparer la région des yeux pour le modèle
            eyes_region = roi_color[y_min:y_max, x_min:x_max]
            eyes_region = cv2.resize(eyes_region, (299, 299))
            eyes_region = eyes_region / 255.0
            eyes_region = np.expand_dims(eyes_region, axis=0)

            # Prédiction globale sur les deux yeux
            pred = model.predict(eyes_region, verbose=0)
            prob = pred[0][0]
            class_idx = 1 if prob > 0.5 else 0
            label_text = f"{classes[class_idx]}"

            # Dessiner un cadre bleu autour du visage + label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, label_text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Alerte sonore si Drowsy
            if class_idx == 0 and prob > 0.6:
                winsound.Beep(1000, 500)

            # Capture automatique une seule fois
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if class_idx == 0 and not captured_drowsy:
                cv2.imwrite(f"Drowsy_{timestamp}.png", frame)
                captured_drowsy = True
            elif class_idx == 1 and not captured_nondrowsy:
                cv2.imwrite(f"NonDrowsy_{timestamp}.png", frame)
                captured_nondrowsy = True

            # Délai pour lisibilité
            time.sleep(0.1)

    cv2.imshow("Drowsiness Detection - Temps Réel (2 yeux ensemble, sans contour yeux)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
