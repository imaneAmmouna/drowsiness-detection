import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)
print("TensorFlow version:", tf.__version__)

# Charger le modèle entraîné
model = load_model(r"C:\Users\IMANE\Desktop\real-time-drowsiness-detection\ModelDrowsiness.h5")

# Classes du dataset
classes = ['Drowsy', 'Non Drowsy']

# Charger les Haar Cascades pour visage et yeux
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Ouvrir la webcam (0 = webcam par défaut)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Capture une image depuis la webcam
    if not ret:
        break

    # Convertir l'image en niveaux de gris pour Haar Cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les visages
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Pour chaque visage détecté
    for (x, y, w, h) in faces:
        # Dessiner un rectangle autour du visage
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extraire la région du visage
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Détecter les yeux dans le visage
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # Dessiner un rectangle autour des yeux
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            # Extraire l'image de l'œil pour la prédiction
            eye_img = roi_color[ey:ey+eh, ex:ex+ew]

            # Redimensionner pour correspondre à l'entrée du modèle
            eye_img = cv2.resize(eye_img, (299, 299))  # InceptionV3 input size
            eye_img = eye_img / 255.0  # Normalisation
            eye_img = np.expand_dims(eye_img, axis=0)  # Ajouter une dimension batch

            # Faire la prédiction
            pred = model.predict(eye_img)
            prob = pred[0][0]  # probabilité de Drowsy
            class_idx = 1 if prob > 0.5 else 0
            cv2.putText(frame, f"{classes[class_idx]} ({prob:.2f})", (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

            # Afficher le résultat sur la vidéo
            label = classes[class_idx]
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Afficher la vidéo avec les rectangles et les labels
    cv2.imshow("Drowsiness Detection - Temps Réel", frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la caméra et fermer la fenêtre
cap.release()
cv2.destroyAllWindows()
