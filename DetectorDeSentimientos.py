import os
import cv2
import numpy as np
from keras.models import model_from_json

emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

# Cargar moodelo
json_file = open('./model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

emotion_model.load_weights("./model.h5")
print("Loaded model from disk")

def process_face(frame, x, y, w, h):
    # Obtener recorte de rostro de imagen de camara
    cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
    roi_gray_frame = gray_frame[y:y + h, x:x + w]
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

    # Predecir Emocion utilizando modelo de CNN
    emotion_prediction = emotion_model.predict(cropped_img)
    maxindex = int(np.argmax(emotion_prediction))

    # Mostrar resultado en pantalla
    cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))

    if not ret:
        print('ERROR al iniciar camara')
        break

    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar la caras en el video.
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Procesar con red CNN
    for (x, y, w, h) in num_faces:
        process_face(frame, x, y, w, h)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1):
        break

cap.release()
cv2.destroyAllWindows()
