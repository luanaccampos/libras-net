import pickle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import mediapipe as mp
import cv2

model = pickle.load(open('alphabet_model.pickle', "rb"))

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

hands = mp_hands.Hands(model_complexity=0, max_num_hands=1)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())


        for hand_landmarks in results.multi_hand_world_landmarks:
            aux = []
            for i in range(0, 21):
                aux.append(hand_landmarks.landmark[i].x)
                aux.append(hand_landmarks.landmark[i].y)
                aux.append(hand_landmarks.landmark[i].z)
            
            aux = np.array(aux).reshape(1, -1)
            
            pred = model.predict(aux)
            print(pred)

    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
        break