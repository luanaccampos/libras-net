import numpy as np
import mediapipe as mp
import cv2
import os
from pascal import PascalVOC
import pandas as pd


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, max_num_hands=1)

columns = ['sign', 'signer_id', 'youtube_url', 'frame', 'xmin', 'ymin', 'xmax', 'ymax', 'path_landmarks']
landmarks_columns = ['landmark_index', 'x', 'y', 'z']
Y = []

IDS = os.listdir('Annotations')
IDS.remove('.DS_Store')

for id in IDS:
    print(f'{id} started')
    yt_link = f'https://www.youtube.com/watch?v={id}'
    cap = cv2.VideoCapture(f'libras-videos/{id}.mp4')
    annotations = os.listdir(f'Annotations/{id}/Annotations')
    annotations.sort()
    
    if not os.path.exists(f'landmark_files/{id}'):
        os.mkdir(f'landmark_files/{id}')
    
    
        
    idx = 0
    
    while cap.isOpened():
        success, image_original = cap.read()
        if not success:
            break
        
        ann = PascalVOC.from_xml(f'Annotations/{id}/Annotations/{annotations[idx]}')
        idx += 1
        
        for obj in ann.objects:
            xmin, ymin, xmax, ymax = obj.bndbox
            w = xmax - xmin
            h = ymax - ymin
            
            
            image = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
            image = image[ymin:ymax, xmin:xmax]
            
            results = hands.process(image)
    
            if results.multi_hand_landmarks and results.multi_handedness[0].classification[0].label == 'Left':
                image = cv2.flip(image, 1)
                results = hands.process(image)
            
            if results.multi_hand_landmarks:
                path_landmark = f'landmark_files/{id}/{idx-1}.parquet'
                Y.append([obj.name, id, yt_link, idx-1, xmin, ymin, xmax, ymax, path_landmark])
                aux = []
                for hand_landmarks in results.multi_hand_world_landmarks:
                    for i in range(0, 21):
                        aux.append([i, hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y, hand_landmarks.landmark[i].z])
                
                pd.DataFrame(aux, columns=landmarks_columns).to_parquet(path_landmark)
    
    print(f'{id} ended')

pd.DataFrame(Y, columns=columns).to_parquet('dataset.parquet')  