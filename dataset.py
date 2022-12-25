import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from glob import glob
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
matplotlib.use('TkAgg')
IMAGE_FILES = [i for i in glob("data/*/*")]
classList = {"hand_closed": 0, "hand_open": 1, "sound_off": 2,
             "sound_on": 3, "next_track": 4, "prev_track": 5}
img_count = {"hand_closed": 0, "hand_open": 0, "sound_off": 0,
             "sound_on": 0, "next_track": 0, "prev_track": 0}
completed = []
with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:
    imgs = []
    for idx, file in enumerate(IMAGE_FILES):
        flag = False
        for i in completed:
            if i in file:
                flag = True
        if flag:
            continue

        keypoints = []
        image = cv2.flip(cv2.imread(file), 1)

        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_hand_landmarks:
            continue
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()

        for hand_landmarks in results.multi_hand_landmarks:
            wrist_point = 0
            counter = 1
            for data_point in hand_landmarks.landmark:
                if counter == 1:
                    wrist_point = abs(np.mean([data_point.x, data_point.y]))
                    counter = 0
                    continue

                keypoints.append(
                    abs(wrist_point-abs(np.mean([data_point.x, data_point.y])))
                )

            keypoints = preprocessing.normalize([keypoints])
            keypoints = keypoints.tolist()[0]

        for i in classList:
            if i in file:
                img_count[i] += 1
                if img_count[i] >= 800:
                    completed.append(i)
                keypoints.append(classList[i])
                break

        imgs.append({file: keypoints})

data = {}
for i in range(20):
    data[i] = []
data["Class"] = []
temp_imgs = imgs.copy()
for i in temp_imgs:
    for s in i.values():
        if len(s) != 21:
            imgs.remove(i)


for img in imgs:
    for s in img.values():
        for i in range(20):
            data[i].append(s[i])
        data["Class"].append(s[-1])

df = pd.DataFrame(data)

print(img_count)

df.to_csv("data.csv")
