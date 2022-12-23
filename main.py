import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
from glob import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
# For static images:
import numpy as np
IMAGE_FILES = [i for i in glob("data/asl_dataset/*/*")]
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  imgs = []
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    keypoints = []
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.

    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()

    for hand_landmarks in results.multi_hand_landmarks:
      counter = 1
      for data_point in hand_landmarks.landmark:
        if counter == 1:
          counter= 0
          continue
        keypoints.append(
          np.mean([data_point.x,data_point.y])
        )

      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS)
    keypoints.append(0 if "hand_closed" in file else 1)
    imgs.append({file:keypoints})


data = {}
for i in range(20):
  data[i] = []
data["Class"] = []
temp_imgs = imgs.copy()
for i in temp_imgs:
  for s in i.values():
    if len(s)!=21:
      imgs.remove(i)


for img in imgs:
  for s in img.values():
    for i in range(20):
      data[i].append(s[i])
    data["Class"].append(s[-1])

df = pd.DataFrame(data)

print(df.head(10))

df.to_csv("data.csv")

