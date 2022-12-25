import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

cap = cv2.VideoCapture(0)

detector = HandDetector(detectionCon=0.9)

counter = 0
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands = detector.findHands(img, False)

    if hands:
        counter += 1
        for i in hands:
            if counter <= 1000:
                print(counter)
                cv2.imwrite("data/next_track/file"+str(counter)+".jpg", img[i["bbox"][1]-30:i["bbox"][1]+i["bbox"][3]+40,
                                                                            i["bbox"][0]-30:i["bbox"][0]+i["bbox"][2]+30])

    cv2.imshow("Image", img)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
