import mediapipe as mp
mp_hands = mp.solutions.hands
import cv2
import pyautogui
import numpy as np
import asyncio
import pickle
import winsdk.windows.media.control as wmc
pyautogui.FAILSAFE= False
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

async def getMediaSession():
    sessions = await wmc.GlobalSystemMediaTransportControlsSessionManager.request_async()
    session = sessions.get_current_session()
    return session

def mediaIs(state):
    session = asyncio.run(getMediaSession())
    if session == None:
        return False
    return int(wmc.GlobalSystemMediaTransportControlsSessionPlaybackStatus[state]) == session.get_playback_info().playback_status

from sklearn import preprocessing
flag = True if mediaIs("PLAYING") else False
trust = 0
last_gesture = -1
model_right = pickle.load(open("model_right.sav","rb"))
model_left = pickle.load(open("model_left.sav","rb"))
classList = {0:"Pause",1:"Play",2:"Sound Decrease",3:"Sound Increase",4:"Next Track",5:"Previous Track",-1:"Start"}
cap = cv2.VideoCapture(0)
hand = "None"
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    image = cv2.flip(image, 1)
    image = cv2.rectangle(image,(0,0),(image.shape[1],80),(0,0,0),-1)
    image = cv2.putText(image,classList[last_gesture],(20,50),
                        cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255))

    image = cv2.putText(image,"Hand: "+hand,(image.shape[1]//2+100,50),
                        cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255))
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
      for idx, hand_handedness in enumerate(results.multi_handedness):
        hand = hand_handedness.classification[0].label

      for hand_landmarks in results.multi_hand_landmarks:
        keypoints = []
        wrist_point =0
        counter = 1
        for data_point in hand_landmarks.landmark:
            if counter == 1:
                wrist_point = abs(np.mean([data_point.x,data_point.y]))
                counter= 0
                continue
            
            keypoints.append(
            abs(wrist_point-abs(np.mean([data_point.x,data_point.y])))
            )
        
        keypoints = preprocessing.normalize([keypoints])
        keypoints = keypoints.tolist()[0]
        if hand.lower() =="left":
            print("Using left hand")
            result = model_left.predict(np.reshape(keypoints,(1,-1)))[0]
        else: 
            print("Using right hand")
            result = model_right.predict(np.reshape(keypoints,(1,-1)))[0]

        if trust>20: 
            if last_gesture == 0 and (flag):     
                pyautogui.press("playpause")
                print("PAUSE")
                flag=False
            elif last_gesture == 1 and not (flag):     
                pyautogui.press("playpause")
                print("PLAY")
                flag=True
            elif last_gesture == 2:
                pyautogui.press("volumedown")
                trust= 15
            elif last_gesture == 3:     
                pyautogui.press("volumeup")
                trust= 15
            elif last_gesture == 4:
                pyautogui.press("nexttrack")
                trust= 0
            elif last_gesture == 5:     
                pyautogui.press("prevtrack")
                trust= 0

                
        if result == last_gesture:
            trust+=1
        else:
            last_gesture=result
            trust=0

                               
            
    cv2.imshow('MediaPipe Hands', image)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()