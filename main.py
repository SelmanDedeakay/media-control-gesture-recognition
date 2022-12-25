import pyaudio
from sklearn import preprocessing
import winsdk.windows.media.control as wmc
import pickle
import asyncio
import numpy as np
import pyautogui
import cv2
import mediapipe as mp
mp_hands = mp.solutions.hands
pyautogui.FAILSAFE = False
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


async def check_sound_level():
    # Initialize a PyAudio object
    p = pyaudio.PyAudio()

    # Open a stream for reading audio data
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1, rate=44100, input=True)

    # Read audio data from the stream
    data = stream.read(int(44100 * 2))
    # Convert the audio data to a NumPy array
    data = np.frombuffer(data, dtype='float32')
    # Calculate the RMS value of the audio data
    volume = np.sqrt(np.mean(data**2))

    # Close the stream
    stream.close()
    # Terminate the PyAudio object
    p.terminate()
    return int(volume)


async def getMediaSession():
    sessions = await wmc.GlobalSystemMediaTransportControlsSessionManager.request_async()
    session = sessions.get_current_session()
    return session


def mediaIs(state):
    session = asyncio.run(getMediaSession())
    if session == None:
        return False
    return int(wmc.GlobalSystemMediaTransportControlsSessionPlaybackStatus[state]) == session.get_playback_info().playback_status


flag = True if mediaIs("PLAYING") else False
trust = 0
last_gesture = -1
model_right = pickle.load(open("model_right.sav", "rb"))
model_left = pickle.load(open("model_left.sav", "rb"))
classList = {0: "Pause", 1: "Play", 2: "Sound Decrease",
             3: "Sound Increase", 4: "Next Track", 5: "Previous Track", 6: "Sound On",
             7: "Sound Off", -1: "Start"}
cap = cv2.VideoCapture(0)
hand = "None"
volume = True if check_sound_level() == 0 else False
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        image = cv2.flip(image, 1)
        image = cv2.rectangle(
            image, (0, 0), (image.shape[1], 80), (0, 0, 0), -1)
        image = cv2.putText(image, classList[last_gesture], (20, 50),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))

        image = cv2.putText(image, "Hand: "+hand, (image.shape[1]//2+100, 50),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:

            for idx, hand_handedness in enumerate(results.multi_handedness):
                hand = hand_handedness.classification[0].label

            for hand_landmarks in results.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)
                keypoints = []
                wrist_point = 0
                counter = 1
                for data_point in hand_landmarks.landmark:
                    if counter == 1:
                        wrist_point = abs(
                            np.mean([data_point.x, data_point.y]))
                        counter = 0
                        continue

                    keypoints.append(
                        abs(wrist_point -
                            abs(np.mean([data_point.x, data_point.y])))
                    )

                keypoints = preprocessing.normalize([keypoints])
                keypoints = keypoints.tolist()[0]
                if len(results.multi_handedness) == 2:
                    hand = "Both"
                    result_left = model_left.predict(
                        np.reshape(keypoints, (1, -1)))[0]
                    result_right = model_right.predict(
                        np.reshape(keypoints, (1, -1)))[0]
                    if (result_left == 3 and result_right == 1): result = 6
                    elif (result_left == 2 and result_right == 0): result = 7
                    
                    if trust>5:
                        if last_gesture == 6 and volume:
                            pyautogui.press("volumemute")
                            trust = 0
                            volume = False
                        elif last_gesture == 7 and not volume:
                            trust = 0
                            pyautogui.press("volumemute")
                            volume = True
                    if result == last_gesture:
                        trust += 1
                    else:
                        last_gesture = result
                        trust = 0
                else:
                    if hand.lower() == "left":
                        result = model_left.predict(
                            np.reshape(keypoints, (1, -1)))[0]
                    elif hand.lower() == "right":
                        result = model_right.predict(
                            np.reshape(keypoints, (1, -1)))[0]

                    if trust > 20:
                        if last_gesture == 0 and (flag):
                            pyautogui.press("playpause")

                            flag = False
                        elif last_gesture == 1 and not (flag):
                            pyautogui.press("playpause")

                            flag = True
                        elif last_gesture == 2:
                            pyautogui.press("volumedown")
                            trust = 15
                        elif last_gesture == 3:
                            pyautogui.press("volumeup")
                            trust = 15
                        elif last_gesture == 4:
                            pyautogui.press("nexttrack")
                            trust = 10
                        elif last_gesture == 5:
                            pyautogui.press("prevtrack")
                            trust = 10

                    if result == last_gesture:
                        trust += 1
                    else:
                        last_gesture = result
                        trust = 0

        cv2.imshow('MediaPipe Hands', image)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
