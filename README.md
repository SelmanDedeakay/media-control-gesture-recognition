# Media Control by Hand Gesture Recognition Using Mediapipe

## BIM453 - Introduction to Machine Learning Term Project

This application aims to recognize hand gestures and execute the media control command mapped to that gesture, which are:

- Media Play/Pause
- Volume Increase/Decrease
- Changing to the Previous/Next Track
- Mute/Unmute
### Recognition

For recognition, we are using the landmark points provided by **MediaPipe Hands** module. 
MediaPipe provides us the 21 landmarks of the input image of the hand, which are going to be our inputs for ML models.
Our inputs are **the distance between input hand's "the mean of x and y points of the wrist point of hand and the mean of x and y points of the rest of all points (excluding the wrist point.)"**

### Running the Code

Please be sure that you have downloaded the requirements.txt file.
After that simply execute the following command:

``
pip install -r requirements.txt
``

When the installation of packages done, you can execute the main.py file.
