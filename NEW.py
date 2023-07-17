'''
import libs
detect emotions first
then display emotion on image
then detect face and draw rectangle around it plus the emotion
'''

from fer import FER
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt


font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 30)
fontScale = 1
color = (255, 0, 0)
thickness = 2

test_image_one = cv2.imread('img_2.png')
emo_detector = FER( mtcnn=True )
captured_emotions = emo_detector.detect_emotions( test_image_one )
print( captured_emotions )
plt.imshow( test_image_one )
dominant_emotion, emotion_score = emo_detector.top_emotion( test_image_one )
print( dominant_emotion, emotion_score )

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(test_image_one, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
x: object
for (x,y,w,h) in faces:
    img = cv2.rectangle(test_image_one,(x,y),(x+w,y+h),(255,0,0),2)
    image= cv2.putText(img, dominant_emotion, org, font, fontScale, color, thickness, cv2.LINE_AA)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

# Displaying the image
cv2.imshow( 'window_name', image )
cv2.waitKey( 0 )



    

