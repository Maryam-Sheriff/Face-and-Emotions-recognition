'''
1- IMPORT LIBS
2- Read Images
3- Convert to Grayscale
4- Resize
5- import the dataset
7 Create the model
8- detect faces
9- draw rectangle around the face
11- display the image
'''


import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread('Image1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.2)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

save_img = cv2.imwrite('Image.jpg', img)
#cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#
# #open webcam and detect faces
# cap = cv2.VideoCapture(0)
# while 1:
#     ret, img = cap.read()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     for (x,y,w,h) in faces:
#         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = img[y:y+h, x:x+w]
#         eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.2)
#         for (ex,ey,ew,eh) in eyes:
#             cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#     cv2.imshow('img',img)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
# cap.release()
# cv2.destroyAllWindows()

# We install the FER() library to perform facial recognition
# This installation will also take care of any of the above dependencies if they are missing
# pip
# install
# FER

from fer import FER
import matplotlib.pyplot as plt
import cv2
import numpy as np

test_image_one = cv2.imread('Image.jpg')
emo_detector = FER( mtcnn=True )
# Capture all the emotions on the image
captured_emotions = emo_detector.detect_emotions( test_image_one )
# Print all captured emotions with the image
print( captured_emotions )
# plt.imshow( test_image_one )

# Use the top Emotion() function to call for the dominant emotion in the image
dominant_emotion, emotion_score = emo_detector.top_emotion( test_image_one )
print( dominant_emotion, emotion_score )

# Python program to explain cv2.putText() method

# importing cv2
import cv2

# path
path = r'Image.jpg'
image = cv2.imread( path )
# Reading an image in default mode
# image = cv2.imread( img )

# Window name in which image is displayed
window_name = 'Image'

# font
font = cv2.FONT_HERSHEY_SIMPLEX

# org
org = (50, 50)

# fontScale
fontScale = 1

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 2

# Using cv2.putText() method
imgtext = cv2.putText( image, dominant_emotion, org, font,
                     fontScale, color, thickness, cv2.LINE_AA )

# Displaying the image
cv2.imshow( window_name, image )
cv2.waitKey( 0 )
