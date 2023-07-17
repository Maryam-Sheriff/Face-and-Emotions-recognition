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

#import
import numpy as np
import cv2

#training data
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#read image
img = cv2.imread('img_1.png')
#convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#detect faces
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#draw rectangle around the face
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    #detect eyes
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.2)
    for (ex,ey,ew,eh) in eyes:
        #draw rectangle around the eyes
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

#display the image
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()



#open webcam and detect faces
cap = cv2.VideoCapture(0)
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.2)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()

