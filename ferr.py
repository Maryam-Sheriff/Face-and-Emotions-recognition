#import
from fer import FER
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt
#read with cv then plt

#reading image
test_image_one = cv2.imread('Image1.jpg')
#emotion recognition
emo_detector = FER( mtcnn=True )
# Capture all the emotions on the image
captured_emotions = emo_detector.detect_emotions( test_image_one )
# Print all captured emotions with the image
print( captured_emotions )
plt.imshow( test_image_one )

# Use the top Emotion() function to call for the dominant emotion in the image
dominant_emotion, emotion_score = emo_detector.top_emotion( test_image_one )
print( dominant_emotion, emotion_score )

# Python program to explain cv2.putText() method

# importing cv2
import cv2

# path
path = r'Image1.jpg'

# Reading an image in default mode
image = cv2.imread( path )

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
image = cv2.putText( image, dominant_emotion, org, font,
                     fontScale, color, thickness, cv2.LINE_AA )

# Displaying the image
cv2.imshow( window_name, image )
cv2.waitKey( 0 )
