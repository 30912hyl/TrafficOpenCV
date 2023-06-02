import os
import numpy as np
import cv2 as cv
from rescale import *

haar_cascade = cv.CascadeClassifier('TrafficOpenCV/haar_face.xml')

people = []
for i in os.listdir(r'C:\\Users\\aaron\\OneDrive\\Documents\\OpenCV\\Faces'):
    people.append(i)

#features = np.load('features.npy')
#labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

capture = cv.VideoCapture(0)                                   
while True:
    isTrue, frame = capture.read()      
    
    #cv.imshow('Video', frame) 
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)           
   # Detect the face in the image
    faces_rect = haar_cascade.detectMultiScale(gray,1.1,4)

    for (x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(faces_roi)
        print(f'Label = {people[label]} with a confidence of {confidence}')

        cv.putText(frame,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=2)
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)

    cv.imshow('Detected Face', frame)
    if cv.waitKey(20) & 0xFF==ord('d'): 
        break

capture.release()
cv.destroyAllWindows()
'''
img = cv.imread(r'C:\\Users\\aaron\\OneDrive\\Documents\\OpenCV\\train\\IMG_0323.jpg')
img = rescaleFrame(img, 0.1)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray,1.1,4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)
'''