import cv2 as cv
from rescale import *

#img = cv.imread('Photos/group.jpg')
capture = cv.VideoCapture(1)                                   
while True:
    isTrue, frame = capture.read()      
    
    #cv.imshow('Video', frame) 
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)           
    haar_cascade = cv.CascadeClassifier('TrafficOpenCV/haar_face.xml')

    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    print(f'Number of faces found = {len(faces_rect)}')

    for (x,y,w,h) in faces_rect:
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)

    cv.imshow('Detected Faces', frame)

    if cv.waitKey(20) & 0xFF==ord('d'): 
        break

capture.release()
cv.destroyAllWindows()
'''
img = rescaleFrame(img, 0.5)
cv.imshow('Person', img)

gray...imshow

cv.waitKey(0)
'''