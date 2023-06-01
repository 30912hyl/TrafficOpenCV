import cv2 as cv

#img = cv.imread('Photos/hermes-birkin.jpg') #read image into matrix

#cv.imshow('Bag', img)                  #show image(window name, matrix)

#cv.waitKey(0)                          #wait n until key pressed(0=inf)

#Reading Videos
capture = cv.VideoCapture(1)  #arg: int or path to video file
                                        #0 for webcam, 1 for 2nd connected cam, etc.
while True:
    isTrue, frame = capture.read()      #read each frame as frame, true if successful
    cv.imshow('Video', frame)           #read frame by frame
    
    if cv.waitKey(20) & 0xFF==ord('d'): #stop indefinite playing or if 'd' is pressed
        break

capture.release()
cv.destroyAllWindows()

