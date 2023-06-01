import cv2 as cv

def rescaleFrame(frame, scale): 
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def changeRes(width,height):#300,600 is phone default
    # Live video
    capture.set(3,width)
    capture.set(4,height)
'''
capture = cv.VideoCapture(1) 
#changeRes(300,600)                                        
while True:
    isTrue, frame = capture.read()      
    #rescaledFrame = rescaleFrame(frame,0.75)
    
    cv.imshow('Video', frame) 
    #cv.imshow('Resized Video', rescaledFrame)                 

    if cv.waitKey(20) & 0xFF==ord('d'): 
        break

capture.release()
cv.destroyAllWindows()
'''
