import cv2
import sys

cascPath = "D:\DownloadsActual\FaceDetect-master\FaceDetect-master\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0+cv2.CAP_DSHOW)
print(video_capture)
#cv2.namedWindow("preview")
while True:
    # Capture frame-by-frame
    
    ret, frame = video_capture.read()
    print(video_capture.isOpened())
    print(ret)
    print(frame)
    #cv2.imshow("preview", frame)
    if ret == False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()


































#from imutils.video import VideoStream
#from imutils.video import FPS
#import argparse
#import imutils
#import time
#import cv2
#from datetime import datetime, time
#import numpy as np
#import time as time2


##adding arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-v", "--video", help="path to the video file")
#ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
#ap.add_argument("-t", "--tracker", type=str, default="csrt", help="OpenCV object tracker type")
#args = vars(ap.parse_args())

##print(args)
## extract the OpenCV version info
#(major, minor) = cv2.__version__.split(".")[:2]
## if we are using OpenCV 3.2 or an earlier version, we can use a special factory
## function to create the entity that tracks objects 
#if int(major) == 3 and int(minor) < 3:
#    tracker = cv2.Tracker_create(args["tracker"].upper())
#    #tracker = cv2.TrackerGOTURN_create()   
## otherwise, for OpenCV 3.3 or newer, 
## we need to explicity call the respective constructor that contains the tracker object:
#else:
#    # initialize a dictionary that maps strings to their corresponding
#    # OpenCV object tracker implementations
#    OPENCV_OBJECT_TRACKERS = {
#            "csrt": cv2.TrackerCSRT_create,
#            "kcf": cv2.TrackerKCF_create,
#            "boosting": cv2.TrackerBoosting_create,
#            "mil": cv2.TrackerMIL_create,
#            "tld": cv2.TrackerTLD_create,
#            "medianflow": cv2.TrackerMedianFlow_create,
#            "mosse": cv2.TrackerMOSSE_create
#    }
# #grab the appropriate object tracker using our dictionary of
# #OpenCV object tracker objects
#    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
#    #tracker = cv2.TrackerGOTURN_create()   
# #if the video argument is None, then the code will read from webcam (work in progress)
#if args.get("video", None) is None:
#    vs = cv2.VideoCapture(cv2.CAP_DSHOW)
#    #print(vs)
#    time2.sleep(2.0)
## otherwise, we are reading from a video file
#else:
#    vs = cv2.VideoCapture(args["video"])





## loop over the frames of the video, and store corresponding information from each frame
#firstFrame = None
#initBB2 = None
#fps = None
#differ = None
#now = ''
#framecounter = 0
#trackeron = 0

##cv2.namedWindow("preview")
##vc = cv2.VideoCapture(cv2.CAP_DSHOW)

##if vc.isOpened(): # try to get the first frame
##    rval2, frame2 = vc.read()
##else:
##    rval2 = False

##while rval:
##    cv2.imshow("preview", frame2)
##    rval2, frame2 = vc.read()
##    key = cv2.waitKey(20)
##    if key == 27: # exit on ESC
##        break
##vc.release();
##cv2.destroyWindow("preview")


#cv2.namedWindow("preview")
#if vs.isOpened(): # try to get the first frame
#    rval, frame = vs.read()
#else:
#    rval = False


#while True:
    
#    rval, frame = vs.read()
#    print(frame)
#    cv2.imshow("preview", frame)
#    frame = frame if args.get("video", None) is None else frame[1]
#    # if the frame can not be grabbed, then we have reached the end of the video
#    if frame is None:
#            break
#    key = cv2.waitKey(20)
#    if key == 27: # exit on ESC
#            break

#    # resize the frame to 500
#    #frame = imutils.resize(frame, width=500)

#    framecounter = framecounter+1
#    if framecounter > 1 :

#        (H, W) = frame.shape[:2]
#        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#        gray = cv2.GaussianBlur(gray, (21, 21), 0)

#        # if the first frame is None, initialize it
#        if firstFrame is None:
#            firstFrame = gray
#            continue

#        # compute the absolute difference between the current frame and first frame
#        frameDelta =  cv2.absdiff(firstFrame, gray)
       
#        print("This is frameDelta:")
#        print(frameDelta)
#        ret,thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

#        # dilate the thresholded image to fill in holes, then find contours on thresholded image
#        print( thresh)
#        cv2.dilate(thresh, thresh, None, iterations=2)
        
#        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

#        # loop over the contours identified
#        contourcount = 0
#        for c in cnts:
#            contourcount =  contourcount + 1

#           # if the contour is too small, ignore it
#            if cv2.contourArea(c) < args["min_area"]:
#                continue

#            # compute the bounding box for the contour, draw it on the frame,
#            (x, y, w, h) = cv2.boundingRect(c)
#            initBB2 =(x,y,w,h)


##TESTs THE VIDEO
########
##import cv2

##cv2.namedWindow("preview")
##vc = cv2.VideoCapture(0)

##if vc.isOpened(): # try to get the first frame
##    rval, frame = vc.read()
##else:
##    rval = False

##while rval:
##    cv2.imshow("preview", frame)
##    rval, frame = vc.read()
##    key = cv2.waitKey(20)
##    if key == 27: # exit on ESC
##        break
##vc.release();
##cv2.destroyWindow("preview")
