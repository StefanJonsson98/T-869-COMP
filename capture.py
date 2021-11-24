import cv2
import pafy
import numpy as np
import argparse
import time

# Start camera
url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4"
video = cv2.VideoCapture(url)

# FPS
fps = video.get(cv2.CAP_PROP_FPS)
#print("Frames per second camera: {0}".format(fps))

# Number of frames to capture
num_frames = 1

print("Capturing {0} frames".format(num_frames))


# Grab a few frames
while True:

    # Start time
    start = time.time()

    ret, frame = video.read()
    #Find x and y coordinates of the area of the image with the largest intensity value
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     # grab the image dimensions
   #h = frame.shape[0]
   #w = frame.shape[1]
    

    (minVal, maxVal, minLoc, maxLoc1) = cv2.minMaxLoc(gray)
    # loop over the image, pixel by pixel
    #pos = (0, 0)
    #brightest = 0
    #(x, y) = gray.shape
    #for i in range(x):
    #    for j in range(y):
    #        if (gray[i, j] > brightest):
    #            brightest = gray[i, j]
    #            pos = (j, i)
    #        

    # define the list of boundaries
    boundaries = [
	([17, 15, 100], [62, 68, 255])]

    for (lower, upper) in boundaries:
    # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
    # find the colors within the specified boundaries and apply
    # the mask
        mask = cv2.inRange(frame, lower, upper)
    (minVal, maxVal, minLoc, maxLoc2) = cv2.minMaxLoc(mask)

	

    #identify brightest spot and the most red spot
    brightest = cv2.circle(frame,maxLoc1, 11, (255, 0, 0), 2)
    reddest = cv2.circle(frame,maxLoc2,11, (0, 0, 255), 2)

    if frame is None:
        print("No frame")
        break

    end = time.time()
    # Time elapsed
    seconds = end - start
    #print ("Time taken : {0} seconds".format(seconds))

    # Calculate frames per second
    fps  = num_frames / seconds
    #print("Estimated frames per second : {0}".format(fps))

    #display fps counter
    cv2.putText(frame, "FPS: " + str(round(fps)), (50,50), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,255,255))

    #display results of brightest spot area
    cv2.imshow('frame', frame)



    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
    
    end = time.time()

# Release camera
video.release()