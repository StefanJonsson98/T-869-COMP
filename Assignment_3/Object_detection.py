import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

####### mobilenet -ssd model - divides the image into patches and asks the classifier to classify
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
file_name = 'labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')
    #classLabels.append(fpt.read())


model.setInputSize(300,300)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

video = cv2.VideoCapture(1)

fps = video.get(cv2.CAP_PROP_FPS)
num_frames = 1

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

countFrames = 0
countNonhumans = 0

while True:
    start = time.time()

    ret, frame = video.read()

    ClassIndex, confidence, bbox = model.detect(frame, confThreshold = 0.55)

    print(ClassIndex)

    if (len(ClassIndex)!=0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if (ClassInd < 80):
                cv2.rectangle(frame,boxes,(255,0,0),2)
                cv2.putText(frame,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=2)
    
    end = time.time()
    seconds = end - start
    fps = num_frames/seconds

    #perky = 100
    #corr = classLabels[ClassInd-1]
    #for i in (fps,100):
    #    if classLabels[ClassInd-1] == corr:
    #        prec = prec/num_frames
    #    else:
    #        prec = 0
    #    print(prec)


    #def findObjects(outputs, frame):
    #    outputNames = [classLabels[ClassInd-1] for i in model.getUnconnectedOutLayers()]
    #    outputs = model.forward(outputNames)
    #    countFrames = 0
    #    countNonhumans = 0
    #    if(findObjects(outputs, frame)):
    #        countNonhumans += 1
    #    countFrames += 1


    cv2.putText(frame, "FPS: " + str(round(fps)), (50,50), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,255,255))
    cv2.imshow('Object detection',frame)

    print(countNonhumans)
    print(countFrames)
    

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()

