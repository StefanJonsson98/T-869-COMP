import cv2
import numpy as np
import utlis

webCamFeed = True
cap = cv2.VideoCapture(0)
cap.set(10,160)
heightImg = 640
widthImg  = 480

utlis.initializeTrackbars()
count=0

while True:

    if webCamFeed:success, img = cap.read()
    #resize the image
    img = cv2.resize(img, (widthImg, heightImg))
    #for inconsistencies
    imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8)
    #grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    #gaussian blur
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) 
    #trackbar values
    thres=utlis.valTrackbars()
    #canny blur and apply
    imgThreshold = cv2.Canny(imgBlur,thres[0],thres[1])
    kernel = np.ones((5, 5))
    #dilation and erosion
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  

    #find all present contours
    imgContours = img.copy() 
    imgBigContour = img.copy() 
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #draw all the detected contours
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)


    # find the biggest contour
    biggest, maxArea = utlis.biggestContour(contours)
    if biggest.size != 0:
        biggest=utlis.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)
        imgBigContour = utlis.drawRectangle(imgBigContour,biggest,2)
        #prepare the points for warp
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        # remove two pixels per side
        imgWarpColored=imgWarpColored[2:imgWarpColored.shape[0] - 2, 2:imgWarpColored.shape[1] - 2]
        imgWarpColored = cv2.resize(imgWarpColored,(widthImg,heightImg))

        # apply adaptive threshold
        imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre,3)


        edgedetect = imgBigContour
        warpedhoe = imgWarpColored

    else:
        edgedetect = imgBigContour
        warpedhoe = imgBlank

    cv2.imshow("edgedetect",edgedetect)
    cv2.imshow("warp", warpedhoe)

    # boom
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.waitKey(300)
        count += 1
        break

cv2.destroyAllWindows
    

