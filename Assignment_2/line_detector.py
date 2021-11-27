import cv2
import time
import numpy as np
from numpy import linalg as LA

cap = cv2.VideoCapture(0)
pTime = 0
temp = np.ndarray
delta = 5

while True:
    success, img = cap.read()
    img = img[0:640, 0:480]
    canny = cv2.Canny(img, 0, 400)
    
    temp = np.argwhere(canny > 0)
    
    if len(temp) != 0:
        points = temp.shape[0]
        for k in range(30):
            rand1 = np.random.randint(0,len(temp))
            rand2 = np.random.randint(0,len(temp))
            p1 = temp[rand1]
            p2 = temp[rand2]
            inliers = 2
            number = inliers
            for i in range(temp.shape[0],20):
                d = LA.norm(np.cross(p2-p1, p1-temp[i]))/LA.norm(p2-p1)
                if (d) < delta:
                    inliers += 1
            if inliers >= number:
                number = inliers
                point1 = p1
                point2 = p2
            
    else:
        for i in range(temp.shape[0],20):
            points = 100
            point1, point2 = [0, 0], [0, 0]
            inliers = 2
        number = inliers

    percentage = number/points
    print(percentage)
    print(number)

    
    
    if number > 50:
        cv2.line(img, [point1[1], point1[0]], [point2[1], point2[0]], (0, 255, 0), thickness=10, lineType=8)
        cv2.circle(img, [point1[1], point1[0]], 10, (255, 0, 0), 2)
        cv2.circle(img, [point2[1], point2[0]], 10, (255, 0, 0), 2)
        
        
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (480, 640), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)