import cv2
import numpy as np
import math
import time

#cnt = [0,0]
l_h = []
l_d = []
m_hullPoints = []
m_defects = []
def record():
    global m_hullPoints, m_defects,l_h, l_d
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        cv2.rectangle(img,(100,100),(600,600),(0,255,0),0)
        crop_img = img[100:600, 100:600]
        grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        value = (31, 31)
        blurred = cv2.GaussianBlur(grey, value, 0)
        _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        cv2.imshow('Thresholded', thresh1)

        (version, _, _) = cv2.__version__.split('.')

        if version is '3':
            image, contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        elif version is '2':
            contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        maxArea, index = 0, 0
        for i in xrange(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > maxArea:
                maxArea = area
                index = i
        realHandContour = contours[index]
        realHandLen = cv2.arcLength(realHandContour, True)

        handContour = cv2.approxPolyDP(realHandContour, 0.001 * realHandLen, True)

        img2 = cv2.imread('bk.png')
        cv2.drawContours(img2, [handContour], -1, (0,0,0), 1)
        
        x,y,w,h = cv2.boundingRect(handContour)
        cv2.rectangle(crop_img,(x,y),(x+w,y+h),(0,0,255),0)

        hullHandContour = cv2.convexHull(handContour,returnPoints = False)
        hullPoints = [handContour[i[0]] for i in hullHandContour]

        l_h += [len(hullPoints)]
        for k in range(0,len(hullPoints)):
            m_hullPoints.append(hullPoints[k])
        hullPoints = np.array(hullPoints, dtype = np.int32)

        defects = cv2.convexityDefects(handContour,hullHandContour)
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(handContour[s][0])
            #print "Start : "
            #print start
            end = tuple(handContour[e][0])
            #print "End : "
            #print end
            far = tuple(handContour[f][0])
            #print "Far : "
            #print far
            m_defects.append([handContour[f][0]])
        l_d += [i]

        #Mask
        drawing = np.zeros(crop_img.shape,np.uint8)
        cv2.drawContours(drawing,[handContour],0,(0,255,0),0)
        cv2.drawContours(drawing,[hullPoints],0,(0,0,255),0)
        cv2.drawContours(thresh1, contours, -1, (0,255,0), 3)

        handMoments = cv2.moments(handContour)
        handXCenterMoment = int(handMoments["m10"] / handMoments["m00"])
        handYCenterMoment = int(handMoments["m01"] / handMoments["m00"])
        handMoment = (handXCenterMoment, handYCenterMoment)
        cv2.circle(img2,(handMoment), 1, (0,0,0), 1)
        #handMomentPositions += [handMoment]

        scaleFactor = 0.3
        shrunk = np.array(handContour * scaleFactor, dtype=np.int32)
        tx, ty, w, h = cv2.boundingRect(shrunk)
        maxPoint = None
        maxRadius = 0
        for x in xrange(w):
            for y in xrange(h):
                rad = cv2.pointPolygonTest(shrunk, (tx + x, ty + y), True)
                if rad > maxRadius:
                    maxPoint = (tx + x, ty + y)
                    maxRadius = rad
        realCenter = np.array(np.array(maxPoint) / scaleFactor,dtype=np.int32)
        error = int((1 / scaleFactor) * 1.5)
        maxPoint = None
        maxRadius = 0
        for x in xrange(realCenter[0] - error, realCenter[0] + error):
            for y in xrange(realCenter[1] - error, realCenter[1] + error):
                rad = cv2.pointPolygonTest(handContour, (x, y), True)
                if rad > maxRadius:
                    maxPoint = (x, y)
                    maxRadius = rad
        palmCenter = maxPoint

        palmRadius = cv2.pointPolygonTest(handContour,tuple(palmCenter), True)

        handDistance = (1000) / float(palmRadius)
        cv2.circle(img2,(palmCenter), int(palmRadius), (0,0,0), 1)
        cv2.drawContours(img2,[hullPoints],0,(0,0,255),0)
        
        crop_img2 = img2[100:600, 100:600]
        final = np.hstack((crop_img, crop_img2))
        #cv2.imshow('img2',img2)
        #cv2.imshow('Gesture', img)
        all_img = np.hstack((drawing, crop_img))
        cv2.imshow('Contours', all_img)
        cv2.imshow('FINAL',final)
        if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
            break

def replay():
    global m_hullPoints, m_defects,l_h, l_d
    #m_hullPoints = np.array(m_hullPoints).astype(np.int32)
    h = 0
    d = 0
    i = 0
    hull = []
    defects = []
    image = cv2.imread('bk.png')
    while (i<len(l_h)):
        image = cv2.imread('bk.png')
        hull += m_hullPoints[h:h+l_h[i]]        
        hull = np.array(hull).reshape((-1,1,2)).astype(np.int32)
        #cv2.drawContours(image, [hull], -1, (0,0,0), 1)
        cv2.polylines(image,[hull],True,(0,0,255))
        
        defects += m_defects[d:d+l_d[i]] 
        defects = np.array(defects).reshape((-1,1,2)).astype(np.int32)
        #cv2.drawContours(image, [defects], -1, (0,0,0), 1)
        
        h = h + l_h[i]
        d = d + l_d[i]
        i = i + 1
        hull = []
        defects = []

        cv2.imshow('img',image)
        time.sleep(1)

        if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
            break
        

record()
replay()