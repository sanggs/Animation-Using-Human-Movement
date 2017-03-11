#Import required modules
import cv2
import numpy as np
import dlib
import time

mvmt = np.ndarray(shape=(1,2))
for i in range(0,1):
    mvmt[i][0] = 0
    mvmt[i][1] = 0

leb10 = 0
leb11 = 0
leb20 = 0
leb21 = 0
leb30 = 0
leb31 = 0
leb40 = 0
leb41 = 0
leb50 = 0
leb51 = 0
reb10 = 0
reb11 = 0
reb20 = 0
reb21 = 0
reb30 = 0 
reb31 = 0
reb40 = 0
reb41 = 0
reb50 = 0
reb51 = 0 

def record():
    global mvmt
    global leb10, leb11,leb20 ,leb21 ,leb30 ,leb31 ,leb40 ,leb41 ,leb50 ,leb51,reb10,reb11,reb20,reb21,reb30,reb31,reb40,reb41,reb50 ,reb51
    #Set up some required objects
    video_capture = cv2.VideoCapture(0) #Webcam object
    detector = dlib.get_frontal_face_detector() #Face detector
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier

    ssc = 0;
    while True:
        ssc = ssc + 1;
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_image = clahe.apply(gray)

        detections = detector(clahe_image, 1) #Detect the faces in the image
        '''
        for k,d in enumerate(detections): #For each detected face
            shape = predictor(clahe_image, d) #Get coordinates        
            for i in range(0,67): #Left eye 36-41 Right Eye 42-47
                cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,255,0), thickness=1) #For each point, draw a red circle with thickness2 on the original frame
        '''
        smsc = 0;
        for k,d in enumerate(detections): #For each detected face
            smsc = smsc + 1;
            shape = predictor(clahe_image, d) #Get coordinates
            contours = np.ndarray(shape=(10,2))
            for i in range(0,10):
                contours[i][0] = 0
                contours[i][1] = 0
            for i in range(17,27): #Left eyebrow 17-21 Right eyebrow 22-26
                contours[i-17][0] = shape.part(i).x
                contours[i-17][1] = shape.part(i).y
            if (smsc == 1 and ssc == 1): 
                print contours
                leb10 = contours[0][0]
                leb11 = contours[0][1]
                leb20 = contours[1][0]
                leb21 = contours[1][1]
                leb30 = contours[2][0]
                leb31 = contours[2][1]
                leb40 = contours[3][0]
                leb41 = contours[3][1]
                leb50 = contours[4][0]
                leb51 = contours[4][1]
                
                reb10 = contours[5][0]
                reb11 = contours[5][1]
                reb20 = contours[6][0]
                reb21 = contours[6][1]
                reb30 = contours[7][0]
                reb31 = contours[7][1]
                reb40 = contours[8][0]
                reb41 = contours[8][1]
                reb50 = contours[9][0]
                reb51 = contours[9][1]

            mvmt = np.append(mvmt, contours)
            contours = np.append(contours, contours[::-1])
            ctr = np.array(contours).reshape((-1,1,2)).astype(np.int32)
            cv2.drawContours(frame, [ctr], -1, (0,255,0), 1)
            contours = np.empty(shape=(10,2))
            for i in range(36,48): #Left eye 36-41 Right Eye 42-47
                cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,255,0), thickness=1) #For each point, draw a red circle with thickness2 on the original frame
        #'''
        cv2.imshow("image", frame) #Display the frame

        if cv2.waitKey(1) & 0xFF == ord('e'): #Exit program when the user presses 'q'
            break

def replay():
    global mvmt
    global leb10, leb11,leb20 ,leb21 ,leb30 ,leb31 ,leb40 ,leb41 ,leb50 ,leb51,reb10,reb11,reb20,reb21,reb30,reb31,reb40,reb41,reb50 ,reb51
    '''leb10 = 585
    leb11 = 118
    leb20 = 594
    leb21 = 107
    leb30 = 608
    leb31 = 108
    leb40 = 624
    leb41 = 113
    leb50 = 639
    leb51 = 121
    
    reb10 = 671
    reb11 = 119
    reb20 = 696
    reb21 = 109
    reb30 = 723
    reb31 = 106
    reb40 = 751
    reb41 = 112
    reb50 = 773
    reb51 = 125'''

    img = cv2.imread('bk.png')
    l_contours = np.ndarray(shape=(10,2))
    l_contours = np.array([[ leb10,  leb11],[ leb20,  leb21],[ leb30,  leb31],[ leb40,  leb41],[ leb50,  leb51]], np.int32)
    l_contours = np.append(l_contours, l_contours[::-1])

    l_ctr = np.array(l_contours).reshape((-1,1,2)).astype(np.int32)
    cv2.drawContours(img, [l_ctr], -1, (0,255,0), 1)

    r_contours = np.ndarray(shape=(10,2))
    r_contours = np.array([[ reb10,  reb11],[ reb20,  reb21],[ reb30,  reb31],[ reb40,  reb41],[ reb50,  reb51]], np.int32)
    r_contours = np.append(r_contours, r_contours[::-1])

    r_ctr = np.array(r_contours).reshape((-1,1,2)).astype(np.int32)
    cv2.drawContours(img, [r_ctr], -1, (0,255,0), 1)

    cv2.imshow('img',img)
    time.sleep(5)
    #'''
    for i in range(2,len(mvmt),20):
        img = cv2.imread('bk.png')
        l_contours = np.ndarray(shape=(10,2))
        #l_contours = np.array([[ leb10+mvmt[i+0],  leb11+mvmt[i+1]],[ leb20+mvmt[i+2],  leb21+mvmt[i+3]],[ leb30+mvmt[i+4],  leb31+mvmt[i+5]],[ leb40+mvmt[i+6],  leb41+mvmt[i+7]],[ leb50+mvmt[i+8],  leb51+mvmt[i+9]]], np.int32)
        l_contours = np.array([[ mvmt[i+0],  mvmt[i+1]],[ mvmt[i+2],  mvmt[i+3]],[ mvmt[i+4],  mvmt[i+5]],[ mvmt[i+6],  mvmt[i+7]],[ mvmt[i+8],  mvmt[i+9]]], np.int32)
        l_contours = np.append(l_contours, l_contours[::-1])

        l_ctr = np.array(l_contours).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(img, [l_ctr], -1, (0,255,0), 1)

        r_contours = np.ndarray(shape=(10,2))
        #r_contours = np.array([[ reb10+mvmt[i+10],  reb11+mvmt[i+11]],[ reb20+mvmt[i+12],  reb21+mvmt[i+13]],[ reb30+mvmt[i+14],  reb31+mvmt[i+15]],[ reb40+mvmt[i+16],  reb41+mvmt[i+17]],[ reb50+mvmt[i+18],  reb51+mvmt[i+19]]], np.int32)
        r_contours = np.array([[ mvmt[i+10],  mvmt[i+11]],[ mvmt[i+12],  mvmt[i+13]],[ mvmt[i+14],  mvmt[i+15]],[ mvmt[i+16],  mvmt[i+17]],[ mvmt[i+18],  mvmt[i+19]]], np.int32)
        r_contours = np.append(r_contours, r_contours[::-1])

        r_ctr = np.array(r_contours).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(img, [r_ctr], -1, (0,255,0), 1)
        
        cv2.imshow('img',img)
        time.sleep(1)

        if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
            break

    if cv2.waitKey(1) & 0xFF == ord('e'): #Exit program when the user presses 'q'
        print "REPLAY DONE. RETURNING."
        return

    '''
    img = cv2.imread('bk.png')
    l_contours = np.ndarray(shape=(10,2))
    l_contours = np.array([[ 585,  118],[ 594,  107],[ 608,  108],[ 624,  113],[ 639,  121]], np.int32)
    l_contours = np.append(l_contours, l_contours[::-1])

    l_ctr = np.array(l_contours).reshape((-1,1,2)).astype(np.int32)
    cv2.drawContours(img, [l_ctr], -1, (0,255,0), 1)

    r_contours = np.ndarray(shape=(10,2))
    r_contours = np.array([[ 671,  119],[ 696,  109],[ 723,  106],[ 751,  112],[ 773,  125]], np.int32)
    r_contours = np.append(r_contours, r_contours[::-1])

    r_ctr = np.array(r_contours).reshape((-1,1,2)).astype(np.int32)
    cv2.drawContours(img, [r_ctr], -1, (0,255,0), 1)

    cv2.imshow('img',img)
    cv2.waitKey(0)
    '''
    cv2.destroyAllWindows()

record();
#for i in range(12,len(mvmt),1):
#    mvmt[i-10] = mvmt[i] - mvmt[i-10]
#print mvmt[11]
replay();
