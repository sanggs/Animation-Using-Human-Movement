#Import required modules
import cv2
import numpy as np
import dlib
import time

mvmt = np.ndarray(shape=(1,2))
for i in range(0,len(mvmt)):
    mvmt[i][0] = 0
    mvmt[i][1] = 0

le_mvmt = np.ndarray(shape=(1,2))
for i in range(0,len(le_mvmt)):
    le_mvmt[i][0] = 0
    le_mvmt[i][1] = 0

re_mvmt = np.ndarray(shape=(1,2))
for i in range(0,len(re_mvmt)):
    re_mvmt[i][0] = 0
    re_mvmt[i][1] = 0

j_mvmt = np.ndarray(shape=(1,2))
for i in range(0,len(j_mvmt)):
    j_mvmt[i][0] = 0
    j_mvmt[i][1] = 0

ol_mvmt = np.ndarray(shape=(1,2))
for i in range(0,len(ol_mvmt)):
    ol_mvmt[i][0] = 0
    ol_mvmt[i][1] = 0

il_mvmt = np.ndarray(shape=(1,2))
for i in range(0,len(il_mvmt)):
    il_mvmt[i][0] = 0
    il_mvmt[i][1] = 0



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
    global mvmt, le_mvmt, re_mvmt, j_mvmt, ol_mvmt, il_mvmt
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
            
            #Jawline
            j_contours = np.ndarray(shape=(17,2))
            for i in range(0,17):
                j_contours[i][0] = 0
                j_contours[i][1] = 0
            for i in range(0,17): #Left eyebrow 17-21 Right eyebrow 22-26
                j_contours[i][0] = shape.part(i).x
                j_contours[i][1] = shape.part(i).y

            j_mvmt = np.append(j_mvmt, j_contours)
            j_contours = np.append(j_contours, j_contours[::-1])
            ctr = np.array(j_contours).reshape((-1,1,2)).astype(np.int32)
            cv2.drawContours(frame, [ctr], -1, (0,255,0), 1)
            j_contours = np.empty(shape=(17,2))

            #Eyebrows
            eb_contours = np.ndarray(shape=(10,2))
            for i in range(0,10):
                eb_contours[i][0] = 0
                eb_contours[i][1] = 0
            for i in range(17,27): #Left eyebrow 17-21 Right eyebrow 22-26
                eb_contours[i-17][0] = shape.part(i).x
                eb_contours[i-17][1] = shape.part(i).y

            mvmt = np.append(mvmt, eb_contours)
            eb_contours = np.append(eb_contours, eb_contours[::-1])
            ctr = np.array(eb_contours).reshape((-1,1,2)).astype(np.int32)
            cv2.drawContours(frame, [ctr], -1, (0,255,0), 1)
            eb_contours = np.empty(shape=(10,2))

            #Left Eye            
            le_contours = np.ndarray(shape=(6,2))
            for i in range(0,6):
                le_contours[i][0] = 0
                le_contours[i][1] = 0
            for i in range(36,42): #Left eye 36-41
                le_contours[i-36][0] = shape.part(i).x
                le_contours[i-36][1] = shape.part(i).y

            le_mvmt = np.append(le_mvmt, le_contours)
            #le_contours = np.append(le_contours, le_contours[::-1])
            ctr = np.array(le_contours).reshape((-1,1,2)).astype(np.int32)
            cv2.drawContours(frame, [ctr], -1, (0,255,0), 1)
            le_contours = np.empty(shape=(6,2))

            #Right Eye            
            re_contours = np.ndarray(shape=(6,2))
            for i in range(0,6):
                re_contours[i][0] = 0
                re_contours[i][1] = 0
            for i in range(42,48): #Right eye 42-47
                re_contours[i-42][0] = shape.part(i).x
                re_contours[i-42][1] = shape.part(i).y

            re_mvmt = np.append(re_mvmt, re_contours)
            #re_contours = np.append(re_contours, re_contours[::-1])
            ctr = np.array(re_contours).reshape((-1,1,2)).astype(np.int32)
            cv2.drawContours(frame, [ctr], -1, (0,255,0), 1)
            re_contours = np.empty(shape=(6,2))

            #Outer Lip           
            ol_contours = np.ndarray(shape=(12,2))
            for i in range(0,12):
                ol_contours[i][0] = 0
                ol_contours[i][1] = 0
            for i in range(48,60): #Outer Lip 48-59
                ol_contours[i-48][0] = shape.part(i).x
                ol_contours[i-48][1] = shape.part(i).y

            ol_mvmt = np.append(ol_mvmt, ol_contours)
            #ol_contours = np.append(ol_contours, ol_contours[::-1])
            ctr = np.array(ol_contours).reshape((-1,1,2)).astype(np.int32)
            cv2.drawContours(frame, [ctr], -1, (0,255,0), 1)
            ol_contours = np.empty(shape=(12,2))

            #Inner Lip           
            il_contours = np.ndarray(shape=(8,2))
            for i in range(0,8):
                il_contours[i][0] = 0
                il_contours[i][1] = 0
            for i in range(60,68): #Inner Lip 60-67
                il_contours[i-60][0] = shape.part(i).x
                il_contours[i-60][1] = shape.part(i).y

            il_mvmt = np.append(il_mvmt, il_contours)
            #il_contours = np.append(il_contours, il_contours[::-1])
            ctr = np.array(il_contours).reshape((-1,1,2)).astype(np.int32)
            cv2.drawContours(frame, [ctr], -1, (0,255,0), 1)
            il_contours = np.empty(shape=(12,2))

            
            #for i in range(36,48): #Left eye 36-41 Right Eye 42-47
            #    cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,255,0), thickness=1) #For each point, draw a red circle with thickness2 on the original frame
        #'''
        cv2.imshow("image", frame) #Display the frame

        if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
            break

def replay():
    global mvmt, le_mvmt, re_mvmt, j_mvmt, ol_mvmt, il_mvmt
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
    '''
    img = cv2.imread('bk.png')
    leb_contours = np.ndarray(shape=(10,2))
    leb_contours = np.array([[ leb10,  leb11],[ leb20,  leb21],[ leb30,  leb31],[ leb40,  leb41],[ leb50,  leb51]], np.int32)
    leb_contours = np.append(leb_contours, leb_contours[::-1])

    l_ctr = np.array(leb_contours).reshape((-1,1,2)).astype(np.int32)
    cv2.drawContours(img, [l_ctr], -1, (0,255,0), 1)

    reb_contours = np.ndarray(shape=(10,2))
    reb_contours = np.array([[ reb10,  reb11],[ reb20,  reb21],[ reb30,  reb31],[ reb40,  reb41],[ reb50,  reb51]], np.int32)
    reb_contours = np.append(reb_contours, reb_contours[::-1])

    r_ctr = np.array(reb_contours).reshape((-1,1,2)).astype(np.int32)
    cv2.drawContours(img, [r_ctr], -1, (0,255,0), 1)

    cv2.imshow('img',img)
    time.sleep(5)
    #'''

    jaw = 2
    leb = 2
    reb = 2
    le = 2
    re = 2
    ol = 2
    il = 2

    for jaw in range(2,len(j_mvmt),34):
        
        img = cv2.imread('bk.png')

        #Jaw
        jaw_contours = np.ndarray(shape=(17,2))
        jaw_contours = np.array([[ j_mvmt[jaw+0],  j_mvmt[jaw+1]],[ j_mvmt[jaw+2],  j_mvmt[jaw+3]],[ j_mvmt[jaw+4],  j_mvmt[jaw+5]],[ j_mvmt[jaw+6],  j_mvmt[jaw+7]],[ j_mvmt[jaw+8],  j_mvmt[jaw+9]],[ j_mvmt[jaw+10],  j_mvmt[jaw+11]],[ j_mvmt[jaw+12],  j_mvmt[jaw+13]],[ j_mvmt[jaw+14],  j_mvmt[jaw+15]],[ j_mvmt[jaw+16],  j_mvmt[jaw+17]],[ j_mvmt[jaw+18],  j_mvmt[jaw+19]],[ j_mvmt[jaw+20],  j_mvmt[jaw+21]],[ j_mvmt[jaw+22],  j_mvmt[jaw+23]],[ j_mvmt[jaw+24],  j_mvmt[jaw+25]],[ j_mvmt[jaw+26],  j_mvmt[jaw+27]],[ j_mvmt[jaw+28],  j_mvmt[jaw+29]],[ j_mvmt[jaw+30],  j_mvmt[jaw+31]],[ j_mvmt[jaw+32],  j_mvmt[jaw+33]]], np.int32)
        jaw_contours = np.append(jaw_contours, jaw_contours[::-1])
        ctr = np.array(jaw_contours).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(img, [ctr], -1, (0,0,0), 1)

        #Left eyebrow
        leb_contours = np.ndarray(shape=(5,2))
        leb_contours = np.array([[ mvmt[leb+0],  mvmt[leb+1]],[ mvmt[leb+2],  mvmt[leb+3]],[ mvmt[leb+4],  mvmt[leb+5]],[ mvmt[leb+6],  mvmt[leb+7]],[ mvmt[leb+8],  mvmt[leb+9]]], np.int32)
        leb_contours = np.append(leb_contours, leb_contours[::-1])
        ctr = np.array(leb_contours).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(img, [ctr], -1, (0,0,0), 1)

        #Right eyebrow
        reb_contours = np.ndarray(shape=(5,2))
        reb_contours = np.array([[ mvmt[leb+10],  mvmt[leb+11]],[ mvmt[leb+12],  mvmt[leb+13]],[ mvmt[leb+14],  mvmt[leb+15]],[ mvmt[leb+16],  mvmt[leb+17]],[ mvmt[leb+18],  mvmt[leb+19]]], np.int32)
        reb_contours = np.append(reb_contours, reb_contours[::-1])
        ctr = np.array(reb_contours).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(img, [ctr], -1, (0,0,0), 1)

        #Left eye
        le_contours = np.ndarray(shape=(6,2))
        le_contours = np.array([[ le_mvmt[le+0],  le_mvmt[le+1]],[ le_mvmt[le+2],  le_mvmt[le+3]],[ le_mvmt[le+4],  le_mvmt[le+5]],[ le_mvmt[le+6],  le_mvmt[le+7]],[ le_mvmt[le+8],  le_mvmt[le+9]],[ le_mvmt[le+10],  le_mvmt[le+11]]], np.int32)
        #le_contours = np.append(le_contours, le_contours[::-1])
        ctr = np.array(le_contours).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(img, [ctr], -1, (0,0,0), 1)

        #Right eye
        re_contours = np.ndarray(shape=(6,2))
        re_contours = np.array([[ re_mvmt[re+0],  re_mvmt[re+1]],[ re_mvmt[re+2],  re_mvmt[re+3]],[ re_mvmt[re+4],  re_mvmt[re+5]],[ re_mvmt[re+6],  re_mvmt[re+7]],[ re_mvmt[re+8],  re_mvmt[re+9]],[ re_mvmt[re+10],  re_mvmt[re+11]]], np.int32)
        #re_contours = np.append(re_contours, re_contours[::-1])
        ctr = np.array(re_contours).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(img, [ctr], -1, (0,0,0), 1)

        #Outer lip
        ol_contours = np.ndarray(shape=(12,2))
        ol_contours = np.array([[ ol_mvmt[ol+0],  ol_mvmt[ol+1]],[ ol_mvmt[ol+2],  ol_mvmt[ol+3]],[ ol_mvmt[ol+4],  ol_mvmt[ol+5]],[ ol_mvmt[ol+6],  ol_mvmt[ol+7]],[ ol_mvmt[ol+8],  ol_mvmt[ol+9]],[ ol_mvmt[ol+10],  ol_mvmt[ol+11]],[ ol_mvmt[ol+12],  ol_mvmt[ol+13]],[ ol_mvmt[ol+14],  ol_mvmt[ol+15]],[ ol_mvmt[ol+16],  ol_mvmt[ol+17]],[ ol_mvmt[ol+18],  ol_mvmt[ol+19]],[ ol_mvmt[ol+20],  ol_mvmt[ol+21]],[ ol_mvmt[ol+22],  ol_mvmt[ol+23]]], np.int32)
        #ol_contours = np.append(ol_contours, ol_contours[::-1])
        ctr = np.array(ol_contours).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(img, [ctr], -1, (0,0,0), 1)

        #Inner lip
        il_contours = np.ndarray(shape=(8,2))
        il_contours = np.array([[ il_mvmt[il+0],  il_mvmt[il+1]],[ il_mvmt[il+2],  il_mvmt[il+3]],[ il_mvmt[il+4],  il_mvmt[il+5]],[ il_mvmt[il+6],  il_mvmt[il+7]],[ il_mvmt[il+8],  il_mvmt[il+9]],[ il_mvmt[il+10],  il_mvmt[il+11]],[ il_mvmt[il+12],  il_mvmt[il+13]],[ il_mvmt[il+14],  il_mvmt[il+15]]], np.int32)
        #il_contours = np.append(il_contours, il_contours[::-1])
        ctr = np.array(il_contours).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(img, [ctr], -1, (0,0,0), 1)

        #update variables
        leb = leb + 20
        reb = reb + 10
        le = le + 12
        re = re + 12
        ol = ol + 24
        il = il + 16

        cv2.imshow('img',img)
        time.sleep(1)

        if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
            break

    print "DONE"

    '''
    img = cv2.imread('bk.png')
    leb_contours = np.ndarray(shape=(10,2))
    leb_contours = np.array([[ 585,  118],[ 594,  107],[ 608,  108],[ 624,  113],[ 639,  121]], np.int32)
    leb_contours = np.append(leb_contours, leb_contours[::-1])

    l_ctr = np.array(leb_contours).reshape((-1,1,2)).astype(np.int32)
    cv2.drawContours(img, [l_ctr], -1, (0,255,0), 1)

    reb_contours = np.ndarray(shape=(10,2))
    reb_contours = np.array([[ 671,  119],[ 696,  109],[ 723,  106],[ 751,  112],[ 773,  125]], np.int32)
    reb_contours = np.append(reb_contours, reb_contours[::-1])

    r_ctr = np.array(reb_contours).reshape((-1,1,2)).astype(np.int32)
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



