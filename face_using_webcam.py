#Import required modules
import cv2
import numpy as np
import dlib

#Set up some required objects
video_capture = cv2.VideoCapture(0) #Webcam object
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("/home/sangeetha/Desktop/CG/dlib/python_examples/shape_predictor_68_face_landmarks.dat") #Landmark identifier

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)

    detections = detector(clahe_image, 1) #Detect the faces in the image

    for k,d in enumerate(detections): #For each detected face
        shape = predictor(clahe_image, d) #Get coordinates
        #contours = []
        contours = np.ndarray(shape=(10,2))
        for i in range(0,10):
            contours[i][0] = 0
            contours[i][1] = 0
        #print contours
        for i in range(17,27): #Left eyebrow 17-21 Right eyebrow 22-26
            print "actual x"
            print shape.part(i).x
            #contours = np.append(contours,np.ndarray([shape.part(i).x, shape.part(i).y]))
            contours[i-17][0] = shape.part(i).x
            print contours[i-17][0]
            contours[i-17][1] = shape.part(i).y
        print contours
        contours = np.append(contours, contours[::-1])
        ctr = np.array(contours).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(frame, [ctr], -1, (0,255,0), 1)
        contours = np.empty(shape=(6,2))

            #cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (255,0,0), thickness=2) #For each point, draw a red circle with thickness2 on the original frame
        for i in range(36,48): #Left eye 36-41 Right Eye 42-47
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,255,0), thickness=1) #For each point, draw a red circle with thickness2 on the original frame

    cv2.imshow("image", frame) #Display the frame

    if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
        break
