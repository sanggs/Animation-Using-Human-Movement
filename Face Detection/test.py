#Import required modules
import cv2
import numpy as np
import dlib
import time
import Tkinter as tk
from PIL import Image, ImageTk
import multiprocessing
import threading
from threading import Thread

stop_flag=0
record_flag=0
replay_flag=0
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier

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



def record(frame):
	global stop_flag, record_flag
	global detector, predictor
	global mvmt, le_mvmt, re_mvmt, j_mvmt, ol_mvmt, il_mvmt
	global leb10, leb11,leb20 ,leb21 ,leb30 ,leb31 ,leb40 ,leb41 ,leb50 ,leb51,reb10,reb11,reb20,reb21,reb30,reb31,reb40,reb41,reb50 ,reb51

	ssc = 0;
	
	#while True:
	print "in record loop"
	#pasted code here
	ssc = ssc + 1;
	ret, frame = video_capture.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	clahe_image = clahe.apply(gray)

	detections = detector(clahe_image, 1) #Detect the faces in the image
	
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

		print "After jaw"

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

		print "After eyebrow"

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

		print "After left eye"
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
		print "After Right eye"
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

		print "After ol "

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
		print " After il"
		
		#for i in range(36,48): #Left eye 36-41 Right Eye 42-47
		#    cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,255,0), thickness=1) #For each point, draw a red circle with thickness2 on the original frame
	#'''
	#cv2.imshow("image", frame) #Display the frame
	return frame[0:1000, 320:960]
	# end of pasted code

	#time.sleep(3)
	if stop_flag == 1:
		# stop recording
		return

ic_jaw = []
ic_leb = []
ic_reb = []
ic_le = []
ic_re = []
ic_ol = []
ic_il = []

def check_faces():
	global ic_jaw, ic_leb, ic_reb, ic_le, ic_re, ic_ol, ic_il
	global mvmt, le_mvmt, re_mvmt, j_mvmt, ol_mvmt, il_mvmt
	global leb10, leb11,leb20 ,leb21 ,leb30 ,leb31 ,leb40 ,leb41 ,leb50 ,leb51,reb10,reb11,reb20,reb21,reb30,reb31,reb40,reb41,reb50 ,reb51

	jaw = 36
	leb = 22
	le = 14
	re = 14
	ol = 26
	il = 18

	#Initial Face
	i_jaw = np.ndarray(shape=(17,2))
	i_leb = np.ndarray(shape=(5,2))
	i_reb = np.ndarray(shape=(5,2))
	i_le = np.ndarray(shape=(5,2))
	i_re = np.ndarray(shape=(5,2))
	i_ol = np.ndarray(shape=(5,2))
	i_il = np.ndarray(shape=(5,2))


	i_jaw = np.array([
	[558, 376],
	[561, 410],
	[567, 443],
	[574, 475],
	[585, 505],
	[602, 530],
	[624, 549],
	[649, 562],
	[679, 568],
	[708, 565],
	[734, 551],
	[756, 535],
	[774, 511],
	[787, 483],
	[797, 453],
	[805, 424],
	[812, 391]])
	ic_jaw = np.array([
	[558, 376],
	[561, 410],
	[567, 443],
	[574, 475],
	[585, 505],
	[602, 530],
	[624, 549],
	[649, 562],
	[679, 568],
	[708, 565],
	[734, 551],
	[756, 535],
	[774, 511],
	[787, 483],
	[797, 453],
	[805, 424],
	[812, 391]])
	i_leb = np.array([[578, 335],[594, 314],[618, 303],[645, 305],[669, 316]],np.int32)
	ic_leb = np.array([[578, 335],[594, 314],[618, 303],[645, 305],[669, 316]],np.int32)
	
	i_reb = np.array([[706, 318],[731, 310],[759, 312],[782, 325],[795, 348]],np.int32)
	ic_reb = np.array([[706, 318],[731, 310],[759, 312],[782, 325],[795, 348]],np.int32)
	
	i_le = np.array([[602, 361],[617, 353],[635, 353],[652, 364],[634, 368],[616, 369]],np.int32)
	ic_le = np.array([[602, 361],[617, 353],[635, 353],[652, 364],[634, 368],[616, 369]],np.int32)
	
	i_re = np.array([[720, 369],[738, 359],[756, 361],[770, 371],[756, 377],[737, 375]],np.int32)
	ic_re = np.array([[720, 369],[738, 359],[756, 361],[770, 371],[756, 377],[737, 375]],np.int32)
	
	i_ol = np.array([[633, 485],[652, 472],[672, 464],[684, 470],[697, 467],[714, 479],[731, 493],[712, 503],[694, 507],[680, 507],[667, 504],[650, 498]],np.int32)
	ic_ol = np.array([[633, 485],[652, 472],[672, 464],[684, 470],[697, 467],[714, 479],[731, 493],[712, 503],[694, 507],[680, 507],[667, 504],[650, 498]],np.int32)
	
	i_il = np.array([[642, 485],[671, 482],[683, 484],[696, 484],[721, 491],[694, 483],[681, 484],[669, 481]],np.int32)
	ic_il = np.array([[642, 485],[671, 482],[683, 484],[696, 484],[721, 491],[694, 483],[681, 484],[669, 481]],np.int32)

	#Face in recording
	jaw_contours = np.ndarray(shape=(17,2))
	jaw_contours = np.array([[ j_mvmt[jaw+0],  j_mvmt[jaw+1]],[ j_mvmt[jaw+2],  j_mvmt[jaw+3]],[ j_mvmt[jaw+4],  j_mvmt[jaw+5]],[ j_mvmt[jaw+6],  j_mvmt[jaw+7]],[ j_mvmt[jaw+8],  j_mvmt[jaw+9]],[ j_mvmt[jaw+10],  j_mvmt[jaw+11]],[ j_mvmt[jaw+12],  j_mvmt[jaw+13]],[ j_mvmt[jaw+14],  j_mvmt[jaw+15]],[ j_mvmt[jaw+16],  j_mvmt[jaw+17]],[ j_mvmt[jaw+18],  j_mvmt[jaw+19]],[ j_mvmt[jaw+20],  j_mvmt[jaw+21]],[ j_mvmt[jaw+22],  j_mvmt[jaw+23]],[ j_mvmt[jaw+24],  j_mvmt[jaw+25]],[ j_mvmt[jaw+26],  j_mvmt[jaw+27]],[ j_mvmt[jaw+28],  j_mvmt[jaw+29]],[ j_mvmt[jaw+30],  j_mvmt[jaw+31]],[ j_mvmt[jaw+32],  j_mvmt[jaw+33]]], np.int32)

	#Left eyebrow
	leb_contours = np.ndarray(shape=(5,2))
	leb_contours = np.array([[ mvmt[leb+0],  mvmt[leb+1]],[ mvmt[leb+2],  mvmt[leb+3]],[ mvmt[leb+4],  mvmt[leb+5]],[ mvmt[leb+6],  mvmt[leb+7]],[ mvmt[leb+8],  mvmt[leb+9]]], np.int32)
	
	#Right eyebrow
	reb_contours = np.ndarray(shape=(5,2))
	reb_contours = np.array([[ mvmt[leb+10],  mvmt[leb+11]],[ mvmt[leb+12],  mvmt[leb+13]],[ mvmt[leb+14],  mvmt[leb+15]],[ mvmt[leb+16],  mvmt[leb+17]],[ mvmt[leb+18],  mvmt[leb+19]]], np.int32)
	
	#Left eye
	le_contours = np.ndarray(shape=(6,2))
	le_contours = np.array([[ le_mvmt[le+0],  le_mvmt[le+1]],[ le_mvmt[le+2],  le_mvmt[le+3]],[ le_mvmt[le+4],  le_mvmt[le+5]],[ le_mvmt[le+6],  le_mvmt[le+7]],[ le_mvmt[le+8],  le_mvmt[le+9]],[ le_mvmt[le+10],  le_mvmt[le+11]]], np.int32)
	
	#Right eye
	re_contours = np.ndarray(shape=(6,2))
	re_contours = np.array([[ re_mvmt[re+0],  re_mvmt[re+1]],[ re_mvmt[re+2],  re_mvmt[re+3]],[ re_mvmt[re+4],  re_mvmt[re+5]],[ re_mvmt[re+6],  re_mvmt[re+7]],[ re_mvmt[re+8],  re_mvmt[re+9]],[ re_mvmt[re+10],  re_mvmt[re+11]]], np.int32)
	
	#Outer lip
	ol_contours = np.ndarray(shape=(12,2))
	ol_contours = np.array([[ ol_mvmt[ol+0],  ol_mvmt[ol+1]],[ ol_mvmt[ol+2],  ol_mvmt[ol+3]],[ ol_mvmt[ol+4],  ol_mvmt[ol+5]],[ ol_mvmt[ol+6],  ol_mvmt[ol+7]],[ ol_mvmt[ol+8],  ol_mvmt[ol+9]],[ ol_mvmt[ol+10],  ol_mvmt[ol+11]],[ ol_mvmt[ol+12],  ol_mvmt[ol+13]],[ ol_mvmt[ol+14],  ol_mvmt[ol+15]],[ ol_mvmt[ol+16],  ol_mvmt[ol+17]],[ ol_mvmt[ol+18],  ol_mvmt[ol+19]],[ ol_mvmt[ol+20],  ol_mvmt[ol+21]],[ ol_mvmt[ol+22],  ol_mvmt[ol+23]]], np.int32)
	
	#Inner lip
	il_contours = np.ndarray(shape=(8,2))
	il_contours = np.array([[ il_mvmt[il+0],  il_mvmt[il+1]],[ il_mvmt[il+2],  il_mvmt[il+3]],[ il_mvmt[il+4],  il_mvmt[il+5]],[ il_mvmt[il+6],  il_mvmt[il+7]],[ il_mvmt[il+8],  il_mvmt[il+9]],[ il_mvmt[il+10],  il_mvmt[il+11]],[ il_mvmt[il+12],  il_mvmt[il+13]],[ il_mvmt[il+14],  il_mvmt[il+15]]], np.int32)
	
	'''
	#Do the initial faces match?

	#JAW
	
	for i in range(0,17):
		i_jaw[i][0] = abs(i_jaw[i][0] - jaw_contours[i][0])
		i_jaw[i][1] = abs(i_jaw[i][1] - jaw_contours[i][1])

	for i in range(0,16):
		i_jaw[i][0] = i_jaw[i+1][0] - i_jaw[i][0]
		i_jaw[i][1] = i_jaw[i+1][1] - i_jaw[i][1]
		if (i_jaw[i][0] > 4 or i_jaw[i][1] > 6):
			print "INITIAL JAWS DO NOT MATCH"
			print "diff - "+str(i_jaw[i][0])+" "+str(i_jaw[i][1])
			exit(0)

	#LEFT EYEBROW
	for i in range(0,5):
		i_leb[i][0] = abs(i_leb[i][0] - leb_contours[i][0])
		i_leb[i][1] = abs(i_leb[i][1] - leb_contours[i][1])

	for i in range(0,4):
		i_leb[i][0] = i_leb[i+1][0] - i_leb[i][0]
		i_leb[i][1] = i_leb[i+1][1] - i_leb[i][1]
		if (i_leb[i][0] > 2 or i_leb[i][1] > 2):
			print "INITIAL LEB DO NOT MATCH"
			print "diff - "+str(i_leb[i][0])+" "+str(i_leb[i][1])
			exit(0)

	#RIGHT EYEBROW
	for i in range(0,5):
		i_reb[i][0] = abs(i_reb[i][0] - reb_contours[i][0])
		i_reb[i][1] = abs(i_reb[i][1] - reb_contours[i][1])

	for i in range(0,4):
		i_reb[i][0] = i_reb[i+1][0] - i_reb[i][0]
		i_reb[i][1] = i_reb[i+1][1] - i_reb[i][1]
		if (i_reb[i][0] > 2 or i_reb[i][1] > 2):
			print "INITIAL REB DO NOT MATCH"
			print "diff - "+str(i_reb[i][0])+" "+str(i_reb[i][1])
			exit(0)

	#LEFT EYE
	for i in range(0,6):
		i_le[i][0] = abs(i_le[i][0] - le_contours[i][0])
		i_le[i][1] = abs(i_le[i][1] - le_contours[i][1])

	for i in range(0,5):
		i_le[i][0] = i_le[i+1][0] - i_le[i][0]
		i_le[i][1] = i_le[i+1][1] - i_le[i][1]
		if (i_le[i][0] > 4 or i_le[i][1] > 4):
			print "INITIAL LE DO NOT MATCH"
			print "diff - "+str(i_le[i][0])+" "+str(i_le[i][1])
			exit(0)

	#RIGHT EYE
	for i in range(0,6):
		i_re[i][0] = abs(i_re[i][0] - re_contours[i][0])
		i_re[i][1] = abs(i_re[i][1] - re_contours[i][1])

	for i in range(0,5):
		i_re[i][0] = i_re[i+1][0] - i_re[i][0]
		i_re[i][1] = i_re[i+1][1] - i_re[i][1]
		if (i_re[i][0] > 4 or i_re[i][1] > 4):
			print "INITIAL RE DO NOT MATCH"
			print "diff - "+str(i_re[i][0])+" "+str(i_re[i][1])
			exit(0)

	#OUTER LIP
	for i in range(0,12):
		i_ol[i][0] = abs(i_ol[i][0] - ol_contours[i][0])
		i_ol[i][1] = abs(i_ol[i][1] - ol_contours[i][1])

	for i in range(0,11):
		i_ol[i][0] = i_ol[i+1][0] - i_ol[i][0]
		i_ol[i][1] = i_ol[i+1][1] - i_ol[i][1]
		if (i_ol[i][0] > 4 or i_ol[i][1] > 4):
			print "INITIAL OL DO NOT MATCH"
			print "diff - "+str(i_ol[i][0])+" "+str(i_ol[i][1])
			exit(0)

	#INNER LIP
	for i in range(0,8):
		i_il[i][0] = abs(i_il[i][0] - il_contours[i][0])
		i_il[i][1] = abs(i_il[i][1] - il_contours[i][1])

	for i in range(0,7):
		i_il[i][0] = i_il[i+1][0] - i_il[i][0]
		i_il[i][1] = i_il[i+1][1] - i_il[i][1]
		if (i_il[i][0] > 4 or i_il[i][1] > 4):
			print "INITIAL IL DO NOT MATCH"
			print "diff - "+str(i_il[i][0])+" "+str(i_il[i][1])
			exit(0)
	'''	

def translation():
	global mvmt, le_mvmt, re_mvmt, j_mvmt, ol_mvmt, il_mvmt
	for i in range(22,len(mvmt),20):
		mvmt[i] = mvmt[i] - mvmt[2]
		mvmt[i+1] = mvmt[i+1] - mvmt[3]
		mvmt[i+2] = mvmt[i+2] - mvmt[4]
		mvmt[i+3] = mvmt[i+3] - mvmt[5]
		mvmt[i+4] = mvmt[i+4] - mvmt[6]
		mvmt[i+5] = mvmt[i+5] - mvmt[7]
		mvmt[i+6] = mvmt[i+6] - mvmt[8]
		mvmt[i+7] = mvmt[i+7] - mvmt[9]
		mvmt[i+8] = mvmt[i+8] - mvmt[10]
		mvmt[i+9] = mvmt[i+9] - mvmt[11]
		mvmt[i+10] = mvmt[i+10] - mvmt[12]
		mvmt[i+11] = mvmt[i+11] - mvmt[13]
		mvmt[i+12] = mvmt[i+12] - mvmt[14]
		mvmt[i+13] = mvmt[i+13] - mvmt[15]
		mvmt[i+14] = mvmt[i+14] - mvmt[16]
		mvmt[i+15] = mvmt[i+15] - mvmt[17]
		mvmt[i+16] = mvmt[i+16] - mvmt[18]
		mvmt[i+17] = mvmt[i+17] - mvmt[19]
		mvmt[i+18] = mvmt[i+18] - mvmt[20]
		mvmt[i+19] = mvmt[i+19] - mvmt[21]

	for i in range(14,len(le_mvmt),12):
		le_mvmt[i] = le_mvmt[i] - le_mvmt[2]
		le_mvmt[i+1] = le_mvmt[i+1] - le_mvmt[3]
		le_mvmt[i+2] = le_mvmt[i+2] - le_mvmt[4]
		le_mvmt[i+3] = le_mvmt[i+3] - le_mvmt[5]
		le_mvmt[i+4] = le_mvmt[i+4] - le_mvmt[6]
		le_mvmt[i+5] = le_mvmt[i+5] - le_mvmt[7]
		le_mvmt[i+6] = le_mvmt[i+6] - le_mvmt[8]
		le_mvmt[i+7] = le_mvmt[i+7] - le_mvmt[9]
		le_mvmt[i+8] = le_mvmt[i+8] - le_mvmt[10]
		le_mvmt[i+9] = le_mvmt[i+9] - le_mvmt[11]
		le_mvmt[i+10] = le_mvmt[i+10] - le_mvmt[12]
		le_mvmt[i+11] = le_mvmt[i+11] - le_mvmt[13]

	for i in range(14,len(re_mvmt),12):
		re_mvmt[i] = re_mvmt[i] - re_mvmt[2]
		re_mvmt[i+1] = re_mvmt[i+1] - re_mvmt[3]
		re_mvmt[i+2] = re_mvmt[i+2] - re_mvmt[4]
		re_mvmt[i+3] = re_mvmt[i+3] - re_mvmt[5]
		re_mvmt[i+4] = re_mvmt[i+4] - re_mvmt[6]
		re_mvmt[i+5] = re_mvmt[i+5] - re_mvmt[7]
		re_mvmt[i+6] = re_mvmt[i+6] - re_mvmt[8]
		re_mvmt[i+7] = re_mvmt[i+7] - re_mvmt[9]
		re_mvmt[i+8] = re_mvmt[i+8] - re_mvmt[10]
		re_mvmt[i+9] = re_mvmt[i+9] - re_mvmt[11]
		re_mvmt[i+10] = re_mvmt[i+10] - re_mvmt[12]
		re_mvmt[i+11] = re_mvmt[i+11] - re_mvmt[13]

	for i in range(26,len(ol_mvmt),24):
		ol_mvmt[i] = ol_mvmt[i] - ol_mvmt[2]
		ol_mvmt[i+1] = ol_mvmt[i+1] - ol_mvmt[3]
		ol_mvmt[i+2] = ol_mvmt[i+2] - ol_mvmt[4]
		ol_mvmt[i+3] = ol_mvmt[i+3] - ol_mvmt[5]
		ol_mvmt[i+4] = ol_mvmt[i+4] - ol_mvmt[6]
		ol_mvmt[i+5] = ol_mvmt[i+5] - ol_mvmt[7]
		ol_mvmt[i+6] = ol_mvmt[i+6] - ol_mvmt[8]
		ol_mvmt[i+7] = ol_mvmt[i+7] - ol_mvmt[9]
		ol_mvmt[i+8] = ol_mvmt[i+8] - ol_mvmt[10]
		ol_mvmt[i+9] = ol_mvmt[i+9] - ol_mvmt[11]
		ol_mvmt[i+10] = ol_mvmt[i+10] - ol_mvmt[12]
		ol_mvmt[i+11] = ol_mvmt[i+11] - ol_mvmt[13]
		ol_mvmt[i+12] = ol_mvmt[i+12] - ol_mvmt[14]
		ol_mvmt[i+13] = ol_mvmt[i+13] - ol_mvmt[15]
		ol_mvmt[i+14] = ol_mvmt[i+14] - ol_mvmt[16]
		ol_mvmt[i+15] = ol_mvmt[i+15] - ol_mvmt[17]
		ol_mvmt[i+16] = ol_mvmt[i+16] - ol_mvmt[18]
		ol_mvmt[i+17] = ol_mvmt[i+17] - ol_mvmt[19]
		ol_mvmt[i+18] = ol_mvmt[i+18] - ol_mvmt[20]
		ol_mvmt[i+19] = ol_mvmt[i+19] - ol_mvmt[21]
		ol_mvmt[i+20] = ol_mvmt[i+20] - ol_mvmt[22]
		ol_mvmt[i+21] = ol_mvmt[i+21] - ol_mvmt[23]
		ol_mvmt[i+22] = ol_mvmt[i+22] - ol_mvmt[24]
		ol_mvmt[i+23] = ol_mvmt[i+23] - ol_mvmt[25]

	for i in range(18,len(il_mvmt),16):
		il_mvmt[i] = il_mvmt[i] - il_mvmt[2]
		il_mvmt[i+1] = il_mvmt[i+1] - il_mvmt[3]
		il_mvmt[i+2] = il_mvmt[i+2] - il_mvmt[4]
		il_mvmt[i+3] = il_mvmt[i+3] - il_mvmt[5]
		il_mvmt[i+4] = il_mvmt[i+4] - il_mvmt[6]
		il_mvmt[i+5] = il_mvmt[i+5] - il_mvmt[7]
		il_mvmt[i+6] = il_mvmt[i+6] - il_mvmt[8]
		il_mvmt[i+7] = il_mvmt[i+7] - il_mvmt[9]
		il_mvmt[i+8] = il_mvmt[i+8] - il_mvmt[10]
		il_mvmt[i+9] = il_mvmt[i+9] - il_mvmt[11]
		il_mvmt[i+10] = il_mvmt[i+10] - il_mvmt[12]
		il_mvmt[i+11] = il_mvmt[i+11] - il_mvmt[13]
		il_mvmt[i+12] = il_mvmt[i+12] - il_mvmt[14]
		il_mvmt[i+13] = il_mvmt[i+13] - il_mvmt[15]
		il_mvmt[i+14] = il_mvmt[i+14] - il_mvmt[16]
		il_mvmt[i+15] = il_mvmt[i+15] - il_mvmt[17]

	for i in range(36,len(j_mvmt),34):
		j_mvmt[i] = j_mvmt[i] - j_mvmt[2]
		j_mvmt[i+1] = j_mvmt[i+1] - j_mvmt[3]
		j_mvmt[i+2] = j_mvmt[i+2] - j_mvmt[4]
		j_mvmt[i+3] = j_mvmt[i+3] - j_mvmt[5]
		j_mvmt[i+4] = j_mvmt[i+4] - j_mvmt[6]
		j_mvmt[i+5] = j_mvmt[i+5] - j_mvmt[7]
		j_mvmt[i+6] = j_mvmt[i+6] - j_mvmt[8]
		j_mvmt[i+7] = j_mvmt[i+7] - j_mvmt[9]
		j_mvmt[i+8] = j_mvmt[i+8] - j_mvmt[10]
		j_mvmt[i+9] = j_mvmt[i+9] - j_mvmt[11]
		j_mvmt[i+10] = j_mvmt[i+10] - j_mvmt[12]
		j_mvmt[i+11] = j_mvmt[i+11] - j_mvmt[13]
		j_mvmt[i+12] = j_mvmt[i+12] - j_mvmt[14]
		j_mvmt[i+13] = j_mvmt[i+13] - j_mvmt[15]
		j_mvmt[i+14] = j_mvmt[i+14] - j_mvmt[16]
		j_mvmt[i+15] = j_mvmt[i+15] - j_mvmt[17]
		j_mvmt[i+16] = j_mvmt[i+16] - j_mvmt[18]
		j_mvmt[i+17] = j_mvmt[i+17] - j_mvmt[19]
		j_mvmt[i+18] = j_mvmt[i+18] - j_mvmt[20]
		j_mvmt[i+19] = j_mvmt[i+19] - j_mvmt[21]
		j_mvmt[i+20] = j_mvmt[i+20] - j_mvmt[22]
		j_mvmt[i+21] = j_mvmt[i+21] - j_mvmt[23]
		j_mvmt[i+22] = j_mvmt[i+22] - j_mvmt[24]
		j_mvmt[i+23] = j_mvmt[i+23] - j_mvmt[25]
		j_mvmt[i+24] = j_mvmt[i+24] - j_mvmt[26]
		j_mvmt[i+25] = j_mvmt[i+25] - j_mvmt[27]
		j_mvmt[i+26] = j_mvmt[i+26] - j_mvmt[28]
		j_mvmt[i+27] = j_mvmt[i+27] - j_mvmt[29]
		j_mvmt[i+28] = j_mvmt[i+28] - j_mvmt[30]
		j_mvmt[i+29] = j_mvmt[i+29] - j_mvmt[31]
		j_mvmt[i+30] = j_mvmt[i+30] - j_mvmt[32]
		j_mvmt[i+31] = j_mvmt[i+31] - j_mvmt[33]
		j_mvmt[i+32] = j_mvmt[i+32] - j_mvmt[34]
		j_mvmt[i+33] = j_mvmt[i+33] - j_mvmt[35]


def replay(jaw,leb,le,re,ol,il):
	global stop_flag, replay_flag, record_flag
	if stop_flag==0:
		stop() 
	print "Replaying"
	if jaw == 36:
		check_faces()
	# put replay code here
	global ic_jaw, ic_leb, ic_reb, ic_le, ic_re, ic_ol, ic_il
	global mvmt, le_mvmt, re_mvmt, j_mvmt, ol_mvmt, il_mvmt
	global leb10, leb11,leb20 ,leb21 ,leb30 ,leb31 ,leb40 ,leb41 ,leb50 ,leb51,reb10,reb11,reb20,reb21,reb30,reb31,reb40,reb41,reb50 ,reb51
	
	if jaw == 36:
		translation()

	img = cv2.imread('bk_white.png')

	print "Jaw"

	#Jaw
	jaw_contours = np.ndarray(shape=(17,2))
	jaw_contours = np.array([[ ic_jaw[0][0]+j_mvmt[jaw+0],  ic_jaw[0][1]+j_mvmt[jaw+1]],[ ic_jaw[1][0]+j_mvmt[jaw+2],  ic_jaw[1][1]+j_mvmt[jaw+3]],[ ic_jaw[2][0]+j_mvmt[jaw+4],  ic_jaw[2][1]+j_mvmt[jaw+5]],[ ic_jaw[3][0]+j_mvmt[jaw+6],  ic_jaw[3][1]+j_mvmt[jaw+7]],[ ic_jaw[4][0]+j_mvmt[jaw+8],  ic_jaw[4][1]+j_mvmt[jaw+9]],[ ic_jaw[5][0]+j_mvmt[jaw+10],  ic_jaw[5][1]+j_mvmt[jaw+11]],[ ic_jaw[6][0]+j_mvmt[jaw+12],  ic_jaw[6][1]+j_mvmt[jaw+13]],[ ic_jaw[7][0]+j_mvmt[jaw+14],  ic_jaw[7][1]+j_mvmt[jaw+15]],[ ic_jaw[8][0]+j_mvmt[jaw+16],  ic_jaw[8][1]+j_mvmt[jaw+17]],[ ic_jaw[9][0]+j_mvmt[jaw+18],  ic_jaw[9][1]+j_mvmt[jaw+19]],[ ic_jaw[10][0]+j_mvmt[jaw+20],  ic_jaw[10][1]+j_mvmt[jaw+21]],[ ic_jaw[11][0]+j_mvmt[jaw+22],  ic_jaw[11][1]+j_mvmt[jaw+23]],[ ic_jaw[12][0]+j_mvmt[jaw+24],  ic_jaw[12][1]+j_mvmt[jaw+25]],[ ic_jaw[13][0]+j_mvmt[jaw+26],  ic_jaw[13][1]+j_mvmt[jaw+27]],[ ic_jaw[14][0]+j_mvmt[jaw+28],  ic_jaw[14][1]+j_mvmt[jaw+29]],[ ic_jaw[15][0]+j_mvmt[jaw+30],  ic_jaw[15][1]+j_mvmt[jaw+31]],[ ic_jaw[16][0]+j_mvmt[jaw+32],  ic_jaw[16][1]+j_mvmt[jaw+33]]], np.int32)
	if (jaw == 2):
		print "jaw"
		print jaw_contours
	jaw_contours = np.append(jaw_contours, jaw_contours[::-1])
	ctr = np.array(jaw_contours).reshape((-1,1,2)).astype(np.int32)
	cv2.drawContours(img, [ctr], -1, (0,0,0), 1)
	print "Leb"
	#Left eyebrow
	leb_contours = np.ndarray(shape=(5,2))
	leb_contours = np.array([[ ic_leb[0][0] + mvmt[leb+0],  ic_leb[0][1] + mvmt[leb+1]],[ ic_leb[1][0] + mvmt[leb+2],  ic_leb[1][1] + mvmt[leb+3]],[ ic_leb[2][0] + mvmt[leb+4],  ic_leb[2][1] + mvmt[leb+5]],[ ic_leb[3][0] + mvmt[leb+6],  ic_leb[3][1] + mvmt[leb+7]],[ ic_leb[4][0] + mvmt[leb+8],  ic_leb[4][1] + mvmt[leb+9]]], np.int32)
	if (jaw == 2):
		print "Left eyebrow"
		print leb_contours
	leb_contours = np.append(leb_contours, leb_contours[::-1])
	ctr = np.array(leb_contours).reshape((-1,1,2)).astype(np.int32)
	cv2.drawContours(img, [ctr], -1, (0,0,0), 1)
	print "Reb"
	#Right eyebrow
	reb_contours = np.ndarray(shape=(5,2))
	reb_contours = np.array([[ ic_reb[0][0] + mvmt[leb+10],  ic_reb[0][1] + mvmt[leb+11]],[ ic_reb[1][0] + mvmt[leb+12],  ic_reb[1][1] + mvmt[leb+13]],[ ic_reb[2][0] + mvmt[leb+14],  ic_reb[2][1] + mvmt[leb+15]],[ ic_reb[3][0] + mvmt[leb+16],  ic_reb[3][1] + mvmt[leb+17]],[ ic_reb[4][0] + mvmt[leb+18],  ic_reb[4][1] + mvmt[leb+19]]], np.int32)
	if (jaw == 2):
		print "Right eyebrow"
		print reb_contours
	reb_contours = np.append(reb_contours, reb_contours[::-1])
	ctr = np.array(reb_contours).reshape((-1,1,2)).astype(np.int32)
	cv2.drawContours(img, [ctr], -1, (0,0,0), 1)
	print "Le"
	#Left eye
	le_contours = np.ndarray(shape=(6,2))
	le_contours = np.array([[ ic_le[0][0] + le_mvmt[le+0],  ic_le[0][1] + le_mvmt[le+1]],[ ic_le[1][0] + le_mvmt[le+2],  ic_le[1][1] + le_mvmt[le+3]],[ ic_le[2][0] + le_mvmt[le+4],  ic_le[2][1] + le_mvmt[le+5]],[ ic_le[3][0] + le_mvmt[le+6],  ic_le[3][1] + le_mvmt[le+7]],[ ic_le[4][0] + le_mvmt[le+8],  ic_le[4][1] + le_mvmt[le+9]],[ ic_le[5][0] + le_mvmt[le+10],  ic_le[5][1] + le_mvmt[le+11]]], np.int32)
	if (jaw == 2):
		print "Left eye"
		print le_contours
	#le_contours = np.append(le_contours, le_contours[::-1])
	ctr = np.array(le_contours).reshape((-1,1,2)).astype(np.int32)
	cv2.drawContours(img, [ctr], -1, (0,0,0), 1)
	print "Re"
	#Right eye
	re_contours = np.ndarray(shape=(6,2))
	re_contours = np.array([[ ic_re[0][0] + re_mvmt[re+0],  ic_re[0][1] + re_mvmt[re+1]],[ ic_re[1][0] + re_mvmt[re+2],  ic_re[1][1] + re_mvmt[re+3]],[ ic_re[2][0] + re_mvmt[re+4],  ic_re[2][1] + re_mvmt[re+5]],[ ic_re[3][0] + re_mvmt[re+6],  ic_re[3][1] + re_mvmt[re+7]],[ ic_re[4][0] + re_mvmt[re+8],  ic_re[4][1] + re_mvmt[re+9]],[ ic_re[5][0] + re_mvmt[re+10],  ic_re[5][1] + re_mvmt[re+11]]], np.int32)
	if (jaw == 2):
		print "Right eye"
		print re_contours
	#re_contours = np.append(re_contours, re_contours[::-1])
	ctr = np.array(re_contours).reshape((-1,1,2)).astype(np.int32)
	cv2.drawContours(img, [ctr], -1, (0,0,0), 1)
	print "Ol"
	#Outer lip
	ol_contours = np.ndarray(shape=(12,2))
	ol_contours = np.array([[ ic_ol[0][0] + ol_mvmt[ol+0],  ic_ol[0][1] + ol_mvmt[ol+1]],[ ic_ol[1][0] + ol_mvmt[ol+2],  ic_ol[1][1] + ol_mvmt[ol+3]],[ ic_ol[2][0] + ol_mvmt[ol+4],  ic_ol[2][1] + ol_mvmt[ol+5]],[ ic_ol[3][0] + ol_mvmt[ol+6],  ic_ol[3][1] + ol_mvmt[ol+7]],[ ic_ol[4][0] + ol_mvmt[ol+8],  ic_ol[4][1] + ol_mvmt[ol+9]],[ ic_ol[5][0] + ol_mvmt[ol+10],  ic_ol[5][1] + ol_mvmt[ol+11]],[ ic_ol[6][0] + ol_mvmt[ol+12],  ic_ol[6][1] + ol_mvmt[ol+13]],[ ic_ol[7][0] + ol_mvmt[ol+14],  ic_ol[7][1] + ol_mvmt[ol+15]],[ ic_ol[8][0] + ol_mvmt[ol+16],  ic_ol[8][1] + ol_mvmt[ol+17]],[ ic_ol[9][0] + ol_mvmt[ol+18],  ic_ol[9][1] + ol_mvmt[ol+19]],[ ic_ol[10][0] + ol_mvmt[ol+20],  ic_ol[10][1] + ol_mvmt[ol+21]],[ ic_ol[11][0] + ol_mvmt[ol+22],  ic_ol[11][1] + ol_mvmt[ol+23]]], np.int32)
	if (jaw == 2):
		print "Outer lip"
		print ol_contours
	#ol_contours = np.append(ol_contours, ol_contours[::-1])
	ctr = np.array(ol_contours).reshape((-1,1,2)).astype(np.int32)
	cv2.drawContours(img, [ctr], -1, (0,0,0), 1)
	print "Il"
	#Inner lip
	il_contours = np.ndarray(shape=(8,2))
	il_contours = np.array([[ ic_il[0][0] + il_mvmt[il+0],  ic_il[0][1] + il_mvmt[il+1]],[ ic_il[1][0] + il_mvmt[il+2],  ic_il[1][1] + il_mvmt[il+3]],[ ic_il[2][0] + il_mvmt[il+4],  ic_il[2][1] + il_mvmt[il+5]],[ ic_il[3][0] + il_mvmt[il+6],  ic_il[3][1] + il_mvmt[il+7]],[ ic_il[4][0] + il_mvmt[il+8],  ic_il[4][1] + il_mvmt[il+9]],[ ic_il[5][0] + il_mvmt[il+10],  ic_il[5][1] + il_mvmt[il+11]],[ ic_il[6][0] + il_mvmt[il+12],  ic_il[6][1] + il_mvmt[il+13]],[ ic_il[7][0] + il_mvmt[il+14],  ic_il[7][1] + il_mvmt[il+15]]], np.int32)
	if (jaw == 2):
		print "Inner lip"
		print il_contours
	#il_contours = np.append(il_contours, il_contours[::-1])
	ctr = np.array(il_contours).reshape((-1,1,2)).astype(np.int32)
	cv2.drawContours(img, [ctr], -1, (0,0,0), 1)

	'''#update variables
	leb = leb + 20
	le = le + 12
	re = re + 12
	ol = ol + 24
	il = il + 16'''

	#cv2.imwrite("img.jpg",img[0:1000, 320:960]

	#cv2.imshow("Replay",img)
	print "After imshow"
	return img[0:1000, 320:960]


f = []

def stop():
	global stop_flag, record_flag
	stop_flag = 1
	record_flag = 0
	print "Recording stopped"


def is_record():
	global f
	global record_flag, stop_flag
	record_flag=1
	if stop_flag==1:
		print "Operation not allowed: Cannot record again."
		#Thread(target = record(f)).start()

def is_replay():
	global replay_flag
	replay_flag = 1
	print "inside is_replay"



class vid():      
	def __init__(self,video_capture,root,canvas,canvas2,jaw,leb,le,re,ol,il):
		self.video_capture = video_capture
		self.root = root
		self.canvas = canvas
		self.canvas2=canvas2
		self.jaw = jaw
		self.leb = leb
		self.le = le
		self.re = re
		self.ol = ol
		self.il = il

	def update_video(self):
		global f, record_flag
		global j_mvmt
		global root
		(self.readsuccessful,self.f) = self.video_capture.read()
		self.f = self.f[0:1000, 320:960]
		self.f = cv2.flip(self.f, 1)
		self.g = cv2.cvtColor(self.f, cv2.COLOR_BGR2RGB)
		f = self.g
		if (record_flag == 1):
			self.f = record(f)
		self.a = Image.fromarray(self.f)
		self.b = ImageTk.PhotoImage(image=self.a)
		self.canvas.create_image(0,0,image=self.b,anchor=tk.NW)
		self.root.update()
		if (record_flag == 0 and replay_flag == 1):
			self.animate()
		self.root.after(33,self.update_video)
	
	def animate(self):
		if self.jaw <len(j_mvmt):
			print self.jaw
			self.f2 = replay(self.jaw,self.leb,self.le,self.re,self.ol,self.il)
			#replay(self.jaw,self.leb,self.le,self.re,self.ol,self.il)
			self.a2 = Image.fromarray(self.f2)
			self.b2 = ImageTk.PhotoImage(image = self.a2)
			self.canvas2.create_image(0,0,image=self.b2,anchor=tk.NW)
			self.root.update()
			time.sleep(1)
			self.jaw = self.jaw + 34
			self.leb += 20
			self.le += 12
			self.re += 12
			self.ol += 24
			self.il += 16

			if (self.jaw >= len(j_mvmt)):
				exit(0)
			root.after(33,self.animate())
	'''
	def update_animation(self):
		print "inside ua"
		self.f2 = replay()
		self.a2 = Image.fromarray(self.f2)
		self.b2 = ImageTk.PhotoImage(image=self.a2)
		self.canvas2.create_image(0,0,image=self.b2,anchor=tk.NW)
		self.root.update()'''

root = None

if __name__ == '__main__':
	#global replay_flag
	root = tk.Tk()
	videoframe = tk.LabelFrame(root,text='Captured video')
	videoframe.grid(column=0,row=0,columnspan=2,rowspan=2,padx=5, pady=5, ipadx=5, ipady=5)
	canvas = tk.Canvas(videoframe, width=550,height=600, bg="black")
	canvas.grid(column=0,row=0)
	canvas2 = tk.Canvas(videoframe, width=550,height=600)
	canvas2.grid(column=1,row=0)

	coord = 10, 50, 240, 210
	
	button1 = tk.Button(text='Record',master=videoframe, command=is_record)
	button1.grid(column=0,row=1)
	button2 = tk.Button(text='Stop',master=videoframe, command=stop)
	button2.grid(column=0,row=2)
	button3 = tk.Button(text='Replay',master=videoframe, command=is_replay)
	button3.grid(column=1,row=1)
	button4 = tk.Button(text='Quit',master=videoframe,command=root.destroy)
	button4.grid(column=1,row=2)
	
	video_capture = cv2.VideoCapture(0)
	jaw = 36
	leb = 22
	le = 14
	re = 14
	ol = 26
	il = 18
	x = vid(video_capture,root,canvas,canvas2,jaw,leb,le,re,ol,il)
	#root.after(0,Thread(target = x.update_video).start())
	print "abc"
	#if replay_flag == 1:
	#	print "inside replay_flag"
	#	root.x.update_animation
	root.after(0,x.update_video)

	
	root.mainloop()
	del video_capture
#try:
#	root.mainloop()
#except:
#	print "GUI FAILED"


