#Import required modules
import cv2
import numpy as np
import math
import time

import Tkinter as tk
from PIL import Image, ImageTk

stop_flag = 0
record_flag = 0
replay_flag = 0
m_contours = []
l_contours = []

def record(crop_img):
	global l_contours, m_contours
	#cap = cv2.VideoCapture(0)
	#while(cap.isOpened()):
		#ret, img = cap.read()
	crop_img = cv2.flip(crop_img, 1)
	#cv2.rectangle(img,(200,200),(700,700),(0,255,0),0)
	#crop_img = img[200:700, 200:700]
	grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
	value = (31, 31)
	blurred = cv2.GaussianBlur(grey, value, 0)
	_, thresh1 = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	#cv2.imshow('Thresholded', thresh1)

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
	cv2.drawContours(crop_img, [handContour], -1, (0,0,255), 1)
	
	m_contours += [handContour]
	l_contours += [len(handContour)]

	x,y,w,h = cv2.boundingRect(handContour)
	cv2.rectangle(crop_img,(x,y),(x+w,y+h),(0,0,255),0)

	hullHandContour = cv2.convexHull(handContour,returnPoints = False)
	hullPoints = [handContour[i[0]] for i in hullHandContour]

	defects = cv2.convexityDefects(handContour,hullHandContour)

	for i in range(defects.shape[0]):
		s,e,f,d = defects[i,0]
		start = tuple(handContour[s][0])
		end = tuple(handContour[e][0])
		far = tuple(handContour[f][0])

	drawing = np.zeros(crop_img.shape,np.uint8)
	cv2.drawContours(drawing,[handContour],0,(0,255,0),0)

	cv2.drawContours(thresh1, contours, -1, (0,255,0), 3)

	handMoments = cv2.moments(handContour)
	handXCenterMoment = int(handMoments["m10"] / handMoments["m00"])
	handYCenterMoment = int(handMoments["m01"] / handMoments["m00"])
	handMoment = (handXCenterMoment, handYCenterMoment)
	#cv2.circle(img2,(handMoment), 1, (0,0,0), 1)

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
	cv2.circle(crop_img,(palmCenter), int(palmRadius), (0,0,255), 1)
	
	crop_img2 = img2[0:1000, 320:960]
	#print crop_img2.shape
	#final = np.concatenate((crop_img, crop_img2),axis = 1)
	return crop_img
	#cv2.imshow('img2',img2)
	#cv2.imshow('Gesture', img)
	#all_img = np.hstack((drawing, crop_img))
	#cv2.imshow('Contours', all_img)
	#cv2.imshow('FINAL',final)
	#if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
	#	break

def replay(l_ctr):
	global stop_flag, replay_flag, record_flag
	if stop_flag==0:
		stop() 
	print "Replaying :"+str(l_ctr)

	global l_contours, m_contours
	image = cv2.imread('bk.png')
	ctr = []

	image = cv2.imread('bk.png')
	ctr = m_contours[l_ctr]
	cv2.drawContours(image, [ctr], -1, (0,0,0), 1)

	#cv2.imshow('Animation',image)
	return image[0:1000, 0:640]

f = []

def stop():
	global stop_flag, record_flag
	stop_flag = 1
	record_flag = 0
	print "Recording stopped"


def is_record():
	global f
	global record_flag, stop_flag
	record_flag = 1 
	if stop_flag == 1:
		print "Operation not allowed: Cannot record again."
		#Thread(target = record(f)).start()

def is_replay():
	global replay_flag
	replay_flag = 1
	print "inside is_replay"

class vid():      
	def __init__(self,video_capture,root,canvas,canvas2,l_ctr):
		self.video_capture = video_capture
		self.root = root
		self.canvas = canvas
		self.canvas2 = canvas2
		self.l_ctr = l_ctr

	def update_video(self):
		global f, record_flag
		global m_contours
		global root
		(self.readsuccessful,self.f) = self.video_capture.read()
		self.f = self.f[0:1000, 320:960]
		f = self.f
		if (record_flag == 1):
			self.f = record(f)
		self.f = cv2.cvtColor(self.f, cv2.COLOR_BGR2RGB)
		self.a = Image.fromarray(self.f)
		self.b = ImageTk.PhotoImage(image=self.a)
		self.canvas.create_image(0,0,image=self.b,anchor=tk.NW)
		self.root.update()
		if (record_flag == 0 and replay_flag == 1):
			self.animate()
		self.root.after(33,self.update_video)
	
	def animate(self):
		global m_contours
		global root
		if self.l_ctr < len(m_contours):
			print self.l_ctr
			self.f2 = replay(self.l_ctr)
			self.a2 = Image.fromarray(self.f2)
			self.b2 = ImageTk.PhotoImage(image = self.a2)
			self.canvas2.create_image(0,0,image=self.b2,anchor=tk.NW)
			self.root.update() 
			time.sleep(1)
			print "Got up!!"
			self.l_ctr = self.l_ctr + 1
			root.after(33,self.animate)

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
	l_ctr = 0
	x = vid(video_capture,root,canvas,canvas2, l_ctr)
	#root.after(0,Thread(target = x.update_video).start())
	print "abc"
	#if replay_flag == 1:
	#	print "inside replay_flag"
	#	root.x.update_animation
	root.after(0,x.update_video)

	
	root.mainloop()
	video_capture.release()
	del video_capture
#try:
#	root.mainloop()
#except:
#	print "GUI FAILED"


