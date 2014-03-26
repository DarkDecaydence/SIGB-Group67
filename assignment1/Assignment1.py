import cv2
import cv
import pylab
import math
from SIGBTools import RegionProps
from SIGBTools import getLineCoordinates
from SIGBTools import ROISelector
from SIGBTools import getImageSequence
from SIGBTools import getCircleSamples
from SIGBTools import getLineCoordinates
import numpy as np
import sys
from scipy.cluster.vq import *
#from scipy.misc import imresize
from matplotlib.pyplot import *



inputFile = "Sequences/EyeBizaro.avi"
outputFile = "Sequences/PupilBizaro_decent.avi"

#--------------------------
#         Global variable
#--------------------------
global imgOrig,leftTemplate,rightTemplate,frameNr
imgOrig = [];
#These are used for template matching
leftTemplate = []
rightTemplate = []
frameNr =0;


def eucledianDistance(point1, point2):
	x = point2[0]-point1[0]
	y = point2[1] - point1[1]
	x = x**2
	y = y**2
	return math.sqrt(x+y)

def getGradientImageInfo(gray):
	xIm = cv2.Sobel(gray, -1, 1, 0)
	yIm = cv2.Sobel(gray, -1, 0, 1)
	#xIm, yIm = np.gradient(gray)
	m, n = gray.shape
	angleIm =  np.array([[0] * n] * m)
	magIm = np.array([[0] * n] * m)
	for x in range(m):
		for y in range(n):
			angleIm[x][y] = math.atan2(yIm[x][y], xIm[x][y]) * (180 / math.pi)
			magIm[x][y] = math.sqrt(xIm[x][y] ** 2 + yIm[x][y] ** 2)
	return (xIm, yIm, magIm, angleIm)

def circleTest(img, center, radius, samples):
	P = getCircleSamples(center=center, radius=radius, nPoints=samples)
	t = 0
	for (xf,yf,dxf,dyf) in P:
		x, y, dx, dy = int(xf), int(yf), int(dxf * radius), int(dyf * radius)
		cv2.circle(img,(x, y), 2, (0,255,0), 4)
		cv2.line(img, (x - dx, y - dy), (x + dx, y + dy), (0,255,0))

def GetPupil(gray,thr, minSize, maxSize):
	'''Given a gray level image, gray and threshold value return a list of pupil locations'''
	tempResultImg = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR) #used to draw temporary results
	#cv2.imshow("TempResults",tempResultImg)

	props = RegionProps()
	val,binI = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
	
	# Custom Morphology --
	kernel = np.ones((10, 10), np.uint8)
	binI = cv2.dilate(binI, kernel, iterations = 1)
	binI = cv2.erode(binI, kernel, iterations = 1)
	
	cv2.imshow("Threshold",binI)
	#Calculate blobs
	contours, hierarchy = cv2.findContours(binI, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	pupils = [];
	# YOUR IMPLEMENTATION HERE !!!
	for con in contours:
		if len(con) >= 5:
			p = props.CalcContourProperties(con, ["Area", "Boundingbox", "Centroid", "Extend"])
			if minSize < p["Area"] < maxSize \
			and p["Extend"] > 0.5:
				pupils.append(cv2.fitEllipse(con))
		
	
	return pupils

def GetGlints(gray,thr, maxSize):
	''' Given a gray level image, gray and threshold
	value return a list of glint locations'''
	# YOUR IMPLEMENTATION HERE !!!!
	tempResultImg = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR) #used to draw temporary results
	
	props = RegionProps()
	val,binI = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
	cv2.imshow("TempResults",tempResultImg)
	
	#kernel = np.ones((4, 4), np.uint8)
	#binI = cv2.dilate(binI, kernel, iterations = 1)
	#binI = cv2.erode(binI, kernel, iterations = 1)
	
	#cv2.imshow("Threshold",binI)
	#Calculate blobs
	contours, hierarchy = cv2.findContours(binI, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	glints = [];
	# YOUR IMPLEMENTATION HERE !!!
	for con in contours:
		p = props.CalcContourProperties(con, ["Centroid","Area"])
		if p["Area"] < maxSize:
			glints.append(p['Centroid'])
	return glints

def GetIrisUsingThreshold(gray, thr):
	''' Given a gray level image, gray and threshold
	value return a list of iris locations'''

	props = RegionProps()
	val,binI = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
	
	cv2.imshow("Threshold",binI)
	#Calculate blobs
	contours, hierarchy = cv2.findContours(binI, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	iris = [];
	for con in contours:
		if len(con) >= 5:
			p = props.CalcContourProperties(con, ["Area", "Boundingbox", "Centroid", "Extend"])
			if 200 < p["Area"] \
			and p["Extend"] > 0.3:
				iris.append(cv2.fitEllipse(con))
		
	
	return iris

def circularHough(gray):
	''' Performs a circular hough transform of the image, gray and shows the  detected circles
	The circe with most votes is shown in red and the rest in green colors '''
 #See help for http://opencv.itseez.com/modules/imgproc/doc/feature_detection.html?highlight=houghcircle#cv2.HoughCircles
	blur = cv2.GaussianBlur(gray, (31,31), 11)

	dp = 6; minDist = 30
	highThr = 20 #High threshold for canny
	accThr = 850; #accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected
	maxRadius = 50;
	minRadius = 155;
	circles = cv2.HoughCircles(blur,cv2.cv.CV_HOUGH_GRADIENT, dp,minDist, None, highThr,accThr,maxRadius, minRadius)

	#Make a color image from gray for display purposes
	gColor = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
	if (circles !=None):
	 #print circles
	 all_circles = circles[0]
	 M,N = all_circles.shape
	 k=1
	 for c in all_circles:
			cv2.circle(gColor, (int(c[0]),int(c[1])),c[2], (int(k*255/M),k*128,0))
			K=k+1
	 c=all_circles[0,:]
	 cv2.circle(gColor, (int(c[0]),int(c[1])),c[2], (0,0,255),5)
	 cv2.imshow("hough",gColor)

def GetIrisUsingNormals(img, gray, pupils, normalLength):
	xIm, yIm, magIm, angleIm = getGradientImageInfo(gray)
	iris = []
	for ((pX, pY), pRad, pAng) in pupils:
		P = getCircleSamples(center = (pX, pY), radius=120, nPoints=50)
		irisPoints = []
		for (xf, yf, dxf, dyf) in P:
			maxGrad = 0
			maxPoint = (-1, -1)
			band = 40
			ix, iy, idx, idy = int(xf), int(yf), int(dxf * band), int(dyf * band)
			angle = math.atan2(dyf, dxf)* (180 / math.pi)
			maxX, maxY = magIm.shape
			for (x, y) in getLineCoordinates((ix - idx, iy - idy), (ix + idx, iy + idy)):
				if 0 < x < maxX and 0 < y < maxY and magIm[x][y] > maxGrad and math.fabs(angleIm[x][y] - angle) > 45:
					maxGrad = magIm[x][y]
					maxPoint = (x, y)
			if maxGrad > 0:
				irisPoints.append(maxPoint)
		iris.append(cv2.fitEllipse(np.array(irisPoints)))
	return iris

def GetIrisUsingSimplifyedHough(gray,pupil):
	''' Given a gray level image, gray
	return a list of iris locations using a simplified Hough transformation'''
	# YOUR IMPLEMENTATION HERE !!!!
	pass

def GetEyeCorners(leftTemplate, rightTemplate,pupilPosition=None):
	pass

def FilterPupilGlint(pupils,glints,glinsDistance):
	''' Given a list of pupil candidates and glint candidates returns a list of pupil and glints'''
	for i, glint in enumerate(glints):
		keep = False
		for pupil in pupils:
			if eucledianDistance(glint, pupil[0]) < glinsDistance:
				keep = True
				break
		if not keep:
			glints[i] = (-1, -1)



def update(I):
	'''Calculate the image features and display the result based on the slider values'''
	#global drawImg
	global frameNr,drawImg
	img = I.copy()
	sliderVals = getSliderVals()
	gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	gray = cv2.equalizeHist(gray)
	
	pupils = GetPupil(gray,sliderVals['pupilThr'],sliderVals['minSize'],sliderVals['maxSize'])
	glints = GetGlints(gray,sliderVals['glintThr'],sliderVals['glinsMax'])
	FilterPupilGlint(pupils,glints, sliderVals["glinsDistance"])
	#iris = GetIrisUsingNormals(img, gray, pupils, sliderVals["irisThr"])

	#Do template matching
	global leftTemplate
	global rightTemplate
	GetEyeCorners(leftTemplate, rightTemplate)
	#Display results
	global frameNr,drawImg
	x,y = 10,10
	#setText(img,(x,y),"Frame:%d" %frameNr)
	sliderVals = getSliderVals()

	# for non-windows machines we print the values of the threshold in the original image
	if sys.platform != 'win32':
		step=18
		cv2.putText(img, "pupilThr :"+str(sliderVals['pupilThr']), (x, y+step), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)
		cv2.putText(img, "glintThr :"+str(sliderVals['glintThr']), (x, y+2*step), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)
		cv2.putText(img, "minSize :"+str(sliderVals['minSize']), (x, y+3*step), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)
		cv2.putText(img, "maxSize :"+str(sliderVals['maxSize']), (x, y+4*step), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)
	

		#Uncomment these lines as your methods start to work to display the result in the
		#original image
	for pupil in pupils:
		cv2.ellipse(img,pupil,(0,255,0),1)
		C = int(pupil[0][0]),int(pupil[0][1])
		cv2.circle(img,C, 2, (0,0,255),4)
	for glint in glints:
		C = int(glint[0]),int(glint[1])
		cv2.circle(img, C, 2,(255, 0, 255),2)
	
	for ir in iris:
		cv2.ellipse(img,ir,(255,255,0),1)
		#circularHough(gray)

	#copy the image so that the result image (img) can be saved in the movie
	
	cv2.imshow('Result',img)
	drawImg = img.copy()
	
	#detectPupilKMeans(gray, K=sliderVals['distWeight'], distanceWeight=2)


def printUsage():
	print "Q or ESC: Stop"
	print "SPACE: Pause"
	print "r: reload video"
	print 'm: Mark region when the video has paused'
	print 's: toggle video  writing'
	print 'c: close video sequence'

def run(fileName,resultFile):

	''' MAIN Method to load the image sequence and handle user inputs'''
	global imgOrig, frameNr,drawImg
	setupWindowSliders()
	props = RegionProps();
	cap,imgOrig,sequenceOK = getImageSequence(fileName)
	videoWriter = 0

	frameNr =0
	if(sequenceOK):
		update(imgOrig)
	printUsage()
	frameNr=0;
	saveFrames = False

	while(sequenceOK):
		sliderVals = getSliderVals();
		frameNr=frameNr+1
		ch = cv2.waitKey(1)
		#Select regions
		if(ch==ord('m')):
			if(not sliderVals['Running']):
				roiSelect=ROISelector(imgOrig)
				pts,regionSelected= roiSelect.SelectArea('Select left eye corner',(400,200))
				if(regionSelected):
					leftTemplate = imgOrig[pts[0][1]:pts[1][1],pts[0][0]:pts[1][0]]

		if ch == 27:
			break
		if (ch==ord('s')):
			if((saveFrames)):
				videoWriter.release()
				saveFrames=False
				print "End recording"
			else:
				imSize = np.shape(imgOrig)
				videoWriter = cv2.VideoWriter(resultFile, cv.CV_FOURCC('D','I','V','3'), 15.0,(imSize[1],imSize[0]),True) #Make a video writer
				saveFrames = True
				print "Recording..."



		if(ch==ord('q')):
			break
		if(ch==32): #Spacebar
			sliderVals = getSliderVals()
			cv2.setTrackbarPos('Stop/Start','Threshold',not sliderVals['Running'])
		if(ch==ord('r')):
			frameNr =0
			sequenceOK=False
			cap,imgOrig,sequenceOK = getImageSequence(fileName)
			update(imgOrig)
			sequenceOK=True

		sliderVals=getSliderVals()
		if(sliderVals['Running']):
			sequenceOK, imgOrig = cap.read()
			if(sequenceOK): #if there is an image
				update(imgOrig)
			if(saveFrames):
				videoWriter.write(drawImg)
	if(videoWriter!=0):
		videoWriter.release()
        print "Closing videofile..."
#------------------------

def detectPupilKMeans(gray,K=2,distanceWeight=2,reSize=(40,40)):
	''' Detects the pupil in the image, gray, using k-means
		gray              : grays scale image
		K                 : Number of clusters
		distanceWeight    : Defines the weight of the position parameters
		reSize            : the size of the image to do k-means on
	'''
	#Resize for faster performance
	smallI = cv2.resize(gray, reSize)
	M,N = smallI.shape
	#Generate coordinates in a matrix
	X,Y = np.meshgrid(range(M),range(N))
	#Make coordinates and intensity into one vectors
	z = smallI.flatten()
	x = X.flatten()
	y = Y.flatten()
	O = len(x)
	#make a feature vectors containing (x,y,intensity)
	features = np.zeros((O,3))
	features[:,0] = z;
	features[:,1] = y/distanceWeight; #Divide so that the distance of position weighs less than intensity
	features[:,2] = x/distanceWeight;
	features = np.array(features,'f')
	# cluster data
	centroids,variance = kmeans(features,K)
	#use the found clusters to map
	label,distance = vq(features,centroids)
	# re-create image from
	labelIm = np.array(np.reshape(label,(M,N)))
	f = figure(1)
	imshow(labelIm)
	f.canvas.draw()
	f.show()

#------------------------------------------------
#   Methods for segmentation
#------------------------------------------------
def detectPupilKMeans(gray,K=2,distanceWeight=2,reSize=(40,40)):
	''' Detects the pupil in the image, gray, using k-means
			gray              : grays scale image
			K                 : Number of clusters
			distanceWeight    : Defines the weight of the position parameters
			reSize            : the size of the image to do k-means on
		'''
	#Resize for faster performance
	smallI = cv2.resize(gray, reSize)
	M,N = smallI.shape
	#Generate coordinates in a matrix
	X,Y = np.meshgrid(range(M),range(N))
	#Make coordinates and intensity into one vectors
	z = smallI.flatten()
	x = X.flatten()
	y = Y.flatten()
	O = len(x)
	#make a feature vectors containing (x,y,intensity)
	features = np.zeros((O,3))
	features[:,0] = z;
	features[:,1] = y/distanceWeight; #Divide so that the distance of position weighs less than intensity
	features[:,2] = x/distanceWeight;
	features = np.array(features,'f')
	# cluster data
	centroids,variance = kmeans(features,K)
	#use the found clusters to map
	label,distance = vq(features,centroids)
	# re-create image from
	labelIm = np.uint8(np.array(np.reshape(label,(M,N))))
	
	props = RegionProps()
	val,binI = cv2.threshold(labelIm, 2, 255, cv2.THRESH_BINARY_INV)
	contours, hierarchy = cv2.findContours(binI, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	pupils = [];
	# YOUR IMPLEMENTATION HERE !!!
	for con in contours:
		p = props.CalcContourProperties(con, ["Area", "Boundingbox", "Centroid", "Extend"])
		if len(con > 5):
			pupils.append(con)
	
	labelIm = np.zeros((M, N))
	
	for pupil in pupils:
		for p in pupil:
			labelIm[p[0][0]][p[0][1]] = 1
	
	f = figure(1)
	imshow(labelIm)
	f.canvas.draw()
	f.show()

def detectPupilHough(gray):
	#Using the Hough transform to detect ellipses
	blur = cv2.GaussianBlur(gray, (9,9),3)
	##Pupil parameters
	dp = 6; minDist = 10
	highThr = 30 #High threshold for canny
	accThr = 600; #accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected
	maxRadius = 70;
	minRadius = 20;
	#See help for http://opencv.itseez.com/modules/imgproc/doc/feature_detection.html?highlight=houghcircle#cv2.HoughCirclesIn thus
	circles = cv2.HoughCircles(blur,cv2.cv.CV_HOUGH_GRADIENT, dp,minDist, None, highThr,accThr,minRadius, maxRadius)
	#Print the circles
	gColor = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
	if (circles !=None):
		#print circles
		all_circles = circles[0]
		M,N = all_circles.shape
		k=1
		for c in all_circles:
			cv2.circle(gColor, (int(c[0]),int(c[1])),c[2], (int(k*255/M),k*128,0))
			K=k+1
			#Circle with max votes
		c=all_circles[0,:]
		cv2.circle(gColor, (int(c[0]),int(c[1])),c[2], (0,0,255))
	cv2.imshow("hough",gColor)
#--------------------------
#         UI related
#--------------------------

def setText(dst, (x, y), s):
	cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.CV_AA)
	cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)


def setupWindowSliders():
	''' Define windows for displaying the results and create trackbars'''
	cv2.namedWindow("Result")
	cv2.namedWindow('Threshold')
	cv2.namedWindow("TempResults")
	#Threshold value for the pupil intensity
	cv2.createTrackbar('pupilThr','Threshold', 15, 255, onSlidersChange)
	#Threshold value for the glint intensities
	cv2.createTrackbar('glintThr','Threshold', 245, 255,onSlidersChange)
	#define the minimum and maximum areas of the pupil
	cv2.createTrackbar('irisThr','Threshold', 128,255, onSlidersChange)
	cv2.createTrackbar('minSize','Threshold', 500, 5000, onSlidersChange)
	cv2.createTrackbar('maxSize','Threshold', 4800, 5000, onSlidersChange)
	cv2.createTrackbar('glinsMax','Threshold', 20,150, onSlidersChange)
	cv2.createTrackbar('glinsDistance','Threshold', 50,200, onSlidersChange)
	#Value to indicate whether to run or pause the video
	cv2.createTrackbar('Stop/Start','Threshold', 0,1, onSlidersChange)

def getSliderVals():
	'''Extract the values of the sliders and return these in a dictionary'''
	sliderVals={}
	sliderVals['pupilThr'] = cv2.getTrackbarPos('pupilThr', 'Threshold')
	sliderVals['glintThr'] = cv2.getTrackbarPos('glintThr', 'Threshold')
	sliderVals['minSize'] = cv2.getTrackbarPos('minSize', 'Threshold')
	sliderVals['maxSize'] = cv2.getTrackbarPos('maxSize', 'Threshold')
	sliderVals['glinsMax'] = cv2.getTrackbarPos('glinsMax', 'Threshold')
	sliderVals['glinsDistance'] = cv2.getTrackbarPos('glinsDistance', 'Threshold')
	sliderVals['irisThr'] = cv2.getTrackbarPos('irisThr', 'Threshold')
	sliderVals['Running'] = 1==cv2.getTrackbarPos('Stop/Start', 'Threshold')
	return sliderVals

def onSlidersChange(dummy=None):
	''' Handle updates when slides have changed.
	 This  function only updates the display when the video is put on pause'''
	global imgOrig;
	sv=getSliderVals()
	if(not sv['Running']): # if pause
		update(imgOrig)

#--------------------------
#         main
#--------------------------
run(inputFile, outputFile)