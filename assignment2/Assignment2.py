#from scipy import ndimage
import cv2
import cv
import numpy as np
from numpy import linalg
from pylab import *
from matplotlib import *
from matplotlib.pyplot import *
from scipy import *
import math
from SIGBTools import *
import sys
def frameTrackingData2BoxData(data):
    #Convert a row of points into tuple of points for each rectangle
    pts= [ (int(data[i]),int(data[i+1])) for i in range(0,11,2) ]
    boxes = [];
    for i in range(0,7,2):
        box = tuple(pts[i:i+2])
        boxes.append(box)   
    return boxes


def simpleTextureMap():

    I1 = cv2.imread('Images/ITULogo.jpg')
    I2 = cv2.imread('Images/ITUMap.bmp')

    #Print Help
    H,Points  = getHomographyFromMouse(I1,I2,-4)
    h, w,d = I2.shape
    overlay = cv2.warpPerspective(I1, H,(w, h))
    M = cv2.addWeighted(I2, 0.5, overlay, 0.5, 0)

    cv2.imshow("Overlayed Image",M)
    cv2.waitKey(0)

def showImageAndPlot():
    #A simple attenmpt to get mouse inputs and display images using matplotlib
    I = cv2.imread('Images/groundfloor.bmp')
    drawI = I.copy()
    #make figure and two subplots
    fig = figure(1) 
    ax1  = subplot(1,2,1) 
    ax2  = subplot(1,2,2) 
    ax1.imshow(I) 
    ax2.imshow(drawI)
    ax1.axis('image') 
    ax1.axis('off') 
    points = fig.ginput(5) 
    fig.hold('on')
    
    for p in points:
        #Draw on figure
        subplot(1,2,1)
        plot(p[0],p[1],'rx')
        #Draw in image
        cv2.circle(drawI,(int(p[0]),int(p[1])),2,(0,255,0),10)
    ax2.cla
    ax2.imshow(drawI)
    draw() #update display: updates are usually defered 
    show()
    savefig('somefig.jpg')
    cv2.imwrite("drawImage.jpg", drawI)


def texturemapGridSequence():
    """ Skeleton for texturemapping on a video sequence"""
    fn = 'GridVideos/grid1.mp4'
    cap = cv2.VideoCapture(fn)
    drawContours = True;

    texture = cv2.imread('Images/cat.jpg')
    texture = cv2.pyrDown(texture)


    mTex,nTex,_ = texture.shape
    texCorners = np.array([[float(nTex), float(mTex)], [float(0.0), float(mTex)], [float(nTex), float(0.0)], [float(0.0), float(0.0)]])


    #load Tracking data
    running, imgOrig = cap.read()
    mI,nI,_ = cv2.pyrDown(imgOrig).shape

    cv2.imshow("win2",imgOrig)

    pattern_size = (9, 6)
    writer = cv2.VideoWriter('GridTexture.avi', cv.CV_FOURCC('D','I','V','3'), 15.0, (nI, mI), True)

    idx = [0,8,45,53]
    while(running):
    #load Tracking data
        running, imgOrig = cap.read()
        if(running):
            imgOrig = cv2.pyrDown(imgOrig)
            gray = cv2.cvtColor(imgOrig,cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, pattern_size)
            if found:
                term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1 )
                
                rect = np.array([[float(corners[t, 0, 0]), float(corners[t, 0, 1])] for t in idx])
                H, _ = cv2.findHomography(texCorners, rect)
                mI,nI,_ = imgOrig.shape
                overlay = cv2.warpPerspective(texture, H,(nI, mI))
                imgOrig = cv2.addWeighted(imgOrig, 0.4, overlay, 0.6, 0)
            cv2.imshow("win2",imgOrig)
            writer.write(imgOrig)
            cv2.waitKey(1)
    writer.release()

def textureMapGroundFloor():
    fn = "GroundFloorData/sunclipds.avi"
    cap = cv2.VideoCapture(fn)
	
    running, imgOrig = cap.read()
    dataFile = np.loadtxt('GroundFloorData/trackingdata.dat')
    m,n = dataFile.shape
	
    texture = cv2.imread("Images/cat.jpg")   
	
    H, _ = getHomographyFromMouse(texture, imgOrig, -4)
	
    h, w, _ = imgOrig.shape
    overlay = cv2.warpPerspective(texture, H, (w, h))
    
    for i in range(m):
        running, imgOrig = cap.read()
        if running:
            frame = cv2.addWeighted(imgOrig, 0.5, overlay, 0.5, 0)
            cv2.imshow("cat projection", frame)
        cv2.waitKey(1)
    
    cv2.waitKey(0)

def realisticTexturemapGroundFloor():
    fn = "GroundFloorData/sunclipds.avi"
    cap = cv2.VideoCapture(fn)
	
    running, imgOrig = cap.read()
    dataFile = np.loadtxt('GroundFloorData/trackingdata.dat')
    m,n = dataFile.shape
	
    ituMap = cv2.imread("Images/ITUMap.bmp")
    h, w, _ = imgOrig.shape
    
    fig = figure(2)
    ax = subplot(1,2,1)
    ax.imshow(ituMap)
    ax.axis('image')
    title("Click in the image")
    fig.canvas.draw()
    ax.hold('On')
    texPoint = fig.ginput(1, -1)
    
    for i in range(m):
        running, imgOrig = cap.read()
        if running:
            img = realisticTexturemapLoadHomography(0.1, texPoint[0], ituMap, imgOrig)
            cv2.imshow("realistic projection", img)
        cv2.waitKey(1)
        break
    cv2.waitKey(0)

def realisticTexturemapLoadHomography(scale, point, map, ground):
    
    Hmg = linalg.inv(numpy.load("H_G_M.npy"))
    texture = cv2.imread("Images/cat.jpg")
    mTex,nTex,_ = texture.shape
    texCorners = np.array([[float(0.0), float(0.0)], [float(nTex), float(0.0)], [float(0.0), float(mTex)], [float(nTex), float(mTex)]])
    
    mSca = (mTex * scale) / 2
    nSca = (nTex * scale) / 2
    
    mapCorners = np.array([[point[0] - nSca, point[1] - mSca], [point[0] + nSca, point[1] - mSca], \
    [point[0] - nSca, point[1] + mSca], [point[0] + nSca, point[1] + mSca]])
    
    Htm, _ = cv2.findHomography(texCorners, mapCorners)
    nM, mM, _ = map.shape
    nG, mG, _ = ground.shape
    overlay = cv2.warpPerspective(cv2.warpPerspective(texture, Htm,(mM, nM)), Hmg, (mG, nG))
    return cv2.addWeighted(ground, 0.5, overlay, 0.5, 0)

def realisticTexturemap(Hmg, scale, point, map, ground):
    
    texture = cv2.imread("Images/cat.jpg")
    mTex,nTex,_ = texture.shape
    texCorners = np.array([[float(0.0), float(0.0)], [float(nTex), float(0.0)], [float(0.0), float(mTex)], [float(nTex), float(mTex)]])
    
    mSca = (mTex * scale) / 2
    nSca = (nTex * scale) / 2
    
    mapCorners = np.array([[point[0] - nSca, point[1] - mSca], [point[0] + nSca, point[1] - mSca], \
    [point[0] - nSca, point[1] + mSca], [point[0] + nSca, point[1] + mSca]])
    
    Htm, _ = cv2.findHomography(texCorners, mapCorners)
    nM, mM, _ = map.shape
    nG, mG, _ = ground.shape
    overlay = cv2.warpPerspective(cv2.warpPerspective(texture, Htm,(mM, nM)), Hmg, (mG, nG))
    return cv2.addWeighted(ground, 0.5, overlay, 0.5, 0)


def showFloorTrackingData():
    #Load videodata
    fn = "GroundFloorData/sunclipds.avi"
    cap = cv2.VideoCapture(fn)
	
    #load Tracking data
    running, imgOrig = cap.read()
    dataFile = np.loadtxt('GroundFloorData/trackingdata.dat')
    m,n = dataFile.shape
	
    ituMap = cv2.imread("Images/ITUMap.bmp")
    
    Hgm, _ = getHomographyFromMouse(imgOrig, ituMap)
    
    fig = figure()
    path = []
    h1, w1 = imgOrig.shape[:2]
    h2, w2 = ituMap.shape[:2]
    videoSize = (h1 + h2, max(w1, w2), 3)
    
    videoWriter = cv2.VideoWriter("GroundFloorData/tracking.avi", cv.CV_FOURCC('D','I','V','3'), 15.0,(videoSize[1],videoSize[0]),True)
    
    for k in range(m):
        if k % 50 == 0:
            key = cv2.waitKey(0)
            if key == 100:
                sys.exit(0)
        running, imgOrig = cap.read()
        
        if(running):
            boxes = frameTrackingData2BoxData(dataFile[k,:])
            boxColors = [(255,0,0),(0,255,0),(0,0,255)]
            for k in range(0,3):
                aBox = boxes[k]
                cv2.rectangle(imgOrig, aBox[0], aBox[1], boxColors[k])
            
            g = (boxes[1][0][0] + ((boxes[1][1][0] - boxes[1][0][0]) / 2), boxes[1][1][1])
            path.append(g)
            
            Hg = (g[0], g[1], 1)
            Hma = np.dot(Hgm, Hg)
            ma = (int(Hma[0] / Hma[2]), int(Hma[1] / Hma[2]))
            
            trace = copy(ituMap)
            cv2.circle(trace, ma, 2, (0, 0, 255), 2)
            cv2.imshow("map", trace)
            
            vid = np.zeros(videoSize, np.uint8)
            vid[:h1, (w1 / 2):w1 + (w1 / 2)] = imgOrig
            vid[h1: h1 + h2, :w2] = trace
            cv2.imshow("result", vid)
            videoWriter.write(vid)
            
            cv2.waitKey(1)
    
    DisplayTrace(ituMap, path, Hgm)
    cv2.waitKey(0)

def DisplayTrace(img, points, H):
    for p in points:
        Hp = (p[0], p[1], 1)
        Hm = np.dot(H, Hp)
        m = (int(Hm[0] / Hm[2]), int(Hm[1] / Hm[2]))
        cv2.circle(img, m, 1, (0, 0, 255), 1)
    numpy.save("H_G_M", H)
    cv2.imshow("map", img)

def angle_cos(p0, p1, p2):
    d1, d2 = p0-p1, p2-p1
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def findSquares(img,minSize = 2000,maxAngle = 1):
    """ findSquares intend to locate rectangle in the image of minimum area, minSize, and maximum angle, maxAngle, between 
    sides"""
    squares = []
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
         cnt_len = cv2.arcLength(cnt, True)
         cnt = cv2.approxPolyDP(cnt, 0.08*cnt_len, True)
         if len(cnt) == 4 and cv2.contourArea(cnt) > minSize and cv2.isContourConvex(cnt):
             cnt = cnt.reshape(-1, 2)
             max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
             if max_cos < maxAngle:
                 squares.append(cnt)
    return squares

def DetectPlaneObject(I,minSize=1000):
      """ A simple attempt to detect rectangular 
      color regions in the image"""
      HSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
      h = HSV[:,:,0].astype('uint8')
      s = HSV[:,:,1].astype('uint8')
      v = HSV[:,:,2].astype('uint8')
      
      b = I[:,:,0].astype('uint8')
      g = I[:,:,1].astype('uint8')
      r = I[:,:,2].astype('uint8')
     
      # use red channel for detection.
      s = (255*(r>230)).astype('uint8')
      iShow = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
      cv2.imshow('ColorDetection',iShow)
      squares = findSquares(s,minSize)
      return squares
  
def texturemapObjectSequence():
    """ Poor implementation of simple texturemap """
    fn = 'BookVideos/Seq3_scene.mp4'
    cap = cv2.VideoCapture(fn) 
    drawContours = True;
    
    texture = cv2.imread('images/ITULogo.jpg')
    #texture = cv2.transpose(texture)
    mTex,nTex,t = texture.shape
    
    #load Tracking data
    running, imgOrig = cap.read()
    mI,nI,t = imgOrig.shape
    
    print running 
    while(running):
        for t in range(20):
            running, imgOrig = cap.read() 
        
        if(running):
            squares = DetectPlaneObject(imgOrig)
            
            for sqr in squares:
                 #Do texturemap here!!!!
                 #TODO
                 
                 if(drawContours):                
                     for p in sqr:
                         cv2.circle(imgOrig,(int(p[0]),int(p[1])),3,(255,0,0)) 
                 
            
            if(drawContours and len(squares)>0):    
                cv2.drawContours( imgOrig, squares, -1, (0, 255, 0), 3 )

            cv2.circle(imgOrig,(100,100),10,(255,0,0))
            cv2.imshow("Detection",imgOrig)
            cv2.waitKey(1)
#showFloorTrackingData()
#showImageAndPlot()
#simpleTextureMap()
textureMapGroundFloor()
#realisticTexturemapGroundFloor()
#texturemapGridSequence()
