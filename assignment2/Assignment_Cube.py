'''
Created on March 20, 2014

@author: Diako Mardanbegi (dima@itu.dk)
'''
from numpy import *
import numpy as np
from pylab import *
from scipy import linalg
import cv2
import cv2.cv as cv
from SIGBTools2 import *

VideoFile = "Pattern.avi"

def DrawLines(img, points):
    for i in range(1, 17):                
         x1 = points[0, i - 1]
         y1 = points[1, i - 1]
         x2 = points[0, i]
         y2 = points[1, i]
         cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0),5) 
    return img



def update(img):
    image=copy(img)


    if Undistorting:  #Use previous stored camera matrix and distortion coefficient to undistort the image
        ''' <004> Here Undistoret the image''' 
        image = cv2.undistort(image, cameraMatrix, distortion)
        
    if (ProcessFrame): 

        ''' <005> Here Find the Chess pattern in the current frame'''
        patternFound=True          
    
        if patternFound ==True:        
            
            ''' <006> Here Define the cameraMatrix P=K[R|t] of the current frame'''

        
            if ShowText:
                ''' <011> Here show the distance between the camera origin and the world origin in the image'''                    
                
                cv2.putText(image,str("frame:" + str(frameNumber)), (20,10),cv2.FONT_HERSHEY_PLAIN,1, (255, 255, 255))#Draw the text

            ''' <008> Here Draw the world coordinate system in the image'''            
            
            if TextureMap:

                ''' <010> Here Do he texture mapping and draw the texture on the faces of the cube'''

                ''' <012>  calculate the normal vectors of the cube faces and draw these normal vectors on the center of each face'''

                ''' <013> Here Remove the hidden faces'''  


            if ProjectPattern:                  
                ''' <007> Here Test the camera matrix of the current view by projecting the pattern points''' 
            
        

            if WireFrame:                      
                ''' <009> Here Project the box into the current camera image and draw the box edges''' 
    
    cv2.namedWindow('Web cam')
    cv2.imshow('Web cam', image)  
    global result
    result=copy(image)

def getImageSequence(capture, fastForward):
    '''Load the video sequence (fileName) and proceeds, fastForward number of frames.'''
    global frameNumber
   
    for t in range(fastForward):
        isSequenceOK, originalImage = capture.read()  # Get the first frames
        frameNumber = frameNumber+1
    return originalImage, isSequenceOK


def printUsage():
    print "Q or ESC: Stop"
    print "SPACE: Pause"     
    print "p: turning the processing on/off "  
    print 'u: undistorting the image'
    print 'g: project the pattern using the camera matrix (test)'
    print 'x: your key!' 
       
    print 'the following keys will be used in the next assignment'      
    print 'i: show info'
    print 't: texture map'
    print 's: save frame'

    
   
def run(speed,video): 
    
    '''MAIN Method to load the image sequence and handle user inputs'''   

    #--------------------------------video
    capture = cv2.VideoCapture(video)


    image, isSequenceOK = getImageSequence(capture,speed)       

    if(isSequenceOK):
        update(image)
        printUsage()

    while(isSequenceOK):
        OriginalImage=copy(image)
     
        
        inputKey = cv2.waitKey(1)
        
        if inputKey == 32:#  stop by SPACE key
            update(OriginalImage)
            if speed==0:     
                speed = tempSpeed;
            else:
                tempSpeed=speed
                speed = 0;                    
            
        if (inputKey == 27) or (inputKey == ord('q')):#  break by ECS key
            break    
                
        if inputKey == ord('p') or inputKey == ord('P'):
            global ProcessFrame
            if ProcessFrame:     
                ProcessFrame = False;
                
            else:
                ProcessFrame = True;
            update(OriginalImage)
            
        if inputKey == ord('u') or inputKey == ord('U'):
            global Undistorting
            if Undistorting:     
                Undistorting = False;
            else:
                Undistorting = True;
            update(OriginalImage)     
        if inputKey == ord('w') or inputKey == ord('W'):
            global WireFrame
            if WireFrame:     
                WireFrame = False;
                
            else:
                WireFrame = True;
            update(OriginalImage)

        if inputKey == ord('i') or inputKey == ord('I'):
            global ShowText
            if ShowText:     
                ShowText = False;
                
            else:
                ShowText = True;
            update(OriginalImage)
            
        if inputKey == ord('t') or inputKey == ord('T'):
            global TextureMap
            if TextureMap:     
                TextureMap = False;
                
            else:
                TextureMap = True;
            update(OriginalImage)
            
        if inputKey == ord('g') or inputKey == ord('G'):
            global ProjectPattern
            if ProjectPattern:     
                ProjectPattern = False;
                
            else:
                ProjectPattern = True;
            update(OriginalImage)   
             
        if inputKey == ord('x') or inputKey == ord('X'):
            global debug
            if debug:     
                debug = False;                
            else:
                debug = True;
            update(OriginalImage)   
            
                
        if inputKey == ord('s') or inputKey == ord('S'):
            name='Saved Images/Frame_' + str(frameNumber)+'.png' 
            cv2.imwrite(name,result)
           
        if (speed>0):
            update(image)
            image, isSequenceOK = getImageSequence(capture,speed)          








'''-------------------MAIN BODY--------------------------------------------------------------------'''
'''--------------------------------------------------------------------------------------------------------------'''




'''-------variables------'''
global cameraMatrix    
global distortionCoefficient
global homographyPoints
global calibrationPoints
global calibrationCamera
global chessSquare_size
    
ProcessFrame=False
Undistorting=False   
WireFrame=False
ShowText=True
TextureMap=True
ProjectPattern=False
debug=True

tempSpeed=1
frameNumber=0
chessSquare_size=2
       
       

'''-------defining the cube------'''
     
box = getCubePoints([4, 2.5, 0], 1,chessSquare_size)            


i = array([ [0,0,0,0],[1,1,1,1] ,[2,2,2,2]  ])  # indices for the first dim 
j = array([ [0,3,2,1],[0,3,2,1] ,[0,3,2,1]  ])  # indices for the second dim            
TopFace = box[i,j]


i = array([ [0,0,0,0],[1,1,1,1] ,[2,2,2,2]  ])  # indices for the first dim 
j = array([ [3,8,7,2],[3,8,7,2] ,[3,8,7,2]  ])  # indices for the second dim            
RightFace = box[i,j]


i = array([ [0,0,0,0],[1,1,1,1] ,[2,2,2,2]  ])  # indices for the first dim 
j = array([ [5,0,1,6],[5,0,1,6] ,[5,0,1,6]  ])  # indices for the second dim            
LeftFace = box[i,j]

  
i = array([ [0,0,0,0],[1,1,1,1] ,[2,2,2,2]  ])  # indices for the first dim 
j = array([ [5,8,3,0], [5,8,3,0] , [5,8,3,0] ])  # indices for the second dim            
UpFace = box[i,j]


i = array([ [0,0,0,0],[1,1,1,1] ,[2,2,2,2]  ])  # indices for the first dim 
j = array([ [1,2,7,6], [1,2,7,6], [1,2,7,6] ])  # indices for the second dim            
DownFace = box[i,j]



'''----------------------------------------'''
'''----------------------------------------'''



''' <000> Here Call the calibrateCamera from the SIGBTools to calibrate the camera and saving the data''' 
#calibrateCamera()
''' <001> Here Load the numpy data files saved by the cameraCalibrate2''' 
translationVectors = np.load('numpyData/translationVectors.npy')
rotationVectors = np.load('numpyData/rotatioVectors.npy')
objPoints = np.load('numpyData/obj_points.npy')
imgPoints = np.load('numpyData/img_points.npy')
distortion = np.load('numpyData/distortionCoefficient.npy')
cameraMatrix = np.load('numpyData/camera_matrix.npy')

''' <002> Here Define the camera matrix of the first view image (01.png) recorded by the cameraCalibrate2'''
firstFrame = cv2.imread("01.png")
pattern_size = (9,6)
obj_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
obj_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)

camera = np.zeros((3, 3))
h, w, _ = firstFrame.shape

found,corners = cv2.findChessboardCorners(firstFrame, pattern_size)
img_points = corners.reshape(-1, 2)


if found != 0:
    _, camera, _, rotation, translation  = cv2.calibrateCamera([obj_points], [img_points], (w, h) ,camera, np.zeros(4) ,flags = 0)
#constructing the projection matrix R|t
rtMatrix = np.zeros((3, 4))
rotMatrix = cv2.Rodrigues(rotation[0])[0]
rtMatrix[:, :3] = rotMatrix
rtMatrix[:, 3:] = translation[0]

''' <003> Here Load the first view image (01.png) and find the chess pattern and store the 4 corners of the pattern needed for homography estimation''' 
for p in obj_points:
    imgH = np.dot(camera, np.dot(rtMatrix, [p[0], p[1], p[2], 1]))
    imgP = [imgH[0] / imgH[2], imgH[1] / imgH[2]]
    cv2.circle(firstFrame, (int(imgP[0]), int(imgP[1])), 3, (0, 0, 255), 4)
cv2.imshow("projection", firstFrame)
cv2.waitKey(0)

run(1, VideoFile)