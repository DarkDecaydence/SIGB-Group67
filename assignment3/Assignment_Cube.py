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

def getOuterPoints(Corners):
    """
        gets outer points of chess board
    """
    rcCorners = np.array(Corners)
    topLeft = rcCorners[0,0]
    topLeft = (int(topLeft[0]), int(topLeft[1]))
    topRight = rcCorners[8, -1]
    topRight = (int(topRight[0]), int(topRight[1]))
    bottomLeft = rcCorners[45,-1]
    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
    bottomRight = rcCorners[-1,-1]
    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
    return topLeft,topRight,bottomLeft,bottomRight

def getWorldCoordinateSystem(corners):
    rcCorners = np.array(corners)
    topLeft = rcCorners[0,0]
    topLeft = (int(topLeft[0]), int(topLeft[1]))
    topRight = rcCorners[1, -1]
    topRight = (int(topRight[0]), int(topRight[1]))
    bottomLeft = rcCorners[9,-1]
    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
    bottomRight = rcCorners[10,-1]
    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
    return topLeft,topRight,bottomLeft,bottomRight
 
def TextureFace(image, face, cameraMatrix, imageUrl):
    faceImage = cv2.imread(imageUrl)
    faceImageH, faceImageW,_ = faceImage.shape
    #get image points to calculate H
    faceImagePoints = [[0,0],[0,faceImageW],[faceImageH,0], [faceImageW,faceImageH]]
    face = face[0]
    H = estimateHomography(faceImagePoints, face)
    h, w, _ = image.shape
    #overlay is going to work as a mask to enhance the warpPerspective
    overlay = cv2.warpPerspective(faceImage, H, (w, h))
    image = cv2.addWeighted(image, 1.0, overlay, 1.0, 1.0)
    #getting the face in white and everything in black
    thresh, threshI = cv2.threshold(overlay, 10, 255, cv2.THRESH_BINARY)
    #white background and face in black
    mask = 255 - threshI
    # face in black AND image
    temp = cv2.bitwise_and(mask, image)
    # image with face in black OR overlay (face with color)
    image = cv2.bitwise_or(temp, overlay)
    return image

def calculateNormals(image, face):
    face = face[0]
    #re-arrange points to have them in counterclockwise order
    face = np.array([face[0],face[1], face[3], face[2]])
    #to find center of face
    x, y = zip(*face)
    faceNormal = GetFaceNormal(face)
    faceNormal = faceNormal*25    
    #point 1 = center of face
    point1 = tuple( ( int((max(x)+min(x) )/2), int((max(y)+min(y))/2) ))
    #point 2 = normal
    point2 = tuple(( int( faceNormal[2]+point1[0]) , int(faceNormal[2]+point1[1]) ))
    cv2.line(image, point1, point2, (255,255,255), 3)
    return faceNormal

def removeHiddenFaces(faceNormal):
    # V*N > 0 -> Hide
    V = np.array([0,0,-1]).T
    dot = np.dot(faceNormal, V)
    #print dot
    if(dot > 0):
        return False
    return True

def update(img, writer=None, record=False):
    global cubeDefined, topFace, downFace, leftFace, rightFace, upFace, P2_method1, P2_method2
    global drawTopFace, drawDownFace, drawLeftFace, drawRightFace, drawUpFace

    
    image=copy(img)
    if Undistorting:  #Use previous stored camera matrix and distortion coefficient to undistort the image
        ''' <004> Here Undistoret the image''' 
        image=cv2.undistort(image, cameraMatrix, distortionCoeff )
    if (ProcessFrame): 
        ''' <005> Here Find the Chess pattern in the current frame'''
        # H_firstFrame_pattern
        imgGray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        patternFound,currentCorners=cv2.findChessboardCorners(imgGray, (9,6))

        if patternFound ==True:        
            currentViewPoints = [] 
            currentViewPoints = getOuterPoints(currentCorners)
            cv2.circle(image, currentViewPoints[0], 5, (255,0,0), -1)
            cv2.circle(image, currentViewPoints[1], 5, (255,0,0), -1)
            cv2.circle(image, currentViewPoints[2], 5, (255,0,0), -1)
            cv2.circle(image, currentViewPoints[3], 5, (255,0,0), -1)

            firstView = cv2.imread("Images/01.png")
            fVGray=cv2.cvtColor(firstView, cv2.COLOR_BGR2GRAY)  
            foundFV,cornersFV=cv2.findChessboardCorners(fVGray, (9,6))
            fVPoints = [] 
            fVPoints = getOuterPoints(cornersFV)
            H_fv_cv = np.array(estimateHomography(fVPoints, currentViewPoints))
            
            ''' <006> Here Define the cameraMatrix P=K[R|t] of the current frame'''
            H_fv_pattern = np.load("numpyData/H_ff_pattern.npy")
            H_cv_pattern = np.dot( H_fv_pattern, H_fv_cv)
           
            if ShowText:
                ''' <011> Here show the distance between the camera origin and the world origin in the image'''                    
                
                cv2.putText(image,str("frame:" + str(frameNumber)), (20,10),cv2.FONT_HERSHEY_PLAIN,1, (255, 255, 255))#Draw the text

            if TextureMap:

                ''' <010> Here Do he texture mapping and draw the texture on the faces of the cube'''
                if(cubeDefined):
                    if(method == 1):
                        if(drawTopFace):
                            image = TextureFace(image, topFace, P2_method1, "Images/Top.jpg")
                        if(drawDownFace):
                            image = TextureFace(image, downFace, P2_method1, "Images/Down.jpg")
                        if(drawLeftFace):
                            image = TextureFace(image, leftFace, P2_method1, "Images/Left.jpg")
                        if(drawRightFace):
                            image = TextureFace(image, rightFace, P2_method1, "Images/Right.jpg")
                        if(drawUpFace):
                            image = TextureFace(image, upFace, P2_method1, "Images/Up.jpg")
                    else:
                        if(drawTopFace):
                            image = TextureFace(image, topFace, P2_method2, "Images/Top.jpg")
                        if(drawDownFace):
                            image = TextureFace(image, downFace, P2_method2, "Images/Down.jpg")
                        if(drawLeftFace):
                            image = TextureFace(image, leftFace, P2_method2, "Images/Left.jpg")
                        if(drawRightFace):
                            image = TextureFace(image, rightFace, P2_method2, "Images/Right.jpg")
                        if(drawUpFace):
                            image = TextureFace(image, upFace, P2_method2, "Images/Up.jpg")

                ''' <012>  calculate the normal vectors of the cube faces and draw these normal vectors on the center of each face'''
                if(cubeDefined):
                    topFaceNormal = calculateNormals(image, topFace)
                    downFaceNormal = calculateNormals(image, downFace)
                    leftFaceNormal = calculateNormals(image, leftFace)
                    rightFaceNormal = calculateNormals(image, rightFace)
                    upFaceNormal = calculateNormals(image, upFace)
                    ''' <013> Here Remove the hidden faces'''
                    #draw___Face = boolean to "hide" faces
                    drawTopFace = removeHiddenFaces(topFaceNormal)
                    drawDownFace = removeHiddenFaces(downFaceNormal)
                    drawLeftFace = removeHiddenFaces(leftFaceNormal)
                    drawRightFace = removeHiddenFaces(rightFaceNormal)
                    drawUpFace = removeHiddenFaces(upFaceNormal)
                                        
            if ProjectPattern:                  
                ''' <007> Here Test the camera matrix of the current view by projecting the pattern points''' 
                pattern_size=(9,6)
                pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
                pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
                pattern_points *= chessSquare_size
                
                if method == 1 :
                    #break down K[R|t]
                    camera_matrix = dot(cameraMatrix, Kt)
                    K, _ = rq(camera_matrix[:,:3])
                    # make diagonal of K positive
                    T = diag(sign(diag(K)))
                    K = dot(K,T)
                    
                    K_inv = np.linalg.inv(K)
                    P2_method1 = np.zeros((3,4))
                    P2_method1 = np.dot(H_cv_pattern, camera_matrix)
                    Kt_temp = np.dot(K_inv, P2_method1[:,:3])
                    Kt_temp = array([Kt_temp[:,0], Kt_temp[:,1], cross(Kt_temp[:,0], Kt_temp[:,1])]).T
                    P2_method1[:,:3] = np.dot(K, Kt_temp)
                    #draw chessboard pattern points using method1
                    for point in pattern_points:
                        point = array([point[0], point[1], point[2], 1])
                        point = np.dot(P2_method1, point)
                        center = (int(point[0]/point[2]), int(point[1]/point[2]))
                        cv2.circle(image, center, 2, (0,0,255), -1)
                    
                elif method == 2 :    
                    obj_points = [pattern_points]
                    obj_points.append(pattern_points)
                    obj_points = np.array(obj_points,np.float64).T
                    obj_points=obj_points[:,:,0].T
                    #get new rotation and traslation vecs
                    found,rvecs_new,tvecs_new = GetObjectPos(obj_points, currentCorners, cameraMatrix, distortionCoeff)
                    #global P2_method2 
                    P2_method2 = np.zeros((3, 4))
                    if found == True:
                        rot, _ = cv2.Rodrigues(rvecs_new)
                        P2_method2[:, :3] = rot
                        P2_method2[:, 3:] = tvecs_new
                        P2_method2 = dot(cameraMatrix, P2_method2)
                        #draw chessboard pattern points using method2
                    for point in pattern_points:
                        point = [point[0], point[1], point[2], 1]
                        point = np.dot(P2_method2, point)
                        center = (int(point[0]/point[2]), int(point[1]/point[2]))
                        cv2.circle(image, center, 5, (0,255,0), -1)
                ''' <008> Here Draw the world coordinate system in the image'''            
                # topLeft,topRight,bottomLeft,bottomRight
                topLeft, topRight, bottomLeft, bottomRight = getWorldCoordinateSystem(currentCorners)
                
                cv2.line(image, topLeft, topRight, (255,0,0), 2)
                cv2.line(image, topLeft, bottomLeft, (0,255,0), 2)
                cv2.line(image, topLeft, bottomRight, (0,0,255), 2)

                cv2.circle(image, topRight, 5, (255,0,0), -1)
                cv2.circle(image, bottomLeft, 5, (0,255,0), -1)
                cv2.circle(image, bottomRight, 5, (0,0,255), -1)
                    
            if WireFrame:                      
                ''' <009> Here Project the box into the current camera image and draw the box edges''' 
                #using this cube because the given cube is difficult to understand
                tempCube = np.float32([[0,0,-4],
                                       [0,4,-4],
                                       [4,4,-4],
                                       [4,0,-4],
                                       
                                       [0,0,0], 
                                       [0,4,0], 
                                       [4,4,0], 
                                       [4,0,0]])
                tempContour = []
                #project cube points with present method
                for point in tempCube:
                    point = [point[0], point[1], point[2], 1]
                    if method == 1 :
                        point = np.dot(P2_method1, point)
                    else :
                        point = np.dot(P2_method2, point)
                    tempContour.append((int(point[0]/point[2]), int(point[1]/point[2])))
                tempContour = np.array(tempContour)
                #global topFace,downFace,leftFace, rightFace,upFace
                topFace = []
                downFace = []
                leftFace = []
                rightFace = [] 
                upFace = []
                #break points into faces
                #weird indexes to have them in order
                topFace.append(np.array([ tempContour[0], tempContour[1], tempContour[3], tempContour[2] ]))
                downFace.append(np.array([ tempContour[6], tempContour[2], tempContour[5], tempContour[1] ]))
                leftFace.append(np.array([ tempContour[4], tempContour[5], tempContour[0], tempContour[1] ]))
                rightFace.append(np.array([ tempContour[3], tempContour[2], tempContour[7], tempContour[6] ]))
                upFace.append(np.array([ tempContour[4], tempContour[0], tempContour[7], tempContour[3] ]))
                cubeDefined = True
                #draw cube faces
                cv2.drawContours(image, topFace, -1, (255, 0, 255), 1)
                cv2.drawContours(image, downFace, -1, (255, 0, 255), 1)
                cv2.drawContours(image, leftFace, -1, (255, 0, 255), 1)
                cv2.drawContours(image, rightFace, -1, (255, 0, 255), 1)
                cv2.drawContours(image, upFace, -1, (255, 0, 255), 1)
                    
                    
    cv2.namedWindow('Web cam')
    cv2.imshow('Web cam', image)
    if record:
        writer.write(image)
    #global result
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
    print 'r: record video'

    
   
def run(speed,video): 
    
    '''MAIN Method to load the image sequence and handle user inputs'''   

    #--------------------------------video
    capture = cv2.VideoCapture(video)


    image, isSequenceOK = getImageSequence(capture,speed)
    H,W,_ = image.shape
    record = False

    if(isSequenceOK):
        update(image)
        printUsage()
    
    writer = cv2.VideoWriter('CubeProjections.avi', cv.CV_FOURCC('D','I','V','3'), 5.0, (W,H), True)
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
            global method
            if method == 1:     
                method = 2;                
            else:
                method = 1;
            update(OriginalImage)   
            
                
        if inputKey == ord('s') or inputKey == ord('S'):
            name='Saved Images/Frame_' + str(frameNumber)+'.png' 
            cv2.imwrite(name,result)
        
        if inputKey == ord('r') or inputKey == ord('R'):
            record = not record
            print "recording..." if record else "stopped recording"
           
        if (speed>0):
            update(image, writer, record)
            image, isSequenceOK = getImageSequence(capture,speed)

    writer.release()


def firstPart():
    ''' <002> Here Define the camera matrix of the first view image (01_daniel.png) recorded by the cameraCalibrate2'''
    firstFrame = cv2.imread("Images/01.png")
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
    global Kt
    Kt = np.zeros((3, 4))
    rotMatrix = cv2.Rodrigues(rotation[0])[0]
    Kt[:, :3] = rotMatrix
    Kt[:, 3:] = translation[0]
    ''' <003> Here Load the first view image (01_daniel.png) and find the chess pattern and store the 4 corners of the pattern needed for homography estimation''' 
    for p in obj_points:
        imgH = np.dot(camera, np.dot(Kt, [p[0], p[1], p[2], 1]))
        imgP = [imgH[0] / imgH[2], imgH[1] / imgH[2]]
        cv2.circle(firstFrame, (int(imgP[0]), int(imgP[1])), 3, (0, 0, 255), 4)
    #cv2.imshow("projection", firstFrame)
    #cv2.waitKey(0)


def secondPart():
    firstFrame = cv2.imread("Images/01.png")
    pattern = cv2.imread("Images/CalibrationPattern.jpg")
    imgGrayPattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)  
    _,cornersPattern = cv2.findChessboardCorners(imgGrayPattern, (9,6))
    patternP = getOuterPoints(cornersPattern)
    imgGrayFrame = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)  
    _, cornersFrame = cv2.findChessboardCorners(imgGrayFrame, (9,6))
    frame = [] 
    frame = getOuterPoints(cornersFrame)
        
    H = estimateHomography(patternP, frame)
    np.save("numpyData/H_ff_pattern.npy", H)
    print "H saved"


'''-------------------MAIN BODY--------------------------------------------------------------------'''
'''--------------------------------------------------------------------------------------------------------------'''




'''-------variables------'''
global cameraMatrix
global P2_method1    
global P2_method2
global distortionCoeff
global homographyPoints
global calibrationPoints
global calibrationCamera
global chessSquare_size
global cubeDefined
global Kt
global imgPointsFirst
global method
global topFace,downFace,leftFace, rightFace,upFace
global drawTopFace, drawDownFace, drawLeftFace, drawRightFace, drawUpFace
    
ProcessFrame=True
Undistorting=False   
WireFrame=True
ShowText=True
TextureMap=True
ProjectPattern=True
#debug=True
cubeDefined=False
method = 2

tempSpeed=1
frameNumber=0
chessSquare_size=2
       
       
drawTopFace = False
drawDownFace = False
drawLeftFace = False
drawRightFace = False
drawUpFace = False

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

cameraMatrix = np.load("numpyData/camera_matrix.npy")
chessSquareSize = np.load("numpyData/chessSquare_size.npy")
distortionCoeff = np.load("numpyData/distortionCoefficient.npy")
imgPointsFirst = np.load("numpyData/img_points_first.npy")
imgPoints = np.load("numpyData/img_points.npy")
objPoints = np.load("numpyData/obj_points.npy")
rotationVectors = np.load("numpyData/rotatioVectors.npy")
translationVectors = np.load("numpyData/translationVectors.npy")

firstPart()
secondPart()

run(1,0)

#RecordVideoFromCamera()
