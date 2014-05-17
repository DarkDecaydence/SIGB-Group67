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
import math
from SIGBTools2 import *
#from SIGBTools2 import camera

VideoFile = "CubeProjections.avi"

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
 
'''
- places a texture in a cube face
- image: current image; faceVertices: points of the cube face; 
        projMethod: current projection method; imageUrl: url for image to texture
- texturing: 
1) project cube points
2) Homography from texture to cube points
3) warp image with homography
4) enhance color of face with a mask (overlay from warp) and bit operations (AND then OR)
- returns: image with texture in cube face
'''
def TextureFace(image, faceVertices, projMethod, imageUrl):
    face = []
    faceVertices = faceVertices[0]
    #project points before drawing texture
    for point in faceVertices:
        #print "point", point
        point = [point[0], point[1], point[2], 1]
        point = np.dot(projMethod, point)
        face.append((int(point[0]/point[2]), int(point[1]/point[2])))
    face = np.array(face)
    #print face
    cv2.drawContours(image, [face], -1, (255, 0, 255), 1)

    ######## - to find out which points are which
    ##if(imageUrl == "Images/Right.jpg"):#      B  G  R
    ##    cv2.circle(image, tuple(face[0]), 5, (0,0,255), -1)
    ##    cv2.circle(image, tuple(face[1]), 5, (0,255,0), -1)
    ##    cv2.circle(image, tuple(face[2]), 5, (255,0,0), -1)
    ##    cv2.circle(image, tuple(face[3]), 5, (255,255,255), -1)
    ########
    faceImage = cv2.imread(imageUrl)
    faceImageH, faceImageW,_ = faceImage.shape
    #get image points to calculate H
    faceImagePoints = [[0,0],[0,faceImageW],[faceImageH,0], [faceImageW,faceImageH]]
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

'''
- calculates the normals of cube faces
- image: current image; face: cube face; projMethod: method being used to project points -> P2_method1|P2_method2
- projects and draws a vector from the center of the cube along the normal
- returns: normal of the face
'''
def calculateNormals(image, face, projMethod):
    face = face[0]
    #re-arrange points to have them in counterclockwise order
    face = np.array([face[2],face[3], face[1], face[0]])
    #to find center of face
    x, y, z = zip(*face)
    faceNormal = GetFaceNormal(face)
    #point 1 = center of face
    point1 = ( int((max(x)+min(x) )/2), int((max(y)+min(y))/2), int((max(z)+min(z))/2), 1)
    point1_projected = np.dot(projMethod, point1)
    point1_projected = ( int(point1_projected[0]/point1_projected[2]) , int(point1_projected[1]/point1_projected[2]) )
    #point 2 = along normal
    point2 = ( int( faceNormal[0]+point1[0]) , int(faceNormal[1]+point1[1]), int(faceNormal[2]+point1[2]), 1)
    point2_projected = np.dot(projMethod, point2)
    point2_projected = ( int(point2_projected[0]/point2_projected[2]) , int(point2_projected[1]/point2_projected[2]) )
    cv2.line(image, tuple(point1_projected), tuple(point2_projected), (255,0,255), 2)
    cv2.circle(image, point2_projected, 3, (255,0,255), -1)
    
    return faceNormal

'''
- calculates the dotProduct between V and N
- V: center of the camera vector
- N: normal of the cube face
- returns: boolean -> true=show, false=hide
'''
def removeHiddenFacesDotProduct(faceNormal, cameraMatrix):
    global cameraCenter
    cam = Camera(cameraMatrix)
    cameraCenter = cam.center()
    V = np.array(cameraCenter)*-1
    dot = np.dot(faceNormal, V)
    # V*N > 0 -> Hide
    if(dot > 0):
        return False
    return True

'''
- calculates angle between v1 and v2
- returns angle in radians
'''
def getAngle(v1, v2):
    product = np.dot(v1, v2)
    norms = ( np.linalg.norm(v1) *np.linalg.norm(v2))
    return np.arccos(product / norms)
  
'''
- calculates the angle between V and N
- V: center of the camera vector
- N: normal of the cube face
- returns: boolean -> true=show, false=hide
'''
def removeHiddenFacesAngle(faceNormal, cameraMatrix):

    global cameraCenter
    cam = Camera(cameraMatrix)
    ##
    # SIGBTools2.Camera.center() returns the center of the camera  
    ##
    cameraCenter = cam.center()
    V = np.array(cameraCenter)
    angle = getAngle(faceNormal, V)
    ##
    # angles returned in radians
    ##
    if(angle > np.pi/2):
        return False
    return True

def ShadeFace(image,points,faceCorner_Normals, camera):
    global shadeRes
    shadeRes=10
    videoHeight, videoWidth, vd = array(image).shape
    #................................
    points_Proj=camera.project(toHomogenious(points))
    points_Proj1 = np.array([[int(points_Proj[0,0]),int(points_Proj[1,0])],[int(points_Proj[0,1]),\
        int(points_Proj[1,1])],[int(points_Proj[0,3]),int(points_Proj[1,3])],[int(points_Proj[0,2]),int(points_Proj[1,2])]])
    #................................
    square = np.array([[0, 0], [shadeRes-1, 0], [shadeRes-1, shadeRes-1], [0, shadeRes-1]])
    #................................
    H = estimateHomography(square, points_Proj1)
    #................................
    Mr0,Mg0,Mb0=CalculateShadeMatrix(image,shadeRes,points,faceCorner_Normals, camera)
    # HINT
    # type(Mr0): <type 'numpy.ndarray'>
    # Mr0.shape: (shadeRes, shadeRes)
    #................................
    Mr = cv2.warpPerspective(Mr0, H, (videoWidth, videoHeight),flags=cv2.INTER_LINEAR)
    Mg = cv2.warpPerspective(Mg0, H, (videoWidth, videoHeight),flags=cv2.INTER_LINEAR)
    Mb = cv2.warpPerspective(Mb0, H, (videoWidth, videoHeight),flags=cv2.INTER_LINEAR)
    #................................
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    [r,g,b]=cv2.split(image)
    #................................
    whiteMask = np.copy(r)
    whiteMask[:,:]=[0]
    points_Proj2=[]
    points_Proj2.append([int(points_Proj[0,0]),int(points_Proj[1,0])])
    points_Proj2.append([int(points_Proj[0,1]),int(points_Proj[1,1])])
    points_Proj2.append([int(points_Proj[0,3]),int(points_Proj[1,3])])
    points_Proj2.append([int(points_Proj[0,2]),int(points_Proj[1,2])])
    cv2.fillConvexPoly(whiteMask,array(points_Proj2),(255,255,255))
    #................................
    r[nonzero(whiteMask>0)]=map(lambda x: max(min(x,255),0),r[nonzero(whiteMask>0)]*Mr[nonzero(whiteMask>0)])
    g[nonzero(whiteMask>0)]=map(lambda x: max(min(x,255),0),g[nonzero(whiteMask>0)]*Mg[nonzero(whiteMask>0)])
    b[nonzero(whiteMask>0)]=map(lambda x: max(min(x,255),0),b[nonzero(whiteMask>0)]*Mb[nonzero(whiteMask>0)])
    #................................
    image=cv2.merge((r,g,b))
    image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def CalculateShadeMatrix(image,shadeRes,points,faceCorner_Normals, camera):
    #Ambient light IA=[IaR,IaG,IaB]
    IA = np.matrix([5.0, 5.0, 5.0]).T
    #Point light IA=[IpR,IpG,IpB]
    IP = np.matrix([2.0, 2.0, 2.0]).T
    #Light Source Attenuation
    fatt = 2
    #Material properties: e.g., Ka=[kaR; kaG; kaB]
    ka=np.matrix([0.2, 0.2, 0.2]).T
    kd= np.matrix([0.3, 0.3, 0.3]).T
    ks=np.matrix([0.7, 0.7, 0.7]).T
    alpha = 100
    
    mat = [None, None, None]
    lightPos = np.array(camera.c.flat)
    camPos = np.array(camera.c.flat)
    
    for ch in range(3):
        mat[ch] = np.zeros((shadeRes, shadeRes))
        for x in range(shadeRes):
            for y in range(shadeRes):
                viewPoint = np.array(BilinearInterpo(shadeRes, x, y, points, True))
                lightDirec = viewPoint - lightPos
                lightDirec /= np.linalg.norm(lightDirec)
                viewDirec = viewPoint - camPos
                viewDirec /= np.linalg.norm(viewDirec)
                normal = np.array(BilinearInterpo(shadeRes, x, y, faceCorner_Normals, True))
                reflect = lightDirec - 2 * dot(lightDirec, normal) * normal
                mat[ch][x, y] = (IA[ch] * kd[ch] * max(dot(normal, lightDirec), 0)) + (IP[ch] * ks[ch] * (max(dot(reflect, viewDirec) ** fatt, 0)))
    
    return (mat[0], mat[1], mat[2])


def update(img, record=False):
    global cubeDefined, topFace, downFace, leftFace, rightFace, upFace, P2_method1, P2_method2, currentMethod
    global drawTopFace, drawDownFace, drawLeftFace, drawRightFace, drawUpFace
    global writer
    
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
            if ProjectPattern:
                cv2.circle(image, currentViewPoints[0], 5, (255,0,0), -1)
                cv2.circle(image, currentViewPoints[1], 5, (255,0,0), -1)
                cv2.circle(image, currentViewPoints[2], 5, (255,0,0), -1)
                cv2.circle(image, currentViewPoints[3], 5, (255,0,0), -1)

            firstView = cv2.imread(videoSequence)
            firstView = cv2.imread("Images/01.png")
            fVGray=cv2.cvtColor(firstView, cv2.COLOR_BGR2GRAY)  
            foundFV,cornersFV=cv2.findChessboardCorners(fVGray, (9,6))
            fVPoints = [] 
            fVPoints = getOuterPoints(cornersFV)
            H_fv_cv = np.array(estimateHomography(fVPoints, currentViewPoints))
            ##
            #
            #Changed order
            # Before: For some reason the methods where calculated after inside if(ProjectPattern)
            # After: Method is defined before everything and put in currentMethod (global)
            #
            ##
            ''' <006> Here Define the cameraMatrix P=K[R|t] of the current frame'''
            H_fv_pattern = np.load("numpyData/H_ff_pattern.npy")
            H_cv_pattern = np.dot( H_fv_pattern, H_fv_cv)
            pattern_size=(9,6)
            pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
            pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
            pattern_points *= chessSquare_size
            
            if methodNo == 1 :
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
                currentMethod = P2_method1
            elif methodNo == 2 :    
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
                currentMethod = P2_method2
            if ShowText:
                ''' <011> Here show the distance between the camera origin and the world origin in the image''' 
                
                if methodNo == 1:
                    K = P2_method1[:, :3]
                    T = P2_method1[:, 3:]
                else:
                    K = P2_method2[:, :3]
                    T = P2_method2[:, 3:]
                camPos = dot(-(matrix(K).I), T)
                camDist = math.sqrt(camPos[0]**2 + camPos[1]**2 + camPos[2]**2)
                cv2.putText(image,str("frame: " + str(frameNumber)) + "\n origin distance: " + str(camDist), (20,30),cv2.FONT_HERSHEY_PLAIN,2, (255, 255, 255))#Draw the text

            if TextureMap:
                ###
                #
                # Changed structure
                # Before: <010> and <012>, both in separate if(cubeDefined)
                # After: <010> and <012> included inside a same if(cubeDefined)
                #*****tried <012> before <010> but since normals where drawn before the textures they got ocluded
                #
                ### 

                if(cubeDefined):
                    ''' <010> Here Do he texture mapping and draw the texture on the faces of the cube'''
                    TopFaceCornerNormals,RightFaceCornerNormals,LeftFaceCornerNormals,UpFaceCornerNormals,DownFaceCornerNormals = \
                        CalculateFaceCornerNormals(topFace[0],rightFace[0],leftFace[0],upFace[0],downFace[0])
                    cam = Camera(P2_method1 if methodNo == 1 else P2_method2)
                    cam.factor()
                    cam.center()
                    
                    if(drawTopFace):
                        image = TextureFace(image, topFace, currentMethod, "Images/Top.jpg")
                        image=ShadeFace(image, topFace[0].T, TopFaceCornerNormals, cam)
                    if(drawRightFace):
                        image = TextureFace(image, rightFace, currentMethod, "Images/Right.jpg")
                        image=ShadeFace(image, rightFace[0].T, RightFaceCornerNormals, cam)
                    if(drawLeftFace):
                        image = TextureFace(image, leftFace, currentMethod, "Images/Left.jpg")
                        image=ShadeFace(image, leftFace[0].T, LeftFaceCornerNormals, cam)
                    if(drawUpFace):
                        image = TextureFace(image, upFace, currentMethod, "Images/Up.jpg")
                        image=ShadeFace(image, upFace[0].T, UpFaceCornerNormals, cam)
                    if(drawDownFace):
                        image = TextureFace(image, downFace, currentMethod, "Images/Down.jpg")
                        image=ShadeFace(image, downFace[0].T, DownFaceCornerNormals, cam)

                    ''' <012>  calculate the normal vectors of the cube faces and draw these normal vectors on the center of each face'''
                    topFaceNormal = calculateNormals(image, topFace, currentMethod)
                    downFaceNormal = calculateNormals(image, downFace, currentMethod)
                    leftFaceNormal = calculateNormals(image, leftFace, currentMethod)
                    rightFaceNormal = calculateNormals(image, rightFace, currentMethod)
                    upFaceNormal = calculateNormals(image, upFace, currentMethod)
                    ''' <013> Here Remove the hidden faces'''
                    #draw___Face = boolean to "hide" faces
                    ##drawTopFace = removeHiddenFacesDotProduct(topFaceNormal, currentMethod)
                    ##drawDownFace = removeHiddenFacesDotProduct(downFaceNormal, currentMethod)
                    ##drawLeftFace = removeHiddenFacesDotProduct(leftFaceNormal, currentMethod)
                    ##drawRightFace = removeHiddenFacesDotProduct(rightFaceNormal, currentMethod)
                    ##drawUpFace = removeHiddenFacesDotProduct(upFaceNormal, currentMethod)

                    drawTopFace = removeHiddenFacesAngle(topFaceNormal, currentMethod)
                    drawDownFace = removeHiddenFacesAngle(downFaceNormal, currentMethod)
                    drawLeftFace = removeHiddenFacesAngle(leftFaceNormal, currentMethod)
                    drawRightFace = removeHiddenFacesAngle(rightFaceNormal, currentMethod)
                    drawUpFace = removeHiddenFacesAngle(upFaceNormal, currentMethod)

            if ProjectPattern:                  
                ''' <007> Here Test the camera matrix of the current view by projecting the pattern points'''
                #draw chessboard pattern points using currentMethod
                ###
                #
                # Changed order
                # Before: here the projection method was calculated
                # After: projection method calculated before, 
                #        here only projection of pattern_points with currentMethod
                #
                ### 
                
                for point in pattern_points:
                    point = [point[0], point[1], point[2], 1]
                    point = np.dot(currentMethod, point)
                    center = (int(point[0]/point[2]), int(point[1]/point[2]))
                    #cv2.circle(image, center, 5, (0,255,0), -1)
                ''' <008> Here Draw the world coordinate system in the image'''            
                # topLeft,topRight,bottomLeft,bottomRight
                topLeft, topRight, bottomLeft, bottomRight = getWorldCoordinateSystem(currentCorners)
                #draw world coordinate system axis in top left of chessboard pattern
                cv2.line(image, topLeft, topRight, (255,0,0), 2)
                cv2.line(image, topLeft, bottomLeft, (0,255,0), 2)
                cv2.line(image, topLeft, bottomRight, (0,0,255), 2)

                cv2.circle(image, topRight, 5, (255,0,0), -1)
                cv2.circle(image, bottomLeft, 5, (0,255,0), -1)
                cv2.circle(image, bottomRight, 5, (0,0,255), -1)
                    
            if WireFrame:                      
                ''' <009> Here Project the box into the current camera image and draw the box edges'''
                ###
                #
                # The cube is no more projected here. The new projection is made in TextureFace(). 
                # The contours are also drawn in TextureFace()
                # Changed cube points, now it's more centered
                #
                ### 
                #using this cube because the given cube is difficult to understand
                tempCube = np.float32([[2,2,-4],#0
                                       [2,6,-4],#1
                                       [6,6,-4],#2
                                       [6,2,-4],#3
                                       
                                       [2,2,0],#4 
                                       [2,6,0],#5 
                                       [6,6,0],#6 
                                       [6,2,0]])#7
                tempContour = []
                topFace = []
                downFace = []
                leftFace = []
                rightFace = [] 
                upFace = []
                topFace.append(np.array([ tempCube[0], tempCube[1], tempCube[3], tempCube[2] ]))
                downFace.append(np.array([ tempCube[6], tempCube[2], tempCube[5], tempCube[1] ]))
                leftFace.append(np.array([ tempCube[4], tempCube[5], tempCube[0], tempCube[1] ]))
                rightFace.append(np.array([ tempCube[3], tempCube[2], tempCube[7], tempCube[6] ]))
                upFace.append(np.array([ tempCube[4], tempCube[0], tempCube[7], tempCube[3] ]))
                cubeDefined = True
                    
    cv2.namedWindow('grid')
    cv2.imshow('grid', cv2.pyrDown(image))
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
    
    global writer
    writer = cv2.VideoWriter('Cube_Projection.avi', cv.CV_FOURCC('D','I','V','3'), 5.0, (W,H), True)
    
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
            global methodNo
            if methodNo == 1:     
                methodNo = 2
                currentMethod = P2_method2                
            else:
                methodNo = 1
                currentMethod = P2_method1
            update(OriginalImage)   
            
                
        if inputKey == ord('s') or inputKey == ord('S'):
            name='Saved Images/Frame_' + str(frameNumber)+'.png' 
            cv2.imwrite(name,result)
        
        if inputKey == ord('r') or inputKey == ord('R'):
            record = not record
            print "recording..." if record else "stopped recording"
           
        if (speed>0):
            update(image, record)
            image, isSequenceOK = getImageSequence(capture,speed)

    writer.release()


def firstPart():
    ''' <002> Here Define the camera matrix of the first view image (01_daniel.png) recorded by the cameraCalibrate2'''
    firstFrame = cv2.imread(videoSequence)
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
    firstFrame = cv2.imread(videoSequence)
    pattern = cv2.imread("images/CalibrationPattern.jpg")
    imgGrayPattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)  
    _,cornersPattern = cv2.findChessboardCorners(imgGrayPattern, (9,6))
    patternP = getOuterPoints(cornersPattern)
    imgGrayFrame = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)  
    _, cornersFrame = cv2.findChessboardCorners(imgGrayFrame, (9,6))
    frame = [] 
    frame = getOuterPoints(cornersFrame)
        
    H = estimateHomography(patternP, frame)
    np.save("numpyData/H_ff_pattern.npy", H)
    #print "H saved"


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
global methodNo
global writer
###
#
#instead of having IF's every time we use method1|method2 ---> currentMethod holds P2_method1 or P2method_2
#
###
global currentMethod
global topFace,downFace,leftFace, rightFace,upFace
global drawTopFace, drawDownFace, drawLeftFace, drawRightFace, drawUpFace
global cameraCenter
    
ProcessFrame=True
Undistorting=True
WireFrame=True
ShowText=True
TextureMap=True
ProjectPattern=True
#debug=True
Hdefined=False
cubeDefined=False
methodNo = 2

tempSpeed=1
frameNumber=0
chessSquare_size=2
cameraCenter = [0,0,0]       
       
drawTopFace = False
drawDownFace = False
drawLeftFace = False
drawRightFace = False
drawUpFace = False

P2_method1 = np.zeros((3, 4))
P2_method1[:, :3] = np.identity(3)
P2_method2 = np.zeros((3, 4))
P2_method2[:, :3] = np.identity(3)

'''-------defining the cube------'''
     
box = getCubePoints([4, 2.5, 3], 1,chessSquare_size)            


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
#calibrateCamera(fileName="GridVideos/grid2.mp4")
''' <001> Here Load the numpy data files saved by the cameraCalibrate2''' 

cameraMatrix = np.load("numpyData/camera_matrix.npy")
chessSquareSize = np.load("numpyData/chessSquare_size.npy")
distortionCoeff = np.load("numpyData/distortionCoefficient.npy")
imgPointsFirst = np.load("numpyData/img_points_first.npy")
imgPoints = np.load("numpyData/img_points.npy")
objPoints = np.load("numpyData/obj_points.npy")
rotationVectors = np.load("numpyData/rotatioVectors.npy")
translationVectors = np.load("numpyData/translationVectors.npy")

videoSequence = "images/CalibrationFrame.png"

firstPart()
secondPart()

run(5,"GridVideos/grid2.mp4")

#RecordVideoFromCamera()
