'''
Homography UI module
User interface for homography correspondences.

The user uses the keyboard to run the UI command, then click on the image to set the element place.
There are three commands, setting three horizontal lines.
HORIZON and TOP are used to compute the homography.
LIMIT sets the far edge of the zenithal transform.
'''

import numpy as np
import cv2 as cv

class Command:
    HORIZON = 'h'
    LIMIT = 'l'
    TOP = 't'
    EMPTY = ''

def mouseListener(event,x,y,flags,self):
    #print(x,y,self)
    if(event == cv.EVENT_LBUTTONDOWN):
        if(self.command == Command.HORIZON):
            self.horizon = int(y)
            self.calculateRoi()
        elif(self.command == Command.TOP):
            self.top = int(y)
            self.calculateRoi()
        elif(self.command == Command.LIMIT):
            self.limit = int(y)
            self.calculateRoi()

class Hui:
    def __init__(self, windowName, imShowSize, afterRoi=None) -> None:
        self.command = ''
        self.windowName = windowName
        self.imShowSize = imShowSize
        frameHeight = imShowSize[1]
        self.horizon = int(frameHeight * 0.32)
        self.top     = int(frameHeight * 0.75)
        self.limit   = int(frameHeight * 0.45)
        self.zenithalSquareSide = 500
        self.H = None
        self.afterRoi = afterRoi

        cv.namedWindow(self.windowName)
        cv.setMouseCallback(self.windowName, mouseListener, self)


    def drawHorizontalLine(self, im, y, text='', selected=False):
        y = int(y)
        cv.putText(im, text, (0, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255,255,255))
        if selected:
            cv.line(im, (0, y), (im.shape[1], y), (255,255,255), 2)
        else:
            cv.line(im, (0, y), (im.shape[1], y), (128,128,128))

        return im

    def anotate(self, im):
        anotatedIm = im.copy()
        self.drawHorizontalLine(anotatedIm, self.horizon, 'horizon', self.command == Command.HORIZON)
        self.drawHorizontalLine(anotatedIm, self.top, 'top', self.command == Command.TOP)
        self.drawHorizontalLine(anotatedIm, self.limit, 'limit', self.command == Command.LIMIT)
        cv.line(anotatedIm, self.roiPoly[3,0], self.fuga, (128,128,128))
        cv.line(anotatedIm, self.roiPoly[2,0], self.fuga, (128,128,128))
        cv.polylines(anotatedIm, [self.roiPoly], True, (0,255,0))
        cv.imshow(self.windowName, anotatedIm)


    # ROI
    # Cuadrilátero, comenzando por vértice superior izquierdo, en sentido horario
    # Si se proporciona una homografía, la usa para obtener horizonte y tope
    def calculateRoi(self, H = None):
        '''
        H is the homography transforming the trapezoid into a square.
        H2 is the modify homography to get the zenithal view, a wider view than the prior.
        '''
        if(H is not None):
            # Homografía suministrada, se recalculan horizonte y tope
            self.H = H.astype(np.float32)
            puntosClave = np.array(((0,-1,0),(0,0,1)), np.float32)
            #print('puntosClave', puntosClave)
            puntosEnPerspectiva = np.matmul(np.linalg.inv(self.H.astype(np.float32)), puntosClave.T).T
            #print('puntosEnPerspectiva', puntosEnPerspectiva)
            self.horizon = int(puntosEnPerspectiva[0,1]/puntosEnPerspectiva[0,2])
            self.top      = int(puntosEnPerspectiva[1,1]/puntosEnPerspectiva[1,2])
            if(self.limit<self.horizon):
                self.limit = self.horizon
            #print('horizonte y tope', self.horizon, self.top)

        medio = int(self.imShowSize[0]/2)
        self.fuga = (medio, self.horizon)
        xProyectado = int(medio*(self.imShowSize[1]-self.top)/(self.imShowSize[1]-self.horizon))
        roiTrapezoidVertices = np.array([
            [xProyectado,self.top],
            [self.imShowSize[0]-xProyectado,self.top],
            [self.imShowSize[0],self.imShowSize[1]],
            [0,self.imShowSize[1]]
        ], np.int32)
        self.roiPoly = roiTrapezoidVertices.reshape((-1,1,2))

        zenithalSquareVertices = np.array([
            [0,0],
            [self.zenithalSquareSide,0],
            [self.zenithalSquareSide,self.zenithalSquareSide],
            [0,self.zenithalSquareSide]
        ], np.float32)

        if(H is None):
            self.H = cv.getPerspectiveTransform(roiTrapezoidVertices.astype(np.float32), zenithalSquareVertices)
            #print('Homografía:\n', self.H, '\n')
        
        # H2 for cenital view
        Pleft = self.H @ np.array([0.0, self.limit, 1.0], np.float32).reshape(-1, 1)
        Pleft = (Pleft/Pleft[2]).reshape(-1)
        #print('Pleft', Pleft.shape, Pleft)
        translation = np.array([
            [1.0, 0.0, -Pleft[0]],
            [0.0, 1.0, -Pleft[1]],
            [0.0, 0.0,       1.0]
        ], np.float32)
        self.H2 = translation @ self.H

        Pright = self.H2 @ np.array([self.imShowSize[0], self.limit, 1.0], np.float32).reshape(-1, 1)
        Pright = (Pright/Pright[2]).reshape(-1)
        self.zenithalSize = np.array((Pright[0], -Pleft[1]+self.zenithalSquareSide),np.int32)
        #print('self.zenithalSize', type(self.zenithalSize), self.zenithalSize)

        if(self.afterRoi):
            self.afterRoi(self)

        #print(self.horizon, self.top, self.limit)


    def toggleCommand(self, command):
        self.command = '' if self.command == command else command

    def processKey(self, key):
        match chr(key):
            case 'h':
                # Ajustar horizonte
                self.toggleCommand(Command.HORIZON)
            case 'l':
                # Ajustar tope de homografía
                self.toggleCommand(Command.LIMIT)
            case 't':
                # Ajustar tope de homografía
                self.toggleCommand(Command.TOP)

        return key