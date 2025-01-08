#Módulo detector Hough

import numpy as np
import cv2 as cv
import math

'''
A Segments object represent a set of lines (straight segments), stored in coords property, as an array of segments.
Each segment is a pair of 2D points.
All arrays are ndarray with main dimension of n.

Segments properties:

n: number of segments
referencePoint: 2D point, origin for Hough distance calculation

ndarray of n rows:
coords: ndarray de segmentos detectados [n,2,2], indefinido en un objeto vacío, se define una vez y no debería cambiar
deltas: (dx,dy)
lengths: segments lengths
distances: Hough distances to reference point
angles: segments angles
'''
class Segments:
    '''
    Constructor
    Sin argumentos crea un objeto vacío
    El argumento puede ser:
    - ndarray [n,1,4] con segmentos de Fast Line Detector
    - Segments object, copia coords del objecto
    '''
    def __init__(self, segs, referencePoint=None):
        '''
        Constructor
        Parameters:
        segs (np.ndarray or Segments or None): The segments to initialize the object with. 
            - If segs is a numpy ndarray, it is a segments list from FLD with shape (n, 1, 4) where n is the number of segments.
            - If segs is an instance of Segments, it will copy the segments from the given Segments object.
            - If segs is None, an empty Segments object will be created, if you want to fill it yourself.
        '''
        if(segs is None):
            # objeto vacío, sin segmentos todavía
            self.n = 0
        
        elif(isinstance(segs, np.ndarray)):
            # segmentos de FLD o de factory
            self.coords = segs.reshape([-1,2,2])
            self.n = self.coords.shape[0]

        elif(isinstance(segs, Segments)):
            # copia
            self.n = segs.n
            self.coords = segs.coords

        if(referencePoint is not None):
            self.setReferencePoint(referencePoint)

    
    def setReferencePoint(self, point):
        '''
        Set the reference point for Hough distance calculation.
        It doesn't have any other purpose.
        You must explicitly call this method to set the reference point.
        Computing distances without a reference point will raise an exception,
        because having a random referencePoint is worse.
        Parameters:
        point (tuple): The reference point as a 2D tuple.
        '''
        self.referencePoint = np.array(point, dtype=np.float32)

    def computeDeltas(self):
        '''
        Deltas are (dx,dy), second point minus first point.
        '''
        self.deltas = self.coords[:,1,:] - self.coords[:,0,:]    # Shape n,2, formato #segmento, (dx,dy), dx=x1-x0

    def tidyupDeltas(self):
        '''
        Rotates 180º if necessary to have dy >= 0, so normal vectors are pointing up.
        Useful for computing distances.
        '''
        if(not hasattr(self, 'deltas')):
            self.computeDeltas()
        self.deltas = np.where(self.deltas[:,1:]<0, -self.deltas, self.deltas)  # change sign if dy negative

    def computeLengths(self):
        '''
        Computes deltas and lengths from deltas.
        '''
        if(not hasattr(self, 'deltas')):
            self.computeDeltas()
        self.lengths = np.linalg.norm(self.deltas, axis=1)           # Shape n

    def computeAngles(self):
        '''
        Computes angles from deltas, with arctan2, so angle is in [-pi, pi].
        Results are stored in angles property.
        If deltas are normalized (dy>=0), angles are in [0, pi].
        It doesn't take arguments nor return any value.
        '''
        if(not hasattr(self, 'deltas')):
            self.computeDeltas()
        self.angles = np.arctan2(self.deltas[:,1], self.deltas[:,0])    # arctan(y/x); Shape n, angles in +-pi, 0 for horizontal

    def computeDistances(self):
        '''
        Computes distances from referencePoint to the segments, with sign.
        Not setting the referencePoint will raise an exception.
        If deltas are not tidy up, distance signs will depend on deltas.
        '''
        if(not hasattr(self, 'lengths')):
            self.computeLengths()
        self.normalUnityVectors = np.empty_like(self.deltas)
        self.normalUnityVectors[:,0] = +self.deltas[:,1] / self.lengths
        self.normalUnityVectors[:,1] = -self.deltas[:,0] / self.lengths
        linePoints = (self.coords[:,0,:] - self.referencePoint)
        self.distances = (self.normalUnityVectors * linePoints).sum(axis=1)

    def computeAnglesAndDistances(self):
        '''
        It executes it all:
        Computes angles, distances, lengths, deltas and then tidy up deltas.
        '''
        self.tidyupDeltas()
        self.computeAngles()
        self.computeDistances()

def zenithalSegmentsFactory(points, H, referencePoint=None):
    '''
    Usa puntos de un ndarray o de un objeto Segments, 
    aplica la transformación homogénea H [3,3], 
    crea un nuevo objeto Segments con la misma cantidad de segmentos, 
    pero con las coordenadas de sus puntos transformadas por H
    '''
    if(isinstance(points, Segments)):
        points = points.coords
    
    # ndarray de puntos
    points = points.reshape((-1,2)) # Shape 2n,2

    # Conversión a coordenadas homogéneas
    ones = np.ones((points.shape[0],1), np.float32)            # Shape 2n,1
    homogeneousPoints = np.concatenate((points, ones), axis=1) # Shape 2n,3

    # Proyección
    projectedHomogeneousPoints = homogeneousPoints @ H.transpose() # Shape 2n,3

    # Normalización y reducción dimensional
    projectedSegments = (projectedHomogeneousPoints[:,:2]/projectedHomogeneousPoints[:,2:]).reshape((-1,2,2)) # Shape n,2,2

    return Segments(projectedSegments, referencePoint=referencePoint)

'''
Annotator is a class to draw segments over an image.
'''
class SegmentsAnnotator:
    @staticmethod
    def colorMapBGR(intensity):
        
        '''
        Devuelve un color en formato BGR a partir de una intensidad en [0.0,1.0)
        Mapea colores continuos y cíclicos: el color de 0,999 es adyacente al de 0.0.
        Implementa una paleta de 765 colores.
        '''
        decimal, integer = math.modf((intensity % 1)*3)
        range = int(integer)
        scalar = int(decimal*255) # 0..254
        match range:
            case 0: color = (255-scalar, scalar, 0) # 0+ azul a verde
            case 1: color = (0, 255-scalar, scalar) # intermedios entre verde y rojo
            case _: color = (scalar, 0, 255-scalar) # 0- azul a rojo

        return color
        scalar = int(intensity%1*765)
        return (383-abs(scalar-383), abs(scalar-255), abs(scalar-510))

    @staticmethod
    def colorMapYMC(intensity):
        '''
        '''
        decimal, integer = math.modf((intensity % 1)*3)
        range = int(integer)
        scalar = int(decimal*255) # 0..254
        match range:
            case 0: color = (scalar, 255-scalar, 255)
            case 1: color = (255, scalar, 255-scalar)
            case _: color = (255-scalar, 255, scalar)

        return color

    @staticmethod
    def colorMapGray(intensity):
        scalar = int(intensity*256)
        return (scalar, scalar, scalar)

    def __init__(self, color=(0,255,0), thickness=1, withPoints = False, offset=(0,0), scale=1.0, colorMap = colorMapBGR):
        '''
        Constructor
        Sets default values for drawing segments over an image.
        color: segments annotation color, when there is no intensities
        thickness: grosor de los segmentos
        withPoints: dibujar puntos extremos
        offset: desplazamiento de los segmentos
        scale: factor de escala
        colorMap: intentisies to color mapping function
        '''
        self.color = color
        self.thickness = thickness
        self.withPoints = withPoints
        self.offset = np.array(offset, np.float32)
        self.scale = scale
        self.colorMap = colorMap

    def drawSegments(self, image, segments, intensities=None, message=None, color=None, thickness=None, colorMap=None, withPoints = None, offset=None, scale=None):
        '''
        Annotates segments over an image.
        If intensities is provided, color is ignored and colorMap is used.
        If message is provided, it is written on the bottom-left corner.
        Other arguments let the user override the default values.

        Args:
            image (ndarray): Image to annotate.
            segments (Segments): Segments to annotate.
            intensities (ndarray): Intensities to map to colors, in the range [0..1).  Same size as segments.
            message (str): Message to write on the image.
            color (tuple): Color to use for annotation, used if intensities are not provided.
            thickness (int): Thickness of the lines.
            colorMap (function): Function to map intensities to colors.
            withPoints (bool): Whether to draw the endpoints of the segments.
            offset (tuple): Offset to apply to the segments.
            scale (float): Scale factor to apply to the segments

        '''
        # default values from object
        if offset is None:
            offset = self.offset
        if scale is None:
            scale = self.scale
        if color is None:
            color = self.color
        if thickness is None:
            thickness = self.thickness
        if withPoints is None:
            withPoints = self.withPoints

        if(isinstance(segments, Segments)):
            coords = segments.coords
        elif(isinstance(segments, np.ndarray)):
            # ndarray, may be only one segment
            coords = segments.reshape((-1,2,2))
        else:
            coords = np.array(segments).reshape((-1,2,2))
        
        for index, segments in enumerate(coords):
            if(intensities is not None):
                color = self.colorMap(intensities[index])

            pointA = (offset + scale * segments[0]).astype(int)
            pointB = (offset + scale * segments[1]).astype(int)
            cv.line(image, pointA, pointB, color=color, thickness=thickness)
            if(withPoints):
                cv.circle(image, pointA, 2, color)
                cv.circle(image, pointB, 2, color)

        if(message is not None):
            textLines = message.split('\n')
            n = len(textLines)
            for i, textLine in enumerate(textLines):
                cv.putText(image, textLine, (10, image.shape[0]-5-20*(n-i-1)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

class HoughSpace:
    def __init__(self, angleBins=11, maxDistanceAsLanes=4, laneBins=4, laneWidth=210):
        '''
        Constructor
        - angleBins: number of bins for angles from 0 to pi, p/2 is vertical
        - maxDistanceAsLanes: far edge for distances bins, in lanes
        - laneBins: number of bins in one lane
        - laneWidth: width of a lane in pixels
        '''
        self.angleBins = angleBins
        self.maxDistanceAsLanes = maxDistanceAsLanes
        self.laneBins = laneBins
        self.laneWidth = laneWidth

        self.centralAngleBin = angleBins // 2
        self.angle2index = angleBins / math.pi
        self.distance2index = laneBins / laneWidth
        self.maxDistanceInPixels = laneWidth * maxDistanceAsLanes
        self.centralDistanceBin = math.ceil(laneBins * maxDistanceAsLanes)
        self.distanceBins = 2 * self.centralDistanceBin + 1

    def assign2Bins(self, segments):
        self.angleIndices = np.clip((segments.angles * self.angle2index).astype(int), 0, self.angleBins-1)
        self.distanceIndices = np.clip((segments.distances * self.distance2index + self.centralDistanceBin).astype(int), 0, self.distanceBins-1)

    def getIndicesFromBin(self, angleBin, distanceBin):
        '''
        Get the indices of elements in the bin that match the given angle and distance bins.
        '''
        return np.argwhere(np.logical_and(self.angleIndices == angleBin, self.distanceIndices == distanceBin)).reshape(-1)

    def computeVotes(self, votes):
        self.houghSpace = np.zeros((self.angleBins, self.distanceBins), np.float32)
        np.add.at(self.houghSpace, (self.angleIndices, self.distanceIndices), votes)

        self.maxLoc = np.unravel_index(np.argmax(self.houghSpace), self.houghSpace.shape)
        self.maxVal = self.houghSpace[self.maxLoc]

        self.angleHistogram = np.sum(self.houghSpace, axis=1)
        self.laneZone = np.sum(self.houghSpace[self.centralAngleBin-1:self.centralAngleBin+2], axis=0)[np.newaxis,:]

        return self.houghSpace

    def getVisualization(self, scale = None, showMax=False):
        '''
        Produce and return a colormapped image of the histogram produced in computeVotes(),
        optionally highliting the peak if showMax is True,
        '''
        houghSpaceGray = (self.houghSpace * 255/self.maxVal) if self.maxVal>0 else self.houghSpace
        houghSpaceColor = cv.applyColorMap(houghSpaceGray.astype(np.uint8), cv.COLORMAP_DEEPGREEN)
        if(showMax):
            houghSpaceColor[self.maxLoc[0], self.maxLoc[1]] = (0,0,255)
        if(scale is not None):
            houghSpaceColor = cv.resize(houghSpaceColor, None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)

        return houghSpaceColor

    def pasteVisualization(self, image, borderColor=(0,128,255), scale = None, showMax=False):
        houghSpaceColor = self.getVisualization(scale, showMax)
        
        ih, iw, _ = image.shape
        hh, hw, _ = houghSpaceColor.shape
        image[-hh-1:-1, -hw-1:-1] = houghSpaceColor
        if(borderColor is not None):
            cv.rectangle(image, (iw-hw-2, ih-hh-2), (iw-1, ih-1), borderColor)
            cv.line(image, (iw-hw//2-1, ih-hh-2), (iw-hw//2-1, ih-1), borderColor)
            cv.line(image, (iw-hw-2, ih-hh//2-1), (iw- 1, ih-hh//2-1), borderColor)

        if(scale is not None):
            angleHistogram = cv.resize(self.angleHistogram, None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)
            laneZone = cv.resize(self.laneZone, None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)
        else:
            angleHistogram = self.angleHistogram
            laneZone = self.laneZone
        
        image[-hh-1:-1, -hw-10:-hw-10+angleHistogram.shape[1]] = angleHistogram[:,:,np.newaxis]
        image[-hh-3-laneZone.shape[0]:-hh-3, -hw-1:-1] = laneZone[:,:,np.newaxis]

        return houghSpaceColor