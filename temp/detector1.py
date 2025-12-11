#Módulo detector Hough

import numpy as np
import cv2 as cv
from math import modf

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
    def __init__(self, segs):
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

        self.setReferencePoint((0, 0))

    
    def setReferencePoint(self, point):
        self.referencePoint = np.array(point, dtype=np.float32)

    def computeDeltas(self):
        self.deltas = self.coords[:,1,:] - self.coords[:,0,:]    # Shape n,2, formato #segmento, (dx,dy), dx=x1-x0

    def normalizeDeltas(self):
        if(not hasattr(self, 'deltas')):
            self.computeDeltas()
        self.deltas = np.where(self.deltas[:,1:]<0, -self.deltas, self.deltas)  # change sign if dy negative
        #self.deltas[self.deltas[:,1]<0] = -self.deltas[...]

    def computeLengths(self):
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
        if(not hasattr(self, 'lengths')):
            self.computeLengths()
        self.normalUnityVectors = np.empty_like(self.deltas)
        self.normalUnityVectors[:,0] = +self.deltas[:,1] / self.lengths
        self.normalUnityVectors[:,1] = -self.deltas[:,0] / self.lengths
        linePoints = (self.coords[:,0,:] - self.referencePoint)
        self.distances = (self.normalUnityVectors * linePoints).sum(axis=1)
        #self.distances = np.sum(self.normalUnityVectors * (self.coords[:,0,:] - self.referencePoint), axis=1)
        #self.distances = np.sum((self.deltas[:,::-1]/self.lengths[:,None]) * (self.coords[:,0,:] - self.referencePoint), axis=1)

    def computeAnglesAndDistances(self):
        self.normalizeDeltas()
        self.computeAngles()
        self.computeDistances()

def zenithalSegmentsFactory(points, H):
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

    return Segments(projectedSegments)

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
        decimal, integer = modf((intensity % 1)*3)
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
        decimal, integer = modf((intensity % 1)*3)
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

class Bins:
    '''
    HoughSpace is a 2D histogram, its variables are angles and distances.
    They are discretized in bins.
    Bins class configure the bins in its constructor, so they can be used many times in the main loop.
    It stores the bins values during each loop, to be consumed by a HoughSpace object to produce the actual histogram.
    As each histogram is weigthed, many histograms can be produced with different weigths and from the same discretized variables.
    That's why Bins are separated from HoughSpace.

    Use:
    Configure Bins in construction, all of them have a default value and can be changed providing the value to the constructor or after construction:
    - binSizes tell how many bins for angles and distances, in that order
    - maxDistance is the far edge for distances bins
    - histoRange has the span (min, max) of both variables, (0-pi, +-maxDistance)
    - angleIntervals has the boundaries of each angle bin, evenly spaced by default
    - distanceIntervals has the boundaries of each distance bin, evenly spaced by default
    - binRanges needed for histogram2D in HoughSpace

    In the loop:
    - assignToBins(segments) discretizes the variables, and Bins is ready to create HoughSpace
    - makeHoughSpace(votes) make a new HoughSpace object linked to Bins, to analize and get binCoords
    - getIndicesFromBins(binCoords) provides the indices to the segments assigned to a particular bin

    '''
    def __init__(self, binsSizes = (11,41), maxDistance = 1500, verbose=False):
        '''
        Constructor
        - binsSizes: number of bins (angles, distances)
        - maxDistance: far edge for distances bins
        - verbose: print configuration
        '''
        self.histoRange = np.array([[0.0,np.pi],[-maxDistance, maxDistance]], np.float32)
        self.binsSizes = binsSizes
        self.makeBinsRanges()
        if(verbose):
            print('Axis: (angles, distances) (row, column)')
            print('Bins sizes', self.binsSizes)
            print('Histogram ranges', self.histoRange)
            #print('Distances intervals', self.distanceIntervals)

    def makeBinsRanges(self):
        '''
        Bin ranges are used in np.digitize() in assignToBins().
        They are constant during the object lifetime.
        Produced by default in construction, must be recalculated if the user changes histoRange.
        '''
        self.angleIntervals    = np.linspace(self.histoRange[0,0], self.histoRange[0,1], self.binsSizes[0]+1)
        self.distanceIntervals = np.linspace(self.histoRange[1,0], self.histoRange[1,1], self.binsSizes[1]+1)
        self.binsRanges = np.array([[0.0, self.binsSizes[0]-1],[0, self.binsSizes[1]-1]], np.float32)

    def assignToBins(self, segments):
        '''
        Produce inner arrays, same size of segments, with the correspondent bin number.
        Used later in computeVotes().
        For 20 bins, indices span from 0 to 19.  Indices -1 and 20 belong to values out of range.
        '''
        self.angleBinsIndices    = np.digitize(segments.angles, self.angleIntervals) - 1
        self.distanceBinsIndices = np.digitize(segments.distances, self.distanceIntervals) - 1

    def getIndicesFromBin(self, binCoords):
        '''
        Get the indices of elements in the bin that match the given bin coordinates.

        Args:
            binCoords (tuple): A tuple containing the coordinates of the bin (angle, distance).

        Returns:
            tuple: A tuple of arrays, each containing the indices of the elements 
                    that match the given bin coordinates.
        '''
        return np.argwhere(np.logical_and(self.angleBinsIndices == binCoords[0], self.distanceBinsIndices == binCoords[1])).reshape(-1)
    
    def makeHoughSpace(self, votes, windowName):
        '''
        HoughSpace factory.
        It creates a HoughSpace object from this Bins object, with the given votes.
        '''
        return HoughSpace(self, votes, windowName)


'''
Produce the voting space (Hough parametric space) from segments.
It uses distances and angles from segments to address a bin, then adds a weighted vote to it.

- Constructor: it links to a provided Bins object; if votes is provided it runs computeVotes()
- computeVotes: makes a weigthed 2D histogram called houghSpace, locates the maximum and its value
- show: make a colored image of the houghSpace for visualization; optionally highliting the maximum and showing it on a window

How to use:
- contruct an instance assigning a Bins object, or use the factory provided by Bins: Bins.makeHoughSpace()
- computeVotes() if not already done it in construction
- analize houghSpace
- show() houghSpace
'''
class HoughSpace:
    """
    A class to represent the Hough Space for line detection using the Hough Transform.

    Attributes
    ----------
    bins : object
        An object containing the bins information for angles and distances.
    windowName : str
        The name of the window to display the Hough Space image.
    houghSpace : ndarray
        The 2D histogram representing the Hough Space.
    maxLoc : tuple
        The location of the maximum value in the Hough Space.
    maxVal : float
        The maximum value in the Hough Space.

    Methods
    -------
    computeVotes(votes):
        Computes a 2D histogram weighted with the given votes.
    show(showMax=False):
        Produces and returns a colormapped image of the histogram, optionally highlighting the peak.
    """
    def __init__(self, bins, votes, windowName):
        self.bins = bins
        self.windowName = windowName
        if(votes is not None):
            self.computeVotes(votes)

    '''
    Compute 2D histrogram weigthed with given votes.
    Votes is an array the same size of segments, with weigths; for example, segments lentgh.
    The resulting histogram is returned and remains accesible in self.hougSpace .
    If votes is not provided, an unweigthed histogram is computed.
    '''
    def computeVotes(self, votes):
        # histogram2d: rows: angles; columns: distances
        self.houghSpace , xedges, yedges = np.histogram2d(
            self.bins.angleBinsIndices, self.bins.distanceBinsIndices, weights=votes, 
            bins=self.bins.binsSizes, range=self.bins.binsRanges
        )
        self.maxLoc = np.unravel_index(np.argmax(self.houghSpace), self.houghSpace.shape)
        self.maxVal = self.houghSpace[self.maxLoc]
        return self.houghSpace

    def getVisualization(self, showMax=False):
        '''
        Produce and return a colormapped image of the histogram produced in computeVotes(),
        optionally highliting the peak if showMax is True,
        '''
        houghSpaceGray = (self.houghSpace * 255/self.maxVal) if self.maxVal>0 else self.houghSpace
        houghSpaceColor = cv.applyColorMap(houghSpaceGray.astype(np.uint8), cv.COLORMAP_DEEPGREEN)
        if(showMax):
            houghSpaceColor[self.maxLoc[0], self.maxLoc[1]] = (0,0,255)

        return houghSpaceColor

    def show(self, showMax=False):
        houghSpaceColor = self.getVisualization(showMax)
        if(self.windowName):
            cv.imshow(self.windowName, houghSpaceColor)
        return houghSpaceColor
    
    def pasteVisualization(self, image, borderColor=(0,128,255), scale = None, showMax=False):
        houghSpaceColor = self.getVisualization(showMax)
        if(scale is not None):
            houghSpaceColor = cv.resize(houghSpaceColor, None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)
        
        ih, iw, _ = image.shape
        hh, hw, _ = houghSpaceColor.shape
        image[-hh-1:-1, -hw-1:-1] = houghSpaceColor
        if(borderColor is not None):
            #image[-hh-2:, -hw-2:] = borderColor
            cv.rectangle(image, (iw-hw-2, ih-hh-2), (iw-1, ih-1), borderColor)
            cv.line(image, (iw-hw//2-1, ih-hh-2), (iw-hw//2-1, ih-1), borderColor)
            cv.line(image, (iw-hw-2, ih-hh//2-1), (iw- 1, ih-hh//2-1), borderColor)
        return houghSpaceColor


