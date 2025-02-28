"""
Módulo detector Hough

Este módulo se ocupa del procesamiento de la imagen para la detección de segmentos,
su anotación y su representación en el espacio de Hough.

Contiene 3 clases:

- Segments: representa un conjunto de segmentos, con sus deltas, longitudes, distancias y ángulos.
- SegmentsAnnotator: clase para dibujar segmentos sobre una imagen.
- HoughSpace: clase para calcular el espacio de Hough y sus histogramas.

"""

import numpy as np
import cv2 as cv
import math

class Segments:
    """
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
    """

    def __init__(self, segs: Segments|np.ndarray|None, referencePoint:tuple|None=None):
        """
        Constructor
        Arguments:
        segs (np.ndarray or Segments or None): The segments to initialize the object with. 
            - If segs is a numpy ndarray, it is a segments list from FLD with shape (n, 1, 4) where n is the number of segments.
            - If segs is an instance of Segments, it will copy the segments from the given Segments object.
            - If segs is None, an empty Segments object will be created, if you want to fill it yourself.
        referencePoint (tuple): The reference point for Hough distance calculation.
        """

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

    
    def setReferencePoint(self, point:tuple):
        """
        Set the reference point for Hough distance calculation.
        It doesn't have any other purpose.
        You must explicitly call this method to set the reference point.
        Computing distances without a reference point will raise an exception,
        because having a random referencePoint is worse.
        Arguments:
            point (tuple): The reference point as a 2D tuple.
        """

        self.referencePoint = np.array(point, dtype=np.float32)

    def computeDeltas(self):
        """
        Deltas are (dx,dy), second point minus first point.
        """

        self.deltas = self.coords[:,1,:] - self.coords[:,0,:]    # Shape n,2, formato #segmento, (dx,dy), dx=x1-x0

    def tidyupDeltas(self):
        """
        Rotates 180º if necessary to have dy >= 0, so normal vectors are pointing down (positive y points down).
        Useful for computing distances.
        """

        if(not hasattr(self, 'deltas')):
            self.computeDeltas()
        self.deltas = np.where(self.deltas[:,1:]<0, -self.deltas, self.deltas)  # change sign if dy negative

    def computeLengths(self):
        """
        Computes deltas and lengths from deltas.
        """

        if(not hasattr(self, 'deltas')):
            self.computeDeltas()
        self.lengths = np.linalg.norm(self.deltas, axis=1)           # Shape n

    def computeAngles(self):
        """
        Computes angles from deltas, with arctan2, so angle is in [-pi, pi].
        Results are stored in angles property.
        If deltas are normalized (dy>=0), angles are in [0, pi].
        It doesn't take arguments nor return any value.
        """

        if(not hasattr(self, 'deltas')):
            self.computeDeltas()
        self.angles = np.arctan2(self.deltas[:,1], self.deltas[:,0])    # arctan(y/x); Shape n, angles in +-pi, 0 for horizontal

    def computeDistances(self):
        """
        Computes distances from referencePoint to the segments, with sign.
        Not setting the referencePoint will raise an exception.
        If deltas are not tidy up, distance signs will depend on deltas.
        """

        if(not hasattr(self, 'lengths')):
            self.computeLengths()
        self.normalUnityVectors = np.empty_like(self.deltas)
        self.normalUnityVectors[:,0] = +self.deltas[:,1] / self.lengths
        self.normalUnityVectors[:,1] = -self.deltas[:,0] / self.lengths
        linePoints = (self.coords[:,0,:] - self.referencePoint)
        self.distances = (self.normalUnityVectors * linePoints).sum(axis=1)

    def computeAnglesAndDistances(self):
        """
        It executes it all:
        Computes angles, distances, lengths, deltas and then tidy up deltas.
        """

        self.tidyupDeltas()
        self.computeAngles()
        self.computeDistances()

def projectSegments(segments:Segments|np.ndarray|tuple, H:np.ndarray, inverse:bool=False, segmentsShape:bool=True, printFlag:bool=False)->np.ndarray:
    """
    Projects the given segments with the given homography H of shape (3,3).

        P' = H * P
    
    Segments are pairs of points.  
    To projects points instead of segments (an odd number of points, like only one)
    use segmentsShape=False.

    Arguments:
    - segments: Segments object or ndarray of segments.
    - H: homography matrix.
    - inverse: if True, H is considered to be the inverse of the projection.
    - segmentsShape: if True, the result is reshaped to the original segments shape.
    - printFlag: if True, prints intermediate results.

    Returns:
    - ndarray of projected segments.
    """

    if(isinstance(segments, Segments)):
        segments = segments.coords
    elif(not isinstance(segments, np.ndarray)):
        segments = np.array(segments)
    
    # Points ndarray
    points = segments.reshape((-1,2)) # Shape 2n,2
    if(printFlag):
        print(f'points: {points}')

    # Homogeneous coordinates conversion
    ones = np.ones((points.shape[0],1), np.float32)            # Shape 2n,1
    homogeneousPoints = np.concatenate((points, ones), axis=1) # Shape 2n,3
    if(printFlag):
        print(f'homogeneousPoints: {homogeneousPoints}')

    # H must be transpose, because H is thought to multiply a column vector, and these are row vectors.
    # Unless inverse is True: H is homogeneous, so transpose and inverse accomplish the same purpose.
    if(not inverse):
        H = H.transpose()
    else:
        H = np.linalg.inv(H).transpose()

    if(printFlag):
        print(f'H: {H}')

    # Projection
    projectedHomogeneousPoints = homogeneousPoints @ H # Shape 2n,3

    if(printFlag):
        print(f'projectedHomogeneousPoints: {projectedHomogeneousPoints}')


    # Back to vector space
    projectedSegments = (projectedHomogeneousPoints[:,:2]/projectedHomogeneousPoints[:,2:])
    if(segmentsShape):
        projectedSegments.reshape((-1,2,2)) # Shape n,2,2

    if(printFlag):
        print(f'projectedSegments: {projectedSegments}')

    return projectedSegments

class SegmentsAnnotator:
    """
    Annotator is a class to draw segments over an image.
    """

    @staticmethod
    def colorMapBGR(intensity:float)->tuple:
        """
        Return an BGR color from an intensity in the range [0.0,1.0),
        with bright 255 - primary and secondary colors.
        It maps colors in a cyclic pattern: color for 0,999 is neighbour to color for 0.0.
        Adopts a pallete of 765 colors.

        Arguments:
        - intensity: intensity in the range [0.0,1.0).

        Returns:
        - tuple with BGR color.
        """

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
    def colorMapYMC(intensity:float)->tuple:
        """
        Return an BGR color from an intensity in the range [0.0,1.0),
        with bright 510 - secondary and tertiary colors.
        It maps colors in a cyclic pattern: color for 0,999 is neighbour to color for 0.0.
        Adopts a pallete of 765 colors.

        Arguments:
        - intensity: intensity in the range [0.0,1.0).

        Returns:
        - tuple with BGR color.
        """

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
        """Gray scale color mapping"""
        scalar = int(intensity*256)
        return (scalar, scalar, scalar)

    def __init__(self, color:tuple=(0,0,255), thickness:float=1, withPoints:bool = False, offset:tuple=(0,0), scale:float=1.0, colorMap:function=colorMapBGR):
        """
        Constructor
        Sets default values for drawing segments over an image.

        Arguments:
        - color: color to use for annotation.
        - thickness: thickness of the lines.
        - withPoints: whether to draw the endpoints of the segments.
        - offset: offset to apply to the segments.
        - scale: scale factor to apply to the segments.
        - colorMap: function to map intensities to colors.
        """

        self.color = color
        self.thickness = thickness
        self.withPoints = withPoints
        self.offset = np.array(offset, np.float32)
        self.scale = scale
        self.colorMap = colorMap

    def drawSegments(self, image, segments, intensities=None, message=None, color=None, thickness=None, colorMap=None, withPoints = None, offset=None, scale=None):
        """
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

        """

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
    def __init__(self, angleBins=10, maxDistanceAsLanes=4, laneBins=4, laneWidth=210):
        """
        Constructor
        Defines quantity of bins, factor from distances and angles to it corresponding bins.
        angleBins should be even if you want to look into perpendicular angles.
        Angles range from 0 to pi, both ends are horizontal, so pi/2 is vertical.
        distanceBins is odd, so zero distance is in the middle, 
        to the left go negative distances, to the right go positive distances.

        Arguments:
        - angleBins: number of bins for angles from 0 to pi, p/2 is vertical, even number recommended
        - maxDistanceAsLanes: far edge for distances bins, in lanes
        - laneBins: number of bins in one lane
        - laneWidth: width of a lane in pixels
        """

        self.angleBins = angleBins
        self.maxDistanceAsLanes = maxDistanceAsLanes
        self.laneBins = laneBins
        self.laneWidth = laneWidth

        self.centralAngleBin = angleBins // 2
        self.angle2index = angleBins / math.pi
        self.distance2index = laneBins / laneWidth
        self.maxDistanceInPixels = laneWidth * maxDistanceAsLanes
        self.centralDistanceBin = math.ceil(laneBins * maxDistanceAsLanes)
        self.referenceAngleBin = self.centralAngleBin
        self.distanceBins = 2 * self.centralDistanceBin + 1

    def assign2Bins(self, segments):
        """
        Two parallel arrays of indices are created from segments angles and distances,
        pointing to the corresponding bins in the Hough space.
        They are clipped, so values out of range are set to the nearest valid value.
        This affects distanceIndices, the last bins at both ends will add all the distances greater than maxDistanceInPixels.
        """

        self.angleIndices = np.clip((segments.angles * self.angle2index).astype(int), 0, self.angleBins-1)
        self.distanceIndices = np.clip((segments.distances * self.distance2index + self.centralDistanceBin).astype(int), 0, self.distanceBins-1)

    def getIndicesFromBin(self, angleIndex:int, distanceIndex:int)->np.ndarray:
        """
        Get the indices of elements in the bin that match the given angle and distance bins.

        Arguments:
        - angleIndex: the angle bin index.
        - distanceIndex: the distance bin index.

        Returns:
        - indices of elements in the bin.
        """

        return np.argwhere(np.logical_and(self.angleIndices == angleIndex, self.distanceIndices == distanceIndex)).reshape(-1)

    def computeVotes(self, votes:np.ndarray)->np.ndarray:
        """
        Populates the Hough space.
        votes is a parallel array (same size as segments) with the votes for each segment.
        It often is segments.lengths, but can be any other value.
        If you don't want weighted votes but only counted votes, you can use votes=1.
        It also computes 1D histrogram for angles, and 1D histogram for distances near the central angle bin.

        Arguments:
        - votes: array of votes for each segment.

        Returns:
        - the Hough space.
        """

        self.houghSpace = np.zeros((self.angleBins, self.distanceBins), np.float32)
        np.add.at(self.houghSpace, (self.angleIndices, self.distanceIndices), votes)

        self.maxLoc = np.unravel_index(np.argmax(self.houghSpace), self.houghSpace.shape)
        self.maxVal = self.houghSpace[self.maxLoc]

        self.computeAngleHistogram()
        self.computeLaneHistogram()

        return self.houghSpace

    def computeAngleHistogram(self):
        """
        It needs self.houghSpace to be computed.

        Returns:
        - the 1D angle histogram.
        """

        self.angleHistogram = np.sum(self.houghSpace, axis=1)
        return self.angleHistogram

    def computeLaneHistogram(self, referenceAngleBin=None)->np.ndarray:
        """
        Computes and returns the 1D histogram of distance, only in the referenceAngleBin neighborhood.
        Keep it in self.laneHistogram.
        Given referenceAngleBin is registered in self.referenceAngleBin.

        Arguments:
        - referenceAngleBin: the reference angle bin.

        Returns:
        - the 1D lane histogram.
        """

        if(referenceAngleBin is not None):
            self.referenceAngleBin = referenceAngleBin

        self.laneHistogram = np.sum(self.houghSpace[self.referenceAngleBin-1:self.referenceAngleBin+1], axis=0)[np.newaxis,:]
        return self.angleHistogram

    def getVisualization(self, scale:float=None, showMax:bool=False)->np.ndarray:
        """
        Produce and return a colormapped image of the histogram produced in computeVotes(),
        optionally highliting the peak if showMax is True.

        Arguments:
        - scale: scale factor for the image.

        Returns:
        - the visualization of the Hough space.
        """

        houghSpaceGray = (self.houghSpace * 255/self.maxVal) if self.maxVal>0 else self.houghSpace
        houghSpaceColor = cv.applyColorMap(houghSpaceGray.astype(np.uint8), cv.COLORMAP_DEEPGREEN)
        if(showMax):
            houghSpaceColor[self.maxLoc[0], self.maxLoc[1]] = (0,0,255)
        if(scale is not None):
            houghSpaceColor = cv.resize(houghSpaceColor, None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)

        return houghSpaceColor

    def pasteVisualization(self, image:np.ndarray, borderColor:tuple=(0,128,255), scale:float|None=None, showMax:bool=False)->np.ndarray:
        """
        Pastes the visualization of the Hough space over an image, at the bottom-right corner.
        It also shows 1D histograms of angles and distances.

        Arguments:
        - image: the image to paste the visualization on.
        - borderColor: color for the border.
        - scale: scale factor for the histograms.
        - showMax: whether to highlight the peak in the visualization.

        Returns:
        - the image with the visualization
        """

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
            laneZone = cv.resize(self.laneHistogram, None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)
        else:
            angleHistogram = self.angleHistogram
            laneZone = self.laneHistogram
        
        image[-hh-1:-1, -hw-10:-hw-10+angleHistogram.shape[1]] = angleHistogram[:,:,np.newaxis]
        image[-hh-3-laneZone.shape[0]:-hh-3, -hw-1:-1] = laneZone[:,:,np.newaxis]

        return houghSpaceColor