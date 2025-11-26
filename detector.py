"""
Módulo detector Hough

Este módulo se ocupa del procesamiento de la imagen para la detección de segmentos,
su anotación y su representación en el espacio de Hough.

Contiene 3 clases:

- Segments: representa un conjunto de segmentos, con sus deltas, longitudes, distancias y ángulos.
- SegmentsAnnotator: clase para dibujar segmentos sobre una imagen.
- HoughSpace: clase para calcular el espacio de Hough y sus histogramas.

Status:
- modificado para carril 7.py, que busca orientación y líneas de carril
- adopta histograma 2D par par
"""

from __future__ import annotations  # sólo para hint Segments en init
import numpy as np
import cv2 as cv
import math
from typing import Callable

Color = tuple[int, int, int]

class Segments:
    """
    El objeto Segments representa un conjunto de líneas (segmentos rectos), almacenados en la propiedad coords, como un array de segmentos.
    Cada segmento es un par de puntos 2D.
    Todos los arrays son ndarray con dimensión principal de n.

    Propiedades de Segments:

    n: número de segmentos
    referencePoint: punto 2D, origen para el cálculo de la distancia de Hough;
        es un ndarray de 2 elementos float.


    ndarray de n filas:
    coords: ndarray de segmentos detectados [n,2,2], indefinido en un objeto vacío, se define una vez y no debería cambiar
    deltas: (dx,dy)
    lengths: longitudes de los segmentos
    distances: distancias de Hough al punto de referencia
    angles: ángulos de los segmentos
    normalUnityVectors: vectores unitarios normales a los segmentos
    """
    n: int
    referencePoint: np.ndarray = np.full((2,), np.nan)   # ndarray que fallará si no se lo inicializa
    coords: np.ndarray
    deltas: np.ndarray
    lengths: np.ndarray
    distances: np.ndarray
    angles: np.ndarray
    normalUnityVectors: np.ndarray

    def __init__(self, segs: Segments|np.ndarray|None, referencePoint:tuple|np.ndarray|None=None):
        """
        Constructor

        Arguments:
            segs (np.ndarray or Segments or None): Los segmentos para poblar el objeto. 
                - Si segs es un ndarray de numpy, es una lista de segmentos de FLD con forma (n, 1, 4) donde n es el número de segmentos.
                - Si segs es una instancia de Segments, copiará los segmentos del objeto Segments dado.
                - Si segs es None, se creará un objeto Segments vacío, si quieres rellenarlo tú mismo.
            referencePoint (tuple): El punto de referencia para el cálculo de la distancia de Hough.

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
            self.referencePoint[:] = referencePoint


    def getPointAsIntTuple(self, point:np.ndarray)->tuple:
        """
        Devuelve un punto como una tupla de enteros.
        Así lo requiere drawMarker de OpenCV.

        Returns:
            np.ndarray: El punto de referencia como un array 2D.
        """

        return tuple(point.astype(np.int32))


    def computeDeltas(self):
        """
        Deltas son (dx,dy), segundo punto menos primer punto.
        """

        self.deltas = self.coords[:,1,:] - self.coords[:,0,:]    # Shape n,2, formato #segmento, (dx,dy), dx=x1-x0

    def tidyupDeltas(self):
        """
        Rota 180º si es necesario para tener dy >= 0, de modo que los vectores normales apunten hacia abajo (y positivo apunta hacia abajo).
        Útil para calcular distancias.
        """

        if(not hasattr(self, 'deltas')):
            self.computeDeltas()
        self.deltas = np.where(self.deltas[:,1:]<0, -self.deltas, self.deltas)  # change sign if dy negative

    def computeLengths(self):
        """
        Calcula las longitudes a partir de los deltas.
        """

        if(not hasattr(self, 'deltas')):
            self.computeDeltas()
        self.lengths = np.linalg.norm(self.deltas, axis=1)           # Shape n

    def computeAngles(self):
        """
        Calcula los ángulos a partir de los deltas, con arctan2, por lo que el ángulo está en [-pi, pi].
        Los resultados se almacenan en la propiedad angles.
        Si los deltas están normalizados (dy>=0), los ángulos están en [0, pi].
        No toma argumentos ni devuelve ningún valor.
        """

        if(not hasattr(self, 'deltas')):
            self.computeDeltas()
        self.angles = np.arctan2(self.deltas[:,1], self.deltas[:,0])    # arctan(y/x); Shape n, angles in +-pi, 0 for horizontal

    def computeDistances(self):
        """
        Computa distancias desde referencePoint a los segmentos, con signo.
        No establecer referencePoint generará una excepción.
        Si los deltas no están ordenados, los signos de las distancias dependerán de los deltas.
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
        Ejecuta todo:
        Calcula ángulos, distancias, longitudes, deltas y luego ordena los deltas.
        """

        self.tidyupDeltas()
        self.computeAngles()
        self.computeDistances()

def projectSegments(segments:Segments|np.ndarray|tuple, H:np.ndarray, inverse:bool=False, segmentsShape:bool=True, printFlag:bool=False)->np.ndarray:
    """
    Proyecta los segmentos dados con la homografía H de forma (3,3).

        P' = H * P
    
    Los segmentos son pares de puntos.
    Para proyectar puntos en lugar de segmentos (un número impar de puntos, como solo uno)
    use segmentsShape=False.

    Arguments:
    - segments: conjunto de segmentos.
    - H: matriz de homografía.
    - inverse: si es True, H se considera la inversa de la proyección.
    - segmentsShape: si es True, el resultado se remodela a la forma original de los segmentos.
    - printFlag: si es True, imprime resultados intermedios.

    Returns:
    - ndarray de segmentos proyectados.

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
    Clase para dibujar segmentos sobre una imagen.
    """

    @staticmethod
    def colorMapBGR(intensity:float)->Color:
        """
        Devuelve un color BGR a partir de una intensidad en el rango [0.0,1.0),
        con colores primarios y secundarios brillantes 255.
        Mapea los colores en un patrón cíclico: el color para 0,999 es vecino al color para 0.0.
        Adopta una paleta de 765 colores.

        Arguments:
        - intensity: intensidad en el rango [0.0,1.0).

        Returns:
        - tupla con color BGR.
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
    def colorMapYMC(intensity:float)->Color:
        """
        Devuelve un color BGR a partir de una intensidad en el rango [0.0,1.0),
        con colores brillantes 255.
        Mapea los colores en un patrón cíclico: el color para 0,999 es vecino al color para 0.0.
        Adopta una paleta de 765 colores.

        Arguments:
        - intensity: intensidad en el rango [0.0,1.0).

        Returns:
        - tupla con color BGR.

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

    def __init__(self, color:Color=(0,0,255), thickness:int=1, withPoints:bool = False, offset:tuple=(0,0), scale:float=1.0, colorMap:Callable[[float], Color]=colorMapBGR):
        """
        Constructor
        Establece valores predeterminados para dibujar segmentos sobre una imagen.

        Arguments:
        - color: color para la anotación.
        - thickness: grosor de las líneas.
        - withPoints: si se deben dibujar los puntos finales de los segmentos.
        - offset: desplazamiento para aplicar a los segmentos.
        - scale: factor de escala para aplicar a los segmentos.
        - colorMap: función para mapear intensidades a colores.
        """

        self.color = color
        self.thickness = thickness
        self.withPoints = withPoints
        self.offset = np.array(offset, np.float32)
        self.scale = scale
        self.colorMap = colorMap

    def drawSegments(self, image:np.ndarray, segments:Segments|np.ndarray, intensities:np.ndarray|None=None, message:str|None=None, color:tuple|np.ndarray|None=None, thickness:int|None=None, colorMap:function|None=None, withPoints:bool|None=None, offset:tuple|np.ndarray|None=None, scale:float|None=None)->np.ndarray:
        """
        Dibuja los segmentos sobre una imagen.
        Si se proporcionan intensidades, el color se ignora y se utiliza colorMap.
        Si se proporciona un mensaje, se escribe en la esquina inferior izquierda.
        Si se omiten, estos argumentos tienen valores por defecto en el objeto:

        - color
        - thickness
        - withPoints
        - offset
        - scale

        Args:
            image: Imagen para anotar.
            segments: Segmentos a dibujar.
            intensities: Intensidades para mapear a colores, en el rango [0..1). Mismo tamaño que segments.
            message (str): Mensaje para escribir en la imagen.
            color: Color para la anotación, utilizado si no se proporcionan intensidades.
                Un ndarray de forma (n,3) proporciona un color para cada segmento.
                Por defecto usa self.color.
            thickness (int): Grosor de las líneas.
            colorMap (function): Función para mapear intensidades a colores.
            withPoints (bool): Si se deben dibujar los puntos finales de los segmentos.
            offset (tuple): Desplazamiento para aplicar a los segmentos.
            scale (float): Factor de escala para aplicar a los segmentos.
        """

        colorIsArray = isinstance(color, np.ndarray) and color.ndim == 2

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
        
        for index, segment in enumerate(coords):
            if(intensities is not None):
                segmentColor = self.colorMap(intensities[index])
            elif colorIsArray:
                segmentColor = tuple(color[index].tolist())
            else:
                segmentColor = color

            assert isinstance(segmentColor, tuple)
            pointA = tuple((offset + scale * segment[0]).astype(int))
            pointB = tuple((offset + scale * segment[1]).astype(int))
            cv.line(image, pointA, pointB, color=segmentColor, thickness=thickness, lineType=cv.LINE_AA)
            if(withPoints):
                cv.circle(image, pointA, 2, segmentColor)
                cv.circle(image, pointB, 2, segmentColor)

        if(message is not None):
            textLines = message.split('\n')
            n = len(textLines)
            for i, textLine in enumerate(textLines):
                cv.putText(image, textLine, (10, image.shape[0]-5-20*(n-i-1)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        return image

class HoughSpace:
    """
    Clase para calcular el histograma denominado espacio de Hough.

    shape(howManyAngleBins, howManyDistanceBins), par par

    Ángulos: 0..pi (dos cuadrantes), 0 y pi son horizontales, pi/2 es vertical.
    Líneas longitudinates se encuentran en pi/4..3pi/4 (octantes 2 y 3).

    Distancias: en píxeles, a la izquierda negativas, a la derecha positivas.
    Discontinuidad en ángulo 0 y pi (horizontal).
    Distancias positivas en ángulo 0 son negativas en ángulo pi.
    """
    # Parámetros de construcción: los valores predeterminados se muestran aquí pero se asignan en __init__()
    howManyAngleBins:int=10     # número de bins para ángulos
    maxLanes:int=4    # máxima distancia a considerar, en anchos de carril
    howManyBinsInALane:int=5    # número de bins en un carril, conviene que sea impar
    laneWidthInPixels:int=210   # ancho de un carril en coordenadas cenitales

    # Calculados en __init__()
    centralAngleBin:int     # howManyAngleBins // 2
    angle2index:float       # howManyAngleBins / math.pi
    distance2index: float   # laneBins / laneWidth
    #maxDistance:int         # laneWidth * maxDistanceAsLanes, no se usa
    centralDistanceBin:int  # math.ceil(laneBins * maxDistanceAsLanes)
    referenceAngleBin:int   # centralAngleBin
    howManyDistanceBins:int # 2 * centralDistanceBin

    # Computados en assign2Bins()
    angleIndices:np.ndarray     # array paralelo a los segmentos, con índices de bins para ángulos
    distanceIndices:np.ndarray   # array paralelo a los segmentos, con índices de bins para distancias

    # Computados en computeVotes()
    houghSpace:np.ndarray       # espacio de Hough, histograma 2D de angleBins x distanceBins
    maxLoc:tuple                # coordenada 2D del valor máximo en houghSpace
    maxVal:float                # valor máximo de houghSpace

    # Computados en computeAngleHistogram() y computeLaneHistogram()
    angleHistogram:np.ndarray    # histograma 1D de ángulos, sumando todas las distancias
    laneHistogram:np.ndarray     # histograma 1D del carril máximo

    def __init__(self, howManyAngleBins:int=10, maxLanes:int=4, howManyBinsInALane:int=5, laneWidthInPixels:int=210):
        """
        Constructor
        Define la cantidad de bins, el factor de distancias y ángulos a los bins correspondientes.
        howManyAngleBins debe ser par si se quiere tener bins de ángulos perpendiculares.
        Los ángulos van de 0 a pi, 0 y pi son horizontales, pi/2 es vertical.
        distanceBins: a la izquierda van las distancias negativas, a la derecha las positivas.


        Arguments:
        - howManyAngleBins: número de bins para ángulos de 0 a pi, p/2 es vertical, se recomienda número par.
        - maxLanes: excursión (-maxLanes:+maxLanes) .
        - howManyBinsInALane: número de bins en un carril.
        - laneWidthInPixels: ancho de un carril en píxeles.
        """

        self.howManyAngleBins = howManyAngleBins
        self.maxLanes = maxLanes
        self.howManyBinsInALane = howManyBinsInALane
        self.laneWidthInPixels = laneWidthInPixels

        self.centralAngleBin = howManyAngleBins // 2
        self.referenceAngleBin = self.centralAngleBin
        self.angle2index = howManyAngleBins / np.pi

        self.distance2index = howManyBinsInALane / laneWidthInPixels
        self.centralDistanceBin = math.ceil(howManyBinsInALane * maxLanes)
        self.howManyDistanceBins = 2 * self.centralDistanceBin

    def assign2Bins(self, segments:Segments):
        """
        Dos arrays paralelos de índices se crean a partir de los ángulos y distancias de los segmentos,
        apuntando a los bins correspondientes en el espacio de Hough.
        Se recortan, por lo que los valores fuera de rango se establecen en el valor válido más cercano.
        Esto afecta a distanceIndices, los últimos bins en ambos extremos agregarán todas las distancias mayores que maxDistanceInPixels.
        """

        self.angleIndices    = np.clip((segments.angles    * self.angle2index).astype(int), 0, self.howManyAngleBins-1)
        self.distanceIndices = np.clip((segments.distances * self.distance2index + self.centralDistanceBin).astype(int), 0, self.howManyDistanceBins-1)

    def getIndicesFromBin(self, angleIndex:int, distanceIndex:int)->np.ndarray:
        """
        Devuelve los índices de los elementos en el bin que coinciden con los bins de ángulo y distancia dados.

        Arguments:
        - angleIndex: el índice en rangos de ángulo.
        - distanceIndex: el índice en rangos de distancia.

        Returns:
        - indices: array de índices de elementos en el bin.
        """

        return np.argwhere(np.logical_and(self.angleIndices == angleIndex, self.distanceIndices == distanceIndex)).reshape(-1)

    def computeVotes(self, votes:np.ndarray)->np.ndarray:
        """
        Puebla el espacio de Hough.
        Votes es un array paralelo (del mismo tamaño que segments) con los votos para cada segmento.
        Si no quieres votos ponderados sino solo votos contados, puedes usar votes=1.
        También calcula el histograma 1D para ángulos y el histograma 1D para distancias cerca del bin de ángulo central.

        Arguments:
        - votes: array de votos para cada segmento.

        Returns:
        - el espacio de Hough.
        """

        self.houghSpace = np.zeros((self.howManyAngleBins, self.howManyDistanceBins), np.float32)
        np.add.at(self.houghSpace, (self.angleIndices, self.distanceIndices), votes)

        self.maxLoc = np.unravel_index(np.argmax(self.houghSpace), self.houghSpace.shape)
        self.maxVal = float(self.houghSpace[self.maxLoc])   # este cast debería ser implícito, pero Pylint lo reclama

        self.computeAngleHistogram()
        self.computeLaneHistogram()

        return self.houghSpace

    def computeAngleHistogram(self)->np.ndarray:
        """
        Para cada ángulo, suma los bins de todos los lanes (las distancias).
        El resultado devuelto también se registra en self.angleHistogram.
        Requiere que self.houghSpace esté calculado.

        Returns:
        - el histograma 1D de ángulos.
        """

        self.angleHistogram = np.sum(self.houghSpace, axis=1)
        return self.angleHistogram

    def computeLaneHistogram(self, referenceAngleBin:int|None=None)->np.ndarray:
        """
        Computa y devuelve el histograma 1D de distancia, solo en el vecindario de referenceAngleBin.
        Lo registra en self.laneHistogram.
        El referenceAngleBin dado se registra en self.referenceAngleBin.

        Arguments:
        - referenceAngleBin: el rango de ángulos de referencia.

        Returns:
        - el histograma 1D de carriles.

        El histograma resultante se le da la forma de vector columna, con [np.newaxis,:].
        """

        if(referenceAngleBin is not None):
            self.referenceAngleBin = referenceAngleBin

        self.laneHistogram = np.sum(self.houghSpace[self.referenceAngleBin-1:self.referenceAngleBin+1], axis=0)[np.newaxis,:]
        return self.laneHistogram

    def getVisualization(self, scale:float=0.0, showMax:bool=False)->np.ndarray:
        """
        Produce y devuelve una imagen de color mapeado del histograma 2D producido en computeVotes(),
        opcionalmente resaltando el pico si showMax es True.

        Arguments:
        - scale: factor de escala para la imagen.
        - showMax: si se debe resaltar el pico.

        Returns:
        - la visualización del espacio de Hough.
        """

        houghSpaceGray = (self.houghSpace * 255/self.maxVal) if self.maxVal>0 else self.houghSpace
        houghSpaceColor = cv.applyColorMap(houghSpaceGray.astype(np.uint8), cv.COLORMAP_DEEPGREEN)
        if(showMax):
            houghSpaceColor[self.maxLoc[0], self.maxLoc[1]] = (0,0,255)
        if(scale != 0.0):
            houghSpaceColor = cv.resize(houghSpaceColor, None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)

        return houghSpaceColor

    def pasteVisualization(self, image:np.ndarray, houghSpaceColor:np.ndarray|None = None, borderColor:tuple=(0,128,255), scale:float=0.0, showMax:bool=False)->np.ndarray:
        """
        Pega la visualización del espacio de Hough sobre una imagen, en la esquina inferior derecha.
        Muestra histogramas 1D de ángulos y distancias.
        Decora con ejes y bordes si borderColor no es None.
        Se anota sobre la imagen provista.

        Arguments:
        - image: la imagen para pegar la visualización.
        - borderColor: color para el borde.
        - scale: factor de escala para los histogramas.
        - showMax: si se debe resaltar el pico en la visualización.
        - houghSpaceColor: si se proporciona, se usa esta visualización en lugar de crear una nueva.

        Returns:
        - houghSpaceColor: la visualización del espacio de Hough

        Si se proporciona houghSpaceColor, se debe indicar la escala.
        Para más anotaciones, las coordenadas del espacio de Hough son -houghSpaceColor.shape[:2]
        """

        if(houghSpaceColor is None):
            houghSpaceColor = self.getVisualization(scale, showMax)
        
        ih, iw, _ = image.shape
        hh, hw, _ = houghSpaceColor.shape
        image[-hh-1:-1, -hw-1:-1] = houghSpaceColor
        if(borderColor is not None):
            cv.rectangle(image, (iw-hw-2, ih-hh-2), (iw-1, ih-1), borderColor)
            cv.line(image, (iw-hw//2-1, ih-hh-2), (iw-hw//2-1, ih-1), borderColor)
            cv.line(image, (iw-hw-2, ih-hh//2-1), (iw- 1, ih-hh//2-1), borderColor)

        if(scale != 0.0):
            angleHistogram = cv.resize(self.angleHistogram, None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)
            laneZone       = cv.resize(self.laneHistogram,  None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)
        else:
            angleHistogram = self.angleHistogram
            laneZone = self.laneHistogram
        
        image[-hh-1:-1, -hw-10:-hw-10+angleHistogram.shape[1]] = angleHistogram[:,:,np.newaxis]
        image[-hh-3-laneZone.shape[0]:-hh-3, -hw-1:-1] = laneZone[:,:,np.newaxis]

        return houghSpaceColor