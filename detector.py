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
from timeit import default_timer as timer

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

    def __init__(self, segs: Segments|np.ndarray|None=None, referencePoint:tuple|np.ndarray|None=None):
        """
        Constructor

        Arguments:
            segs (np.ndarray or Segments or None): Los segmentos para poblar el objeto. 
                - Si segs es un ndarray de numpy, es una lista de segmentos de FLD con forma (n, 1, 4) donde n es el número de segmentos.
                - Si segs es una instancia de Segments, copiará los segmentos del objeto Segments dado.
                - Si segs es None, se creará un objeto Segments vacío, si quieres rellenarlo tú mismo.
            referencePoint (tuple): El punto de referencia para el cálculo de la distancia de Hough.
                referencePoint no afecta las coordenadas de los segmentos, 
                que por una cuestión de eficiencia se almacenan sin cambios.
                Se usa en computeDistances().

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
        self.deltas = np.where(self.deltas[:,1:]<0, -self.deltas, self.deltas)  # cambia el signo de delta si dy es negativo

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
        Los ángulos van de 0 a pi, 0 y pi son horizontales, pi/2 es vertical, giran en sentido horario.
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
    
class LaneSensor:
    '''
    Esta clase produce y analiza HoughSpace y segmentos cenitales y en perspectiva,
    para determinar carriles, orientación y otros.
    La clase se puede interpretar como un sensor que extrae información del espacio de Hough.

    El constructor inicializa objetos durables, 
    y se actualiza (update) en cada bucle con la nueva imagen.
    '''

    # Propiedades durables, de construcción
    hs:HoughSpace
    fld:cv.ximgproc.FastLineDetector
    imSize:tuple              # tamaño de la imagen (width,height)
    hui:object                # objeto Hui para colgar el listener afterRoi y para consultar limit

    # Propiedades de ciclo, relacionadas con update()
    FLDT:float                # tiempo de ejecución de FLD en segundos
    im:np.ndarray             # imagen en perspectiva actual
    imGray:np.ndarray         # imagen en escala de grises actual
    H:np.ndarray              # homografía que convierte de perspectiva a cenital
    zenithals:Segments
    perspectives:Segments

    # Propiedades de ciclo, diversos métodos
    compassVersor:np.ndarray  # versor de orientación actual
    maxAngleHistogramIndex:int  # índice del bin angular dominante




    def __init__(self, houghSpace:HoughSpace, fld:cv.ximgproc.FastLineDetector, imSize:tuple, hui=None) -> None:
        '''
        Registra los objetos durables ya inicializados HoughSpace y FastLineDetector.
        Cuelga el listener afterRoi si se proporciona el objeto hui.
        Este constructor no define H, que se actualiza con afterRoi o se establece manualmente:

        self.H = someHomographyMatrix
        
        Arguments:
        houghSpace: objeto HoughSpace ya inicializado
        fld: objeto FastLineDetector ya inicializado
        hui: (opcional) objeto Hui para consultar limit y colgar el listener afterRoi que actualiza H
        '''
        self.hs = houghSpace
        self.fld = fld
        self.imSize = imSize
        if(hui is not None):
            # Cuelga el listener para actualizar H
            hui.afterRoi = self.afterRoi
        else:
            hui = object()
            setattr(hui, 'limit', 0)
        self.hui = hui
        

    def update(self, im:np.ndarray) -> bool:
        '''
        Actualiza el estado, disponibiliza los segmentos que se usan en otros métodos.
        
        Arguments:
        im: imagen en perspectiva para procesar

        Returns:
        bool: True si se detectaron segmentos, False en caso contrario.
        '''
        self.im = im
        self.imGray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

        # Segmentos: cada segmento tiene su izquierda clara y su derecha oscura
        startFLDt = timer()
        lines = self.fld.detect(self.imGray[self.hui.limit:]) # type: ignore
        self.FLDT = timer() - startFLDt
        
        if(lines is None):
            return False
        
        # Las líneas se detectaron en una ROI, se ajustan al sistema de referencia de la imagen completa
        lines += [0, self.hui.limit, 0, self.hui.limit] # type: ignore

        self.perspectives = Segments(lines)
        self.perspectives.computeLengths()  # para los votos ponderados de Hough

        self.zenithals = Segments(projectSegments(self.perspectives, self.H), referencePoint = self.referencePoint)
        self.zenithals.computeAnglesAndDistances()  # parámetros de Hough

        # Espacio de Hough
        self.hs.assign2Bins(self.zenithals)
        self.hs.computeVotes(self.perspectives.lengths)


        # Canonical segments
        self.hs.computeAngleHistogram()

        return True

    def compass(self, lastVersor:np.ndarray|None=None)->np.ndarray:
        '''
        Devuelve el versor de orientación alineado con el segmento más largo.

        Arguments:
        lastVersor: versor de orientación anterior para observar cambios abruptos.  No implementado.

        Returns:
        np.ndarray: versor de orientación actual.

        Guarda el versor en self.compassVersor.
        '''

        peakSegmentsIndices = self.hs.getIndicesFromBin(*self.hs.maxLoc)
        peakSegmentIndex:int = peakSegmentsIndices[np.argmax(self.perspectives.lengths[peakSegmentsIndices])]
        peakZenithalVersor = self.zenithals.deltas[peakSegmentIndex]/self.zenithals.lengths[peakSegmentIndex]
        self.compassVersor = peakZenithalVersor

        return peakZenithalVersor

    def lane(self)->tuple:
        '''

        Returns:
        '''
        
        # Innecesario, sólo para que linter no chille
        leftLaneAvgAngle = 0.0
        leftLaneAvgDistance = 0.0
        rightLaneAvgAngle = 0.0
        rightLaneAvgDistance = 0.0


        # Sinónimo
        halfAngleBins = self.hs.centralAngleBin

        '''
        Dirección principal
        Como las dos direcciones principales son perpendiculares,
        se busca el rango angular máximo sumando ambos histogramas 1D.
        De las dos perpendiculares principales, maxAngleHistogramIndex es la que tiene más votos.
        '''
        foldedAngleHistogram = self.hs.angleHistogram.reshape(2, halfAngleBins).sum(axis=0)
        maxFoldedAngleHistogramIndex:int = int(np.argmax(foldedAngleHistogram))
        isMaxInFirstQuadrant: bool = self.hs.angleHistogram[maxFoldedAngleHistogramIndex] > self.hs.angleHistogram[(maxFoldedAngleHistogramIndex + halfAngleBins)]
        self.maxAngleHistogramIndex:int = maxFoldedAngleHistogramIndex if isMaxInFirstQuadrant else (maxFoldedAngleHistogramIndex + halfAngleBins)

        '''
        De los dos máximos angulares perpendiculares, se toma el más cercano a la dirección principal..
        Por ahora se define la dirección principal como la vertical.
        Cuando funcione en el lazo de control, la dirección principal será la una combinación de las dos direcciones dominantes del cuadro anterior.
        la que se detectó en la imagen anterior y adónde se quiere ir.
        El rango es howManyAngleBins // 2.
        mainAngleBin es el índice angular del carril.
        '''
        mainAngle:float = np.pi/2
        offsetBin:int = int((mainAngle - np.pi/4) * self.hs.angle2index + 0.5)
        mainAngleBin:int = int(maxFoldedAngleHistogramIndex - offsetBin) % halfAngleBins + offsetBin
        self.mainAngleBin:int = mainAngleBin

        '''
        Identificado el índice angular principal mainAngleBin,
        se analizará el histograma de distancias 1D para ese ángulo,
        plegando en anchos de carril (howManyBinsInALane).
        '''
        distanceHistogram = self.hs.houghSpace[mainAngleBin]
        foldedDistanceHistogram = distanceHistogram.reshape(-1, self.hs.howManyBinsInALane).sum(axis=0)
        maxFoldedDistanceHistogramIndex = int(np.argmax(foldedDistanceHistogram))
        leftLaneLineIndex = (mainAngleBin, self.hs.centralDistanceBin + maxFoldedDistanceHistogramIndex)
        rightLaneLineIndex = (mainAngleBin, self.hs.centralDistanceBin + maxFoldedDistanceHistogramIndex - self.hs.howManyBinsInALane)

        '''
        Encontrar los segmentos de las líneas izquierda y derecha de carril en cada bin.
        Al cacular el promedio de distancia se desplaza 50% del ancho de la línea hacia el lado claro, 
        hacia el centro de la línea.
        El ancho de línea es 10 px, no está en ninguna variable, 
        está hardcodeado como +-5.0 en el cómputo de displacements
        '''
        leftLaneSegmentsIndices = self.hs.getIndicesFromBin(*leftLaneLineIndex)
        leftLaneDetected = len(leftLaneSegmentsIndices) > 0
        self.leftLaneDetected = leftLaneDetected
        if leftLaneDetected:
            self.leftLaneSegmentsIndices = leftLaneSegmentsIndices
            leftLaneLengths   = self.perspectives.lengths[leftLaneSegmentsIndices]
            leftLaneAngle    = self.zenithals.angles[leftLaneSegmentsIndices]
            leftLaneDistances = self.zenithals.distances[leftLaneSegmentsIndices]
            leftLaneAvgAngle = np.average(leftLaneAngle, weights=leftLaneLengths)
            displacements = np.where(self.zenithals.coords[leftLaneSegmentsIndices,0,1]<self.zenithals.coords[leftLaneSegmentsIndices,1,1], 5.0, -5.0)
            leftLaneAvgDistance = np.average(leftLaneDistances+displacements, weights=leftLaneLengths)

        rightLaneSegmentsIndices = self.hs.getIndicesFromBin(*rightLaneLineIndex)
        rightLaneDetected = len(rightLaneSegmentsIndices) > 0
        self.rightLaneDetected = rightLaneDetected
        if rightLaneDetected:
            self.rightLaneSegmentsIndices = rightLaneSegmentsIndices
            rightLaneLengths   = self.perspectives.lengths[rightLaneSegmentsIndices]
            rightLaneAngle    = self.zenithals.angles[rightLaneSegmentsIndices]
            rightLaneDistances = self.zenithals.distances[rightLaneSegmentsIndices]
            rightLaneAvgAngle = np.average(rightLaneAngle, weights=rightLaneLengths)
            displacements = np.where(self.zenithals.coords[rightLaneSegmentsIndices,0,1]<self.zenithals.coords[rightLaneSegmentsIndices,1,1], 5.0, -5.0)
            rightLaneAvgDistance = np.average(rightLaneDistances+displacements, weights=rightLaneLengths)

        fullLaneDetected = leftLaneDetected and rightLaneDetected
        if fullLaneDetected:
            self.centralLaneAngle = (leftLaneAvgAngle + rightLaneAvgAngle) / 2
            self.centralLaneDistance = (leftLaneAvgDistance + rightLaneAvgDistance) / 2
        elif leftLaneDetected:
            self.centralLaneAngle = leftLaneAvgAngle
            self.centralLaneDistance = leftLaneAvgDistance - self.hs.laneWidthInPixels/2
        elif rightLaneDetected:
            self.centralLaneAngle = rightLaneAvgAngle
            self.centralLaneDistance = rightLaneAvgDistance + self.hs.laneWidthInPixels/2
        else:
            # Ningún carril detectado
            self.laneDetected = False
            return False,False,0.0,0.0
        
        self.laneDetected = leftLaneDetected or rightLaneDetected
        return leftLaneDetected, rightLaneDetected, self.centralLaneAngle, self.centralLaneDistance

    def endOfLane(self)->tuple:
        '''
        Detecta la línea de fin de carril en cada esquina, y mide la distancia.
        Es una línea perpemndicular a las líneas de carril.
        Se buscan en el bin perpendicular al principal, más menos uno.
        Sólo si el carril fue detectado, de otro modo busca el fin del carril.

        Sólo se debe invocar si el carril fue detectado.
        self.mainAngleBin es la dirección del carril.

        Returns:
        tuple: (endOfLaneDetected:bool, endOfLaneDistance:float, endOfLaneIndex:int)
        '''

        if not self.laneDetected:
            print("Warning: LaneSensor.endOfLane(): no hay carril detectado donde buscar el fin de carril.  Antes de invocar chequear LaneSensor.laneDetected.")
            return False, False, 0.0, -1
        
        # Sinónimo
        halfAngleBins = self.hs.centralAngleBin

        minDistanceIndex = -1   # inicializa con -1 para "no encontrado"
        perpendicularAngleHistogramIndex:int = (self.mainAngleBin + halfAngleBins) % self.hs.howManyAngleBins

        binIndices = []
        minSoFarBinDistance = self.hs.howManyDistanceBins # Infinito
        nearIndex:int
        binDistance:int
        fullNearIndex:tuple
        for i in range(-1, 2):
            angleBin = (perpendicularAngleHistogramIndex + i) % self.hs.howManyAngleBins
            negativeDistance = angleBin >= halfAngleBins
            distanceHistogram = self.hs.houghSpace[angleBin]
            #print(f'distanceHistogram: {type(distanceHistogram)}, {distanceHistogram.shape}, {distanceHistogram.dtype}')
            populatedIndices = np.flatnonzero(distanceHistogram[:self.hs.centralDistanceBin] if negativeDistance else distanceHistogram[self.hs.centralDistanceBin:])
            if len(populatedIndices) == 0:
                continue
            #print(f'populatedIndices {type(populatedIndices)}, {len(populatedIndices)}, {populatedIndices}, {populatedIndices[0]}')
            if negativeDistance:
                nearIndex = populatedIndices[-1]
                binDistance = self.hs.centralDistanceBin - 1 - nearIndex
            else:
                nearIndex = populatedIndices[0]
                binDistance = nearIndex

            if binDistance > minSoFarBinDistance:
                continue

            fullNearIndex = (angleBin, nearIndex if negativeDistance else nearIndex + self.hs.centralDistanceBin)
            if self.hs.houghSpace[fullNearIndex] < 60:
                # umbral arbitrario de votos para ignorar segmentos cortos
                continue

            if binDistance < minSoFarBinDistance:
                minSoFarBinDistance = binDistance
                binIndices = [fullNearIndex]
            else: # binDistance == minBinDistance
                binIndices.append(fullNearIndex)
        
        if len(binIndices)>0:
            segmentIndicesList = []
            for binIndex in binIndices:
                indicesFromBin = self.hs.getIndicesFromBin(*binIndex)
                segmentIndicesList.append(indicesFromBin)
            segmentIndices = np.concatenate(segmentIndicesList)
            minDistanceIndex = abs(self.zenithals.distances[segmentIndices]).argmin()
            minDistanceIndex = segmentIndices[minDistanceIndex]

            # Verificar si el segmento está dentro del carril
            coords = self.zenithals.coords[minDistanceIndex]
            perpenducularLaneVersor = np.array([-np.sin(self.centralLaneAngle), np.cos(self.centralLaneAngle)], np.float32)
            laneOrigin = perpenducularLaneVersor * self.centralLaneDistance + self.referencePoint
            distanceToLaneCenter = (coords - laneOrigin) @ perpenducularLaneVersor
            '''
            print('coords:', coords)
            print('laneOrigin:', laneOrigin)
            print('perpenducularLaneVersor:', perpenducularLaneVersor)
            print(f'distanceToLaneCenter: {distanceToLaneCenter}')
            '''
            halfLaneWidth = self.hs.laneWidthInPixels * 0.4    # menos de medio carril, incluye margen de seguridad
            isItIn = not(
                   (distanceToLaneCenter[0] >  halfLaneWidth and distanceToLaneCenter[1] >  halfLaneWidth) \
                or (distanceToLaneCenter[0] < -halfLaneWidth and distanceToLaneCenter[1] < -halfLaneWidth) )


            return True, isItIn, abs(self.zenithals.distances[minDistanceIndex]), minDistanceIndex
        else:
            return False, False, 0.0, -1

    def afterRoi(self, hui) -> None:
        '''
        Listener para colgar en Hui.afterRoi, 
        para actualizar la homografía H cada vez que el usuario dispara su recómputo.
        Ejemplo de uso:

        hui.afterRoi = laneSensor.afterRoi

        Arguments:
        hui: objeto Hui con la propiedad Hview

        No hace falta importar HUI
        '''
        self.H = hui.Hview
        self.referencePoint = np.array((self.hui.zenithalSize[0]//2, self.hui.zenithalSize[1]), np.float32) # type: ignore