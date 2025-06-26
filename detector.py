"""
Módulo detector Hough

Este módulo se ocupa del procesamiento de la imagen para la detección de segmentos,
su anotación y su representación en el espacio de Hough.

Contiene 3 clases:

- Segments: representa un conjunto de segmentos, con sus deltas, longitudes, distancias y ángulos.
- SegmentsAnnotator: clase para dibujar segmentos sobre una imagen.
- HoughSpace: clase para calcular el espacio de Hough y sus histogramas.

"""

from __future__ import annotations  # sólo para hint Segments en init
import numpy as np
#from numpy.typing import NDArray
#from typing import Any
import cv2 as cv
import math

class Segments:
    """
    El objeto Segments representa un conjunto de líneas (segmentos rectos), almacenados en la propiedad coords, como un array de segmentos.
    Cada segmento es un par de puntos 2D.
    Todos los arrays son ndarray con dimensión principal de n.

    Propiedades de Segments:

    n: número de segmentos
    referencePoint: punto 2D, origen para el cálculo de la distancia de Hough

    ndarray de n filas:
    coords: ndarray de segmentos detectados [n,2,2], indefinido en un objeto vacío, se define una vez y no debería cambiar
    deltas: (dx,dy)
    lengths: longitudes de los segmentos
    distances: distancias de Hough al punto de referencia
    angles: ángulos de los segmentos
    normalUnityVectors: vectores unitarios normales a los segmentos
    """
    n: int
    referencePoint: np.ndarray
    coords: np.ndarray
    deltas: np.ndarray
    lengths: np.ndarray
    distances: np.ndarray
    angles: np.ndarray
    normalUnityVectors: np.ndarray

    def __init__(self, segs: Segments|np.ndarray|None, referencePoint:tuple|None=None):
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
            self.setReferencePoint(referencePoint)

    
    def setReferencePoint(self, point:tuple):
        """
        Establece el punto de referencia para el cálculo de la distancia de Hough.
        No tiene otro propósito.
        Debes llamar explícitamente a este método para establecer el punto de referencia.
        Calcular distancias sin un punto de referencia generará una excepción,
        porque tener un referencePoint aleatorio es peor.

        Arguments:
            point (tuple): El punto de referencia como tupla 2D.

        """

        self.referencePoint = np.array(point, dtype=np.float32)

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
    def colorMapBGR(intensity:float)->tuple:
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
    def colorMapYMC(intensity:float)->tuple:
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

    def __init__(self, color:tuple=(0,0,255), thickness:int=1, withPoints:bool = False, offset:tuple=(0,0), scale:float=1.0, colorMap:function=colorMapBGR):
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
        Otros argumentos permiten al usuario anular los valores predeterminados.

        Args:
            image: Imagen para anotar.
            segments: Segmentos a dibujar.
            intensities: Intensidades para mapear a colores, en el rango [0..1). Mismo tamaño que segments.
            message (str): Mensaje para escribir en la imagen.
            color (tuple): Color para la anotación, utilizado si no se proporcionan intensidades.
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
        
        for index, segments in enumerate(coords):
            if(intensities is not None):
                segmentColor = self.colorMap(intensities[index])
            elif colorIsArray:
                segmentColor = color[index].tolist()
                #print(f'index: {index}\ncolor:{segmentColor}, type: {type(segmentColor)}, element type: {type(segmentColor[0])}')
            else:
                segmentColor = color

            pointA = (offset + scale * segments[0]).astype(int)
            pointB = (offset + scale * segments[1]).astype(int)
            cv.line(image, pointA, pointB, color=segmentColor, thickness=thickness)
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
    def __init__(self, angleBins:int=10, maxDistanceAsLanes:int=4, laneBins:int=4, laneWidth:int=210):
        """
        Constructor
        Define la cantidad de bins, el factor de distancias y ángulos a los bins correspondientes.
        angleBins debe ser par si quieres mirar ángulos perpendiculares.
        Los ángulos van de 0 a pi, ambos extremos son horizontales, por lo que pi/2 es vertical.
        distanceBins debe ser impar, por lo que la distancia cero está en el medio,
        a la izquierda van las distancias negativas, a la derecha las positivas.

        Arguments:
        - angleBins: número de bins para ángulos de 0 a pi, p/2 es vertical, se recomienda número par.
        - maxDistanceAsLanes: borde lejano para los bins de distancias, en carriles.
        - laneBins: número de bins en un carril.
        - laneWidth: ancho de un carril en píxeles.
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

    def assign2Bins(self, segments:Segments):
        """
        Dos arrays paralelos de índices se crean a partir de los ángulos y distancias de los segmentos,
        apuntando a los bins correspondientes en el espacio de Hough.
        Se recortan, por lo que los valores fuera de rango se establecen en el valor válido más cercano.
        Esto afecta a distanceIndices, los últimos bins en ambos extremos agregarán todas las distancias mayores que maxDistanceInPixels.
        """

        self.angleIndices = np.clip((segments.angles * self.angle2index).astype(int), 0, self.angleBins-1)
        self.distanceIndices = np.clip((segments.distances * self.distance2index + self.centralDistanceBin).astype(int), 0, self.distanceBins-1)

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

        self.houghSpace = np.zeros((self.angleBins, self.distanceBins), np.float32)
        np.add.at(self.houghSpace, (self.angleIndices, self.distanceIndices), votes)

        self.maxLoc = np.unravel_index(np.argmax(self.houghSpace), self.houghSpace.shape)
        self.maxVal = self.houghSpace[self.maxLoc]

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
        Produce y devuelve una imagen de color mapeado del histograma producido en computeVotes(),
        opcionalmente resaltando el pico si showMax es True.

        Arguments:
        - scale: factor de escala para la imagen.

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

    def pasteVisualization(self, image:np.ndarray, borderColor:tuple=(0,128,255), scale:float=0.0, showMax:bool=False)->np.ndarray:
        """
        Pega la visualización del espacio de Hough sobre una imagen, en la esquina inferior derecha.
        También muestra histogramas 1D de ángulos y distancias.

        Arguments:
        - image: la imagen para pegar la visualización.
        - borderColor: color para el borde.
        - scale: factor de escala para los histogramas.
        - showMax: si se debe resaltar el pico en la visualización.

        Returns:
        - image: la imagen con la visualización.
        """

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
            laneZone = cv.resize(self.laneHistogram, None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)
        else:
            angleHistogram = self.angleHistogram
            laneZone = self.laneHistogram
        
        image[-hh-1:-1, -hw-10:-hw-10+angleHistogram.shape[1]] = angleHistogram[:,:,np.newaxis]
        image[-hh-3-laneZone.shape[0]:-hh-3, -hw-1:-1] = laneZone[:,:,np.newaxis]

        return houghSpaceColor