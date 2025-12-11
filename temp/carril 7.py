'''
Derivada de carril.py, a su vez derivada de carril6.py
Este programa se concentra en la detección de las direcciones dominantes.

La única homografía usada en Hview.  H se carga y se guarda, y se usa exlusivamente en HUI, entre otras cosas para calcular Hview.
'''

import numpy as np
import cv2 as cv
import detector3 as det
import HUI
import argparse
from timeit import default_timer as timer

#Setup

np.set_printoptions(precision=1)

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", help="flujo de video de entrada: la ruta de un archivo de video o el número de cámara", default="bfmc2020_online_2.avi")
parser.add_argument("-l", "--load", help="archivo donde cargar los resultados")
parser.add_argument("-s", "--save", help="archivo donde guardar los resultados")
args = parser.parse_args()

if(not args.save or not args.load):
    filename = args.video + '.yaml'
    if(not args.save):
        args.save = filename
    if(not args.load):
        args.load = filename

# Help
print(
    'Keys:'
    '\n  ESC: exit'
    '\n  SPACE: play/pause'
    '\n  v: change visualization'
    '\n  p: print debug info'
    '\n  s: save homography'
    '\n  l: load homography'
    '\n  by default save and load use the same file name as the video, but with .yaml extension:'
    )

# Util
play = True # play/pause flag
fs = None   # yaml
userVisualizationOption = 0 # user cycles these open options

# Video
video = cv.VideoCapture(args.video)
frameWidth = video.get(cv.CAP_PROP_FRAME_WIDTH)
frameHeight = video.get(cv.CAP_PROP_FRAME_HEIGHT)
print('Video res:',frameHeight, frameWidth)

# Adjust frame size
targetFrameHeight = 480
mustResize = False
if(frameHeight > targetFrameHeight):
    targetFrameWidth = int(frameWidth/frameHeight * targetFrameHeight)
    video.set(cv.CAP_PROP_FRAME_WIDTH,  targetFrameWidth)
    video.set(cv.CAP_PROP_FRAME_HEIGHT, targetFrameHeight)

    _, im = video.read()
    (frameHeight, frameWidth, canales) = im.shape
    if(frameHeight > targetFrameHeight):
        frameHeight = targetFrameHeight
        frameWidth = targetFrameWidth
        mustResize = True

frameSize = (int(frameWidth), int(frameHeight))
print('Resized image size :', frameSize, type(frameWidth))

def load():
    '''
    Carga la matriz desde yaml, y actualiza la anotación.  
    El nombre del archivo está en args.load. Si el archivo no existe no hace nada.

    Esta función se invoca al inicio y luego desde la UI con la tecla 'l'.
    '''
    fs = cv.FileStorage(args.load, cv.FILE_STORAGE_READ)
    if(fs.isOpened()):
        hui.H = fs.getNode('H').mat()
        hui.calculateRoi(hui.H)
        fs.release()

def save():
    '''
    Guarda la matriz en yaml.  
    El nombre del archivo está en args.save.  
    Si el archivo no existe lo crea.

    Esta función se invoca exclusivamente desde la UI con la tecla 's'.
    '''
    fs = cv.FileStorage(args.save, cv.FILE_STORAGE_WRITE)
    fs.write('H', hui.H)
    fs.release()


hui = HUI.Hui('hui', frameSize)
load()
hui.zenithalSquareSide = 250
hui.calculateRoi()


# Inicialización del loop principal
'''
- square side 250px
- lane: about 220px
- lane line: 10px
- angles 0..pi
'''
hs = det.HoughSpace(howManyBinsInALane=5)
lastAngleBin = hs.howManyAngleBins // 2
annotations = det.SegmentsAnnotator()
zenithalAnnotations = det.SegmentsAnnotator(thickness=2, colorMap=det.SegmentsAnnotator.colorMapBGR)
umbralCanny = 160
printFlag = False
fld = cv.ximgproc.createFastLineDetector(canny_th1 = umbralCanny, canny_th2 = umbralCanny*3, do_merge=True)
halfAngleBins: int = hs.centralAngleBin #hs.howManyAngleBins // 2


# Inicialización para suprimir warnings
leftLaneAvgAngle = 0.0
leftLaneAvgDistance = 0.0
rightLaneAvgAngle = 0.0
rightLaneAvgDistance = 0.0

while(True):
    # video feed
    if(play):
        _, im = video.read()
        if(not _):
            break
        if(mustResize):
            im = cv.resize(im, frameSize)
        hui.anotate(im)

    imGray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    # Fast Line Detector - ROI: horizonte hacia abajo
    startFLDt = timer()
    lines = fld.detect(imGray[hui.limit:])
    endFLDt = timer()
    if(lines is None):
        # no lines, skip
        continue

    startProcesst = timer()

    # Detección de segmentos y Hough ======================
    lines += [0, hui.limit, 0, hui.limit]
    segments = det.Segments(lines)
    segments.computeLengths()   # used for votes

    # Los segmentos cenitales están en coordenadas de visualización
    # referencePoint es el punto de origen para la parametrizacíon de Hough, al pie de la imagen, al centro.
    zenithals = det.Segments(det.projectSegments(segments, hui.Hview),
                             referencePoint=(hui.zenithalSize[0]//2, hui.zenithalSize[1]))
    zenithals.computeAnglesAndDistances()   # Hough variables    
    hs.assign2Bins(zenithals)
    hs.computeVotes(zenithals.lengths if userVisualizationOption else segments.lengths)   # alternative: zenithals.lengths

    # Canonical segments
    hs.computeAngleHistogram()
    maxAngleBin = np.argmax(hs.angleHistogram)
    angleBinVariation = abs(maxAngleBin-lastAngleBin) #(maxAngleBin-lastAngleBin+1) % hs.howManyAngleBins # 0..2
    if angleBinVariation <= 1:
        # Normal, máximo cambio +-1
        mainAngleBin = maxAngleBin
        perpendicularAngleBin = (maxAngleBin + halfAngleBins) % hs.howManyAngleBins
        status = 0

    elif abs(angleBinVariation - halfAngleBins) <= 1:  #angleBinVariation >= halfAngleBins and angleBinVariation <= halfAngleBins + 2:
        # Perpendicular
        mainAngleBin = (maxAngleBin - halfAngleBins) % hs.howManyAngleBins
        perpendicularAngleBin = maxAngleBin
        status = 1

    else:
        # Perdido, tomar el máximo como principal
        mainAngleBin = maxAngleBin
        perpendicularAngleBin = (maxAngleBin + halfAngleBins) % hs.howManyAngleBins
        status = 2

    lastAngleBin = mainAngleBin


    '''
    Brújula, dirección del segmento mayor, aproximación simple a la dirección principal.
    Se toma el rango pico, el de más votos, y dentro de él, el segmento más largo.
    peakZenithalVersor es un vector unitario que apunta en la dirección de ese segmento.
    '''
    peakSegmentsIndices = hs.getIndicesFromBin(*hs.maxLoc)
    peakSegmentIndex:int = peakSegmentsIndices[np.argmax(segments.lengths[peakSegmentsIndices])]
    peakZenithalVersor = zenithals.deltas[peakSegmentIndex]/zenithals.lengths[peakSegmentIndex]


    '''
    Dirección principal
    Como las dos direcciones principales son perpendiculares,
    se busca el rango angular máximo sumando ambos histogramas 1D.
    De las dos perpendiculares principales, maxAngleHistogramIndex es la que tiene más votos.
    '''
    foldedAngleHistogram = hs.angleHistogram.reshape(2, halfAngleBins).sum(axis=0)
    maxFoldedAngleHistogramIndex:int = int(np.argmax(foldedAngleHistogram))
    isMaxInFirstQuadrant: bool = hs.angleHistogram[maxFoldedAngleHistogramIndex] > hs.angleHistogram[(maxFoldedAngleHistogramIndex + halfAngleBins)]
    maxAngleHistogramIndex:int = maxFoldedAngleHistogramIndex if isMaxInFirstQuadrant else (maxFoldedAngleHistogramIndex + halfAngleBins)

    '''
    De los dos máximos angulares perpendiculares, se toma el más cercano a la dirección principal..
    Por ahora se define la dirección principal como la vertical.
    Cuando funcione en el lazo de control, la dirección principal será la una combinación de las dos direcciones dominantes del cuadro anterior.
    la que se detectó en la imagen anterior y adónde se quiere ir.
    El rango es howManyAngleBins // 2.
    mainAngleBin es el índice angular del carril.
    '''
    mainAngle:float = np.pi/2
    offsetBin:int = int((mainAngle - np.pi/4) * hs.angle2index + 0.5)
    mainAngleBin:int = int(maxFoldedAngleHistogramIndex - offsetBin) % halfAngleBins + offsetBin

    '''
    Identificado el índice angular principal mainAngleBin,
    se analizará el histograma de distancias 1D para ese ángulo,
    plegando en anchos de carril (howManyBinsInALane).
    '''
    distanceHistogram = hs.houghSpace[mainAngleBin]
    foldedDistanceHistogram = distanceHistogram.reshape(-1, hs.howManyBinsInALane).sum(axis=0)
    maxFoldedDistanceHistogramIndex = int(np.argmax(foldedDistanceHistogram))
    leftLaneLineIndex = (mainAngleBin, hs.centralDistanceBin + maxFoldedDistanceHistogramIndex)
    rightLaneLineIndex = (mainAngleBin, hs.centralDistanceBin + maxFoldedDistanceHistogramIndex - hs.howManyBinsInALane)

    '''
    Encontrar los segmentos de las líneas izquierda y derecha de carril en cada bin.
    '''
    leftLaneSegmentsIndices = hs.getIndicesFromBin(*leftLaneLineIndex)
    leftLaneLengths   = segments.lengths[leftLaneSegmentsIndices]
    leftLaneDetected = len(leftLaneLengths) > 0
    if leftLaneDetected:
        leftLaneAngle    = zenithals.angles[leftLaneSegmentsIndices]
        leftLaneDistances = zenithals.distances[leftLaneSegmentsIndices]
        leftLaneAvgAngle = np.average(leftLaneAngle, weights=leftLaneLengths)
        leftLaneAvgDistance = np.average(leftLaneDistances, weights=leftLaneLengths)

    rightLaneSegmentsIndices = hs.getIndicesFromBin(*rightLaneLineIndex)
    rightLaneLengths   = segments.lengths[rightLaneSegmentsIndices]
    rightLaneDetected = len(rightLaneLengths) > 0
    if rightLaneDetected:
        rightLaneAngle    = zenithals.angles[rightLaneSegmentsIndices]
        rightLaneDistances = zenithals.distances[rightLaneSegmentsIndices]
        rightLaneAvgAngle = np.average(rightLaneAngle, weights=rightLaneLengths)
        rightLaneAvgDistance = np.average(rightLaneDistances, weights=rightLaneLengths)

    laneDetected = leftLaneDetected or rightLaneDetected
    fullLaneDetected = leftLaneDetected and rightLaneDetected
    if fullLaneDetected:
        centralLaneAngle = (leftLaneAvgAngle + rightLaneAvgAngle) / 2
        centralLaneDistance = (leftLaneAvgDistance + rightLaneAvgDistance) / 2
    elif leftLaneDetected:
        centralLaneAngle = leftLaneAvgAngle
        centralLaneDistance = leftLaneAvgDistance - hs.laneWidthInPixels/2
    elif rightLaneDetected:
        centralLaneAngle = rightLaneAvgAngle
        centralLaneDistance = rightLaneAvgDistance + hs.laneWidthInPixels/2


    '''
    Línea de fin de carril, en cada esquina.
    Es una línea perpemndicular a las líneas de carril.
    Se buscan en el bin perpendicular al principal, más menos uno.
    Sólo si el carril fue detectado, de otro modo busca el fin del carril.
    mainAngleBin es la dirección del carril.
    '''

    minDistanceIndex = -1   # inicializa con -1 para "no encontrado"
    if laneDetected:
        perpendicularAngleHistogramIndex:int = (mainAngleBin + halfAngleBins) % hs.howManyAngleBins

        binIndices = []
        minSoFarBinDistance = hs.howManyDistanceBins # Infinito
        nearIndex:int
        binDistance:int
        fullNearIndex:tuple
        for i in range(-1, 2):
            angleBin = (perpendicularAngleHistogramIndex + i) % hs.howManyAngleBins
            negativeDistance = angleBin >= halfAngleBins
            distanceHistogram = hs.houghSpace[angleBin]
            #print(f'distanceHistogram: {type(distanceHistogram)}, {distanceHistogram.shape}, {distanceHistogram.dtype}')
            populatedIndices = np.flatnonzero(distanceHistogram[:hs.centralDistanceBin] if negativeDistance else distanceHistogram[hs.centralDistanceBin:])
            if len(populatedIndices) == 0:
                continue
            #print(f'populatedIndices {type(populatedIndices)}, {len(populatedIndices)}, {populatedIndices}, {populatedIndices[0]}')
            if negativeDistance:
                nearIndex = populatedIndices[-1]
                binDistance = hs.centralDistanceBin - 1 - nearIndex
            else:
                nearIndex = populatedIndices[0]
                binDistance = nearIndex

            if binDistance > minSoFarBinDistance:
                continue

            fullNearIndex = (angleBin, nearIndex if negativeDistance else nearIndex + hs.centralDistanceBin)
            if hs.houghSpace[fullNearIndex] < 60:
                # umbral arbitrario de votos
                continue

            if binDistance < minSoFarBinDistance:
                minSoFarBinDistance = binDistance
                binIndices = [fullNearIndex]
            else: # binDistance == minBinDistance
                binIndices.append(fullNearIndex)
        
        if len(binIndices)>0:
            segmentIndicesList = []
            for binIndex in binIndices:
                indicesFromBin = hs.getIndicesFromBin(*binIndex)
                segmentIndicesList.append(indicesFromBin)
            segmentIndices = np.concatenate(segmentIndicesList)
            minDistanceIndex = abs(zenithals.distances[segmentIndices]).argmin()
            minDistanceIndex = segmentIndices[minDistanceIndex]

            # quitar segmentos muy cortos, de longitur < 60 px


    endProcesst = timer()

    '''
    Visualización y anotación

    Esta sección no está optimizada para velocidad.
    No debería ejecutarse en el auto.
    Incluye conversiones de tipo innecesarias, sólo para suprimir warnings equivocados,
    y la operación lenta warpPerspective.
    '''
    startAnnotationt = timer()

    # Visualización de perspectiva =====================

    imAnnotated = cv.cvtColor(imGray//2, cv.COLOR_GRAY2BGR)
    

    # Pallette:
    colors = np.empty((segments.n, 3), dtype=np.uint8)
    colors[hs.angleIndices == mainAngleBin] = np.array([0,255,0] if status == 0 else [0,128,0], np.uint8) # main direction
    colors[hs.angleIndices == perpendicularAngleBin] = np.array([255,0,0] if status == 1 else [128,0,0], np.uint8) # perpendicular direction
    colors[(hs.angleIndices != mainAngleBin) & (hs.angleIndices != perpendicularAngleBin)] = np.array([0,0,128], np.uint8) # other directions

    '''
    # draw segments
    annotations.drawSegments(imAnnotated, segments, color=colors)
    '''

    # Líneas de carril
    annotations.drawSegments(imAnnotated, 
                             segments.coords[leftLaneSegmentsIndices], 
                             color=(0,128,255), thickness=4)
    annotations.drawSegments(imAnnotated,
                             segments.coords[rightLaneSegmentsIndices], 
                             color=(255,0,128), thickness=4)


    # Brújula
    origin = (imAnnotated.shape[1]//2, (imAnnotated.shape[0]+hui.limit)//2)
    zenithalOrigin = det.projectSegments(origin, hui.Hview, segmentsShape=False, printFlag=printFlag).reshape(-1)
    peakZenithalVector = peakZenithalVersor * hs.laneWidthInPixels/4
    peakZenithalPerpendicularVector = np.array((peakZenithalVector[1], -peakZenithalVector[0]))
    mainAxisZenithalSegments = np.array([
            zenithalOrigin + peakZenithalVector,
            zenithalOrigin - peakZenithalVector,
            zenithalOrigin - peakZenithalPerpendicularVector,
            zenithalOrigin + peakZenithalPerpendicularVector
        ]).reshape(-1, 2, 2)
    mainAxisSegments = det.projectSegments(mainAxisZenithalSegments, hui.Hview, inverse=True, printFlag=printFlag)


    # Brújula
    annotations.drawSegments(imAnnotated, mainAxisSegments, 
                             color=(255,255,0) if status == 0 else (255,128,0) if status == 1 else (64,0,255))


    # Visualización cenital =====================

    # zenithal fustrum view
    zenithalIm = cv.warpPerspective(im, hui.Hview, tuple(hui.zenithalSize))

    # Red: base origin
    cv.drawMarker(zenithalIm, tuple(zenithals.getPointAsIntTuple(zenithals.referencePoint)), (0,0,255), cv.MARKER_CROSS, 20, 3)

    # Green: main axes origin
    cv.drawMarker(zenithalIm, zenithals.getPointAsIntTuple(zenithalOrigin), (0,255,0), cv.MARKER_CROSS, 20, 3)

    # segments, and message
    zenithalAnnotations.drawSegments(zenithalIm, zenithals, #intensities=zenithals.angles/3.17,
                                    color=colors,
                                    message= f'FLD: {(endFLDt-startFLDt)*1000:.0f} ms'
                                    f'\nProcess: {(endProcesst-startProcesst)*1000:.0f} ms'
                                    f'\nSegments {str(len(zenithals.coords))}'
                                    f"\nHough votes: {'zenithals' if userVisualizationOption else 'segments'}"
                                   )

    # Líneas de carril aproximadas
    zenithalAnnotations.drawSegments(zenithalIm, 
                                     zenithals.coords[leftLaneSegmentsIndices], 
                                     color=(0,128,255))
    zenithalAnnotations.drawSegments(zenithalIm, 
                                     zenithals.coords[rightLaneSegmentsIndices], 
                                     color=(255,0,128))

    # Brújula
    zenithalAnnotations.drawSegments(zenithalIm, mainAxisZenithalSegments, color=(255,255,0))

    # Dirección central del carril
    if laneDetected:
        origin = np.array((zenithalIm.shape[1]//2, zenithalIm.shape[0]), dtype=np.int32)
        laneVersor = np.array((np.cos(centralLaneAngle), np.sin(centralLaneAngle)), dtype=np.float32)
        base = origin + (int(centralLaneDistance / laneVersor[1]), 0)        
        arrowColor = (0,192,0) if fullLaneDetected else (0,128,255) if leftLaneDetected else (192,0, 192)

        arrow = base - (laneVersor * 100).astype(np.int32)
        cv.arrowedLine(zenithalIm, tuple(base.astype(int)), tuple(arrow.astype(int)), arrowColor, 4, line_type=cv.LINE_AA)

        perpectiveArrow = det.projectSegments(np.stack([base, arrow]), hui.Hview, inverse=True).astype(np.int32)
        cv.arrowedLine(imAnnotated, tuple(perpectiveArrow[0]), tuple(perpectiveArrow[1]), arrowColor, 4, line_type=cv.LINE_AA)


        # Línea de fin de carril (la transversal más cercana)
        if minDistanceIndex > -1:
            zenithalAnnotations.drawSegments(zenithalIm,
                                            zenithals.coords[minDistanceIndex].reshape(-1,2,2),
                                            color=(128,128,255), thickness=4)
            annotations.drawSegments(imAnnotated,
                                            segments.coords[minDistanceIndex].reshape(-1,2,2),
                                            color=(128,128,255), thickness=4)



    # autoshrink
    while(zenithalIm.shape[0] > 700):
        zenithalIm = cv.resize(zenithalIm, None, fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)

    houghSpaceColor = hs.pasteVisualization(zenithalIm, scale=4.0, showMax=True)
    vizVertex = np.array(zenithalIm.shape[1::-1])-houghSpaceColor.shape[1::-1]
    angleBinCoords = vizVertex + (-9, 4*maxAngleHistogramIndex+1)
    #angleBinCoords = (zenithalIm.shape[1]-houghSpaceColor.shape[1]-9, zenithalIm.shape[0]-houghSpaceColor.shape[0]+4*maxAngleHistogramIndex+1)
    cv.drawMarker(zenithalIm, tuple(angleBinCoords), (0,0,255), cv.MARKER_SQUARE, 4)
    #print('Angle bin:', maxAngleHistogramIndex)

    # Sombra para el texto
    imAnnotated[0:50, 0:150] //= 2

    # Texto
    x=10
    h=20
    color=(255,255,255)
    cv.putText(imAnnotated, f'linea izquierda: {leftLaneDetected}', (x,h), cv.FONT_HERSHEY_SIMPLEX, 0.4, color)
    cv.putText(imAnnotated, f'linea derecha: {rightLaneDetected}', (x,h*2), cv.FONT_HERSHEY_SIMPLEX, 0.4, color)
    if minDistanceIndex > -1:
        cv.putText(imAnnotated, f'fin de carril: {abs(zenithals.distances[minDistanceIndex]):.2f}', (x,h*3), cv.FONT_HERSHEY_SIMPLEX, 0.4, color)



    cv.imshow('Main segments', imAnnotated)
    cv.imshow('zenithal wide', zenithalIm)

    endAnnotationt = timer()
    #print(f'FLD: {endFLDt-startFLDt:.3f} s, Process: {endProcesst-startProcesst:.3f} s, Annotate: {endAnnotationt-startAnnotationt:.3f} s')

    # user keyboard
    printFlag = False
    key = cv.waitKey(30)
    if(key != -1):
        match key:
            case 27:
                break

        hui.processKey(key)

        match chr(key):
            case ' ':
                # play/pause
                play = not play
            case 'v':
                userVisualizationOption += 1
                userVisualizationOption %= 2
            case 'h':
                # print detailed hough info
                print('hs.houghSpace:', hs.houghSpace)
                print('hs.angleHistogram:', hs.angleHistogram)
                print('foldedAngleHistogram:', foldedAngleHistogram)
                print(f'maxAngleHistogramIndex: {maxAngleHistogramIndex}')
                print('hs.distanceHistogram:', distanceHistogram)
                print('foldedDistanceHistogram:', foldedDistanceHistogram)
            case 'p':   
                # print debug info
                printFlag = True
                print('image shape:', im.shape)
                print('roiPoly', hui.roiPoly, type(hui.roiPoly), type(hui.roiPoly[0]), type(hui.roiPoly[0][0]))
                #print(f"winnerBin: {winnerBin}")
                #print(f"mainSegmentsIndices: {len(mainSegmentsIndices)}")
                print(f'H2 det: {np.linalg.det(hui.Hview)}, \nH2: {hui.Hview}')
                print(f'H2 inv: {np.linalg.inv(hui.Hview)}')
                print(f'origin: {origin}')
                print(f'zenithalOrigin: {zenithalOrigin}')
                #print(f'zenithalForward: {zenithalForward}')
                #print(f'zenithalSide: {zenithalSide}')
                print(f'mainAxisSegments: {mainAxisZenithalSegments}')
                print(f'mainAxisSegmentsPerspective: {mainAxisSegments}')
                #print(*mainSegmentsIndices)
                print(f'angle histogram: {hs.angleHistogram}')
                print(f'lane histogram: {hs.laneHistogram}')
                #print('hs.houghSpace:')
                #print(hs.houghSpace)
                #print('angleBinsIndices, distanceBinsIndices:')
                '''
                for a,b,c,d in zip(bins.angleBinsIndices, bins.distanceBinsIndices, zenithals.angles, zenithals.distances):
                    print(a,b,c,d)
                '''
            case 'o':
                # print lane line info
                print('Longitud de línea de fin de carril', segments.lengths[minDistanceIndex] if minDistanceIndex>-1 else 'N/A')
                if minDistanceIndex>-1:
                    for bin in binIndices:
                        print(f'bin: {bin[0]}, {bin[1]}, votos: {hs.houghSpace[bin]}')

                
            case 's':
                save()
            case 'l':
                load()

# Close and exit
if(fs):
    fs.release()
cv.destroyAllWindows()