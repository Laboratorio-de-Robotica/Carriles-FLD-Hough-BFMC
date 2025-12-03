### Copia de carril7.py , para separar en este main y en la biblioteca nueva lane.py


'''
Derivada de carril.py, a su vez derivada de carril6.py
Este programa se concentra en la detección de las direcciones dominantes.

La única homografía usada en Hview.  H se carga y se guarda, y se usa exlusivamente en HUI, entre otras cosas para calcular Hview.
'''

import numpy as np
import cv2 as cv
import detector as det
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


# Inicialización del loop principal
'''
- square side 250px
- lane: about 220px
- lane line: 10px
- angles 0..pi
'''
printFlag = False

hs = det.HoughSpace(howManyAngleBins=10, maxLanes=4, howManyBinsInALane=5, laneWidthInPixels=210)
halfAngleBins: int = hs.centralAngleBin #hs.howManyAngleBins // 2

umbralCanny = 160
fld = cv.ximgproc.createFastLineDetector(canny_th1 = umbralCanny, canny_th2 = umbralCanny*3, do_merge=True)

laneSensor = det.LaneSensor(hs, fld, frameSize, hui)
hui.calculateRoi()

annotations = det.SegmentsAnnotator()
zenithalAnnotations = det.SegmentsAnnotator(thickness=2, colorMap=det.SegmentsAnnotator.colorMapBGR)


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

    ret = laneSensor.update(im)
    if not ret:
        continue

    imGray = laneSensor.imGray



    startProcesst = timer()

    # Detección de segmentos y Hough ======================

    perspectives = laneSensor.perspectives
    zenithals = laneSensor.zenithals

    hs.assign2Bins(zenithals)
    hs.computeVotes(zenithals.lengths if userVisualizationOption else perspectives.lengths)   # alternative: zenithals.lengths
    hs.computeAngleHistogram()


    # Brújula, dirección del segmento mayor, aproximación simple a la dirección principal.
    peakZenithalVersor = laneSensor.compass()

    # Carril
    leftLaneDetected, rightLaneDetected, centralLaneAngle, centralLaneDistance = laneSensor.lane()
    laneDetected = leftLaneDetected or rightLaneDetected
    fullLaneDetected = leftLaneDetected and rightLaneDetected

    # Línea de fin de carril
    if laneDetected:
        endOfLaneDetected, endOfLaneDistance, endOfLaneIndex = laneSensor.endOfLane()
    else:
        endOfLaneDetected = False

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
    

    # Líneas de carril
    if leftLaneDetected:
        annotations.drawSegments(imAnnotated, 
                                perspectives.coords[laneSensor.leftLaneSegmentsIndices], 
                                color=(0,128,255), thickness=4)
    if rightLaneDetected:
        annotations.drawSegments(imAnnotated,
                                perspectives.coords[laneSensor.rightLaneSegmentsIndices], 
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
    annotations.drawSegments(imAnnotated, mainAxisSegments, color=(255,255,0))


    # Visualización cenital =====================

    # zenithal fustrum view
    zenithalIm = cv.warpPerspective(im, hui.Hview, tuple(hui.zenithalSize))

    # Red: base origin
    cv.drawMarker(zenithalIm, tuple(zenithals.getPointAsIntTuple(zenithals.referencePoint)), (0,0,255), cv.MARKER_CROSS, 20, 3)

    # Green: main axes origin
    cv.drawMarker(zenithalIm, zenithals.getPointAsIntTuple(zenithalOrigin), (0,255,0), cv.MARKER_CROSS, 20, 3)

    # segments, and message
    colors = np.empty((perspectives.n, 3), dtype=np.uint8)
    colors[hs.angleIndices == laneSensor.mainAngleBin] = np.array([0,255,0])   # main direction
    colors[(hs.angleIndices != laneSensor.mainAngleBin)] = np.array([0,0,128], np.uint8) # other directions
    zenithalAnnotations.drawSegments(zenithalIm, zenithals,
                                    color=colors,
                                    message= f'FLD: {(laneSensor.FLDT)*1000:.0f} ms'
                                    f'\nProcess: {(endProcesst-startProcesst)*1000:.0f} ms'
                                    f'\nSegments {str(len(zenithals.coords))}'
                                    f"\nHough votes: {'zenithals' if userVisualizationOption else 'segments'}"
                                   )
    '''
    # Dibujar todos los segmetnos
    annotations.drawSegments(imAnnotated, perspectives, color=colors)
    '''

    # Líneas de carril aproximadas
    if leftLaneDetected:
        zenithalAnnotations.drawSegments(zenithalIm, 
                                        zenithals.coords[laneSensor.leftLaneSegmentsIndices], 
                                        color=(0,128,255))
    if rightLaneDetected:
        zenithalAnnotations.drawSegments(zenithalIm, 
                                        zenithals.coords[laneSensor.rightLaneSegmentsIndices], 
                                        color=(255,0,128))

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
        if endOfLaneDetected:
            zenithalAnnotations.drawSegments(zenithalIm,
                                            zenithals.coords[endOfLaneIndex].reshape(-1,2,2),
                                            color=(128,128,255), thickness=4)
            annotations.drawSegments(imAnnotated,
                                            perspectives.coords[endOfLaneIndex].reshape(-1,2,2),
                                            color=(128,128,255), thickness=4)


    # Achicar la imagen cenital si es muy grande
    while(zenithalIm.shape[0] > 700):
        zenithalIm = cv.resize(zenithalIm, None, fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)

    houghSpaceColor = hs.pasteVisualization(zenithalIm, scale=4.0, showMax=True)
    vizVertex = np.array(zenithalIm.shape[1::-1])-houghSpaceColor.shape[1::-1]
    angleBinCoords = vizVertex + (-9, 4*laneSensor.maxAngleHistogramIndex+1)
    cv.drawMarker(zenithalIm, tuple(angleBinCoords), (0,0,255), cv.MARKER_SQUARE, 4)

    # Sombra para el texto
    imAnnotated[0:50, 0:150] //= 2

    # Texto
    x=10
    h=20
    color=(255,255,255)
    cv.putText(imAnnotated, f'linea izquierda: {leftLaneDetected}', (x,h), cv.FONT_HERSHEY_SIMPLEX, 0.4, color)
    cv.putText(imAnnotated, f'linea derecha: {rightLaneDetected}', (x,h*2), cv.FONT_HERSHEY_SIMPLEX, 0.4, color)
    if endOfLaneDetected:
        cv.putText(imAnnotated, f'fin de carril: {endOfLaneDistance:.2f}', (x,h*3), cv.FONT_HERSHEY_SIMPLEX, 0.4, color)


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
                #print('foldedAngleHistogram:', foldedAngleHistogram)
                print(f'maxAngleHistogramIndex: {laneSensor.maxAngleHistogramIndex}')
                #print('hs.distanceHistogram:', distanceHistogram)
                #print('foldedDistanceHistogram:', foldedDistanceHistogram)
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
                print('Longitud de línea de fin de carril', perspectives.lengths[endOfLaneIndex] if endOfLaneIndex>-1 else 'N/A')
                '''
                if endOfLaneDetected:
                    for bin in binIndices:
                        print(f'bin: {bin[0]}, {bin[1]}, votos: {hs.houghSpace[bin]}')
                '''
                
            case 's':
                save()
            case 'l':
                load()

# Close and exit
if(fs):
    fs.release()
cv.destroyAllWindows()