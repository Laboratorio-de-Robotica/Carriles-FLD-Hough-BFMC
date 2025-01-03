import numpy as np
import cv2 as cv
import detector as det
import HUI
import argparse
from timeit import default_timer as timer

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

# Util
play = True # bandera de control play/pausa
fs = None   # yaml

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

# Carga la matriz desde yaml, y actualiza la anotación.  El nombre del archivo está en args.load. Si el archivo no existe no hace nada.
def load():
    fs = cv.FileStorage(args.load, cv.FILE_STORAGE_READ)
    if(fs.isOpened()):
        #global H, hui
        hui.H = fs.getNode('H').mat()
        hui.calculateRoi(hui.H)
        fs.release()

# Guarda la matriz en yaml.  El nombre del archivo está en args.save.  Si el archivo no existe lo crea.
def save():
    fs = cv.FileStorage(args.save, cv.FILE_STORAGE_WRITE)
    fs.write('H', hui.H)
    fs.release()


hui = HUI.Hui('hui', frameSize)
load()
hui.zenithalSquareSide = 250
hui.calculateRoi()

# Bucle principal


'''
Bins:
- square side 250px
- lane: about 220px
- lane line: 10px
- angles 0..pi: 10 bins
- distances -500..+500: 20 bins
'''
bins = det.Bins(maxDistance=1500, binsSizes=(11,41), verbose=True)
annotations = det.SegmentsAnnotator()
cenitalAnnotations = det.SegmentsAnnotator(thickness=2, colorMap=det.SegmentsAnnotator.colorMapBGR)
umbralCanny = 160
fld = cv.ximgproc.createFastLineDetector(canny_th1 = umbralCanny, canny_th2 = umbralCanny*3, do_merge=True)
while(True):
    # video feed
    if(play):
        _, im = video.read()
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

    lines += [0, hui.limit, 0, hui.limit]
    segments = det.Segments(lines)
    segments.computeLengths()   # used for votes

    zenithals = det.zenithalSegmentsFactory(segments, hui.H2)
    zenithals.setReferencePoint((hui.zenithalSize[0]//2, hui.zenithalSize[1]))
    zenithals.computeAnglesAndDistances()   # Hough variables
    
    bins.assignToBins(zenithals)
    houghSpacePerspectiveWeigthed = bins.makeHoughSpace(segments.lengths, 'Hough space perspective weigth')
    #houghSpaceZenithalWeigthed = bins.makeHoughSpace(zenithals.lengths, 'Hough space zenithal weigth')
    endProcesst = timer()


    startAnnotationt = timer()
    #houghSpacePerspectiveWeigthed.show(showMax=True)
    #houghSpaceZenithalWeigthed.show(showMax=True)

    imAnnotated = cv.cvtColor(imGray//2, cv.COLOR_GRAY2BGR)
    annotations.drawSegments(imAnnotated, segments.coords, color=(0,0,255))
    winnerValue = houghSpacePerspectiveWeigthed.maxVal
    winnerBin = houghSpacePerspectiveWeigthed.maxLoc
    mainSegmentsIndices = bins.getIndicesFromBin(winnerBin)
    annotations.drawSegments(imAnnotated, segments.coords[mainSegmentsIndices])
    cv.putText(imAnnotated, f'Winner bin: {winnerBin} value: {winnerValue}', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    cv.imshow('Main segments', imAnnotated)

    # zenithal fustrum view
    zenithalIm2 = cv.warpPerspective(im, hui.H2, hui.zenithalSize)
    cv.drawMarker(zenithalIm2, zenithals.referencePoint.astype(np.int32), (0,0,255), cv.MARKER_CROSS, 20, 2)
    cenitalAnnotations.drawSegments(zenithalIm2, zenithals, intensities=zenithals.angles/3.17, 
                                    message='Segments '+str(len(zenithals.coords))),

    # autoshrink
    while(zenithalIm2.shape[0] > 700):
        zenithalIm2 = cv.resize(zenithalIm2, None, fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)

    houghSpacePerspectiveWeigthed.pasteVisualization(zenithalIm2, scale=4.0, showMax=True)

    cv.imshow('zenithal wide', zenithalIm2)

    endAnnotationt = timer()

    #print(f'FLD: {endFLDt-startFLDt:.3f} s, Process: {endProcesst-startProcesst:.3f} s, Annotate: {endAnnotationt-startAnnotationt:.3f} s')

    # user keyboard
    key = cv.waitKey(30)
    if(key != -1):
        match key:
            case 27:
                break

        hui.processKey(key)

        match chr(key):
            case ' ':
                play = not play
            case 'p':
                print('image shape:', im.shape)
                print('roiPoly', hui.roiPoly, type(hui.roiPoly), type(hui.roiPoly[0]), type(hui.roiPoly[0][0]))
                print(f"winnerBin: {winnerBin}")
                print(f"mainSegmentsIndices: {len(mainSegmentsIndices)}")
                print(*mainSegmentsIndices)
                print('houghSpacePerspectiveWeigthed.houghSpace:')
                print(houghSpacePerspectiveWeigthed.houghSpace)
                print('angleBinsIndices, distanceBinsIndices:')
                '''
                for a,b,c,d in zip(bins.angleBinsIndices, bins.distanceBinsIndices, zenithals.angles, zenithals.distances):
                    print(a,b,c,d)
                '''
                
            case 's':
                save()
            case 'l':
                load()





if(fs):
    fs.release()

cv.destroyAllWindows()