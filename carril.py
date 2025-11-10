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


'''
- square side 250px
- lane: about 220px
- lane line: 10px
- angles 0..pi
'''
hs = det.HoughSpace()
annotations = det.SegmentsAnnotator()
cenitalAnnotations = det.SegmentsAnnotator(thickness=2, colorMap=det.SegmentsAnnotator.colorMapBGR)
umbralCanny = 160
printFlag = False
fld = cv.ximgproc.createFastLineDetector(canny_th1 = umbralCanny, canny_th2 = umbralCanny*3, do_merge=True)
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

    lines += [0, hui.limit, 0, hui.limit]
    segments = det.Segments(lines)
    segments.computeLengths()   # used for votes

    zenithals = det.Segments(det.projectSegments(segments, hui.Hview),
                             referencePoint=(hui.zenithalSize[0]//2, hui.zenithalSize[1]))
    #zenithals.setReferencePoint((hui.zenithalSize[0]//2, hui.zenithalSize[1]))
    zenithals.computeAnglesAndDistances()   # Hough variables    
    hs.assign2Bins(zenithals)
    hs.computeVotes(zenithals.lengths if userVisualizationOption else segments.lengths)   # alternative: zenithals.lengths

    # Main axes from max vote
    mainSegmentsIndices = hs.getIndicesFromBin(*hs.maxLoc)
    mainSegmentIndex = mainSegmentsIndices[np.argmax(segments.lengths[mainSegmentsIndices])]
    mainZenithalDelta = zenithals.deltas[mainSegmentIndex]/zenithals.lengths[mainSegmentIndex]


    endProcesst = timer()

    # Annotations & visualization
    startAnnotationt = timer()

    imAnnotated = cv.cvtColor(imGray//2, cv.COLOR_GRAY2BGR)
    annotations.drawSegments(imAnnotated, segments.coords)
    winnerValue = hs.maxVal
    winnerBin = hs.maxLoc
    mainSegmentsIndices = hs.getIndicesFromBin(*winnerBin)
    annotations.drawSegments(imAnnotated, segments.coords[mainSegmentsIndices], color=(0,255,0))
    cv.putText(imAnnotated, f'Winner bin: {winnerBin} value: {winnerValue}', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    # Draw main axis
    origin = (imAnnotated.shape[1]//2, (imAnnotated.shape[0]+hui.limit)//2)
    zenithalOrigin = det.projectSegments(origin, hui.Hview, segmentsShape=False, printFlag=printFlag).reshape(-1)
    zenithalForward = zenithalOrigin - mainZenithalDelta * hs.laneWidth/2
    zenithalSide = zenithalOrigin + np.array((mainZenithalDelta[1], -mainZenithalDelta[0])) * hs.laneWidth/2
    mainAxisZenithalSegments = np.array([zenithalOrigin, zenithalForward, zenithalOrigin, zenithalSide]).reshape(-1, 2, 2)
    mainAxisSegments = det.projectSegments(mainAxisZenithalSegments, hui.Hview, inverse=True, printFlag=printFlag)
    annotations.drawSegments(imAnnotated, mainAxisSegments, color=(255,255,0))


    cv.imshow('Main segments', imAnnotated)

    # zenithal fustrum view
    zenithalIm = cv.warpPerspective(im, hui.Hview, hui.zenithalSize)
    cv.drawMarker(zenithalIm, zenithals.referencePoint.astype(np.int32), (0,0,255), cv.MARKER_CROSS, 20, 3)
    cv.drawMarker(zenithalIm, zenithalOrigin.astype(np.int32), (0,255,0), cv.MARKER_CROSS, 20, 3)
    cenitalAnnotations.drawSegments(zenithalIm, zenithals, #intensities=zenithals.angles/3.17, 
                                    message= f'FLD: {(endFLDt-startFLDt)*1000:.0f} ms'
                                    f'\nProcess: {(endProcesst-startProcesst)*1000:.0f} ms'
                                    f'\nSegments {str(len(zenithals.coords))}'
                                    f"\nHough votes: {'zenithals' if userVisualizationOption else 'segments'}"
                                    f'\nMax votes: {winnerValue:.0f}'
                                   )
    cenitalAnnotations.drawSegments(zenithalIm, zenithals.coords[mainSegmentsIndices], color=(0,255,0))
    cenitalAnnotations.drawSegments(zenithalIm, zenithals.coords[mainSegmentIndex], color=(0,255,255))
    cenitalAnnotations.drawSegments(zenithalIm, mainAxisZenithalSegments, color=(255,255,0))

    # autoshrink
    while(zenithalIm.shape[0] > 700):
        zenithalIm = cv.resize(zenithalIm, None, fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)

    hs.pasteVisualization(zenithalIm, scale=4.0, showMax=True)
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
                play = not play
            case 'v':
                userVisualizationOption += 1
                userVisualizationOption %= 2
            case 'p':
                printFlag = True
                print('image shape:', im.shape)
                print('roiPoly', hui.roiPoly, type(hui.roiPoly), type(hui.roiPoly[0]), type(hui.roiPoly[0][0]))
                print(f"winnerBin: {winnerBin}")
                #print(f"mainSegmentsIndices: {len(mainSegmentsIndices)}")
                print(f'H2 det: {np.linalg.det(hui.Hview)}, \nH2: {hui.Hview}')
                print(f'H2 inv: {np.linalg.inv(hui.Hview)}')
                print(f'origin: {origin}')
                print(f'zenithalOrigin: {zenithalOrigin}')
                print(f'zenithalForward: {zenithalForward}')
                print(f'zenithalSide: {zenithalSide}')
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
                
            case 's':
                save()
            case 'l':
                load()

# Close and exit
if(fs):
    fs.release()
cv.destroyAllWindows()