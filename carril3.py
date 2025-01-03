import numpy as np
import cv2 as cv
import detector as det
import HUI
import argparse

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

hui = HUI.Hui('hui', frameSize)
hui.calculateRoi()


# Carga la matriz desde yaml, y actualiza la anotación.  El nombre del archivo está en args.load. Si el archivo no existe no hace nada.
def load():
    fs = cv.FileStorage(args.load, cv.FILE_STORAGE_READ)
    if(fs.isOpened()):
        global H
        H = fs.getNode('H').mat()
        hui.calculateRoi(H)
        fs.release()

# Guarda la matriz en yaml.  El nombre del archivo está en args.save.  Si el archivo no existe lo crea.
def save():
    fs = cv.FileStorage(args.save, cv.FILE_STORAGE_WRITE)
    fs.write('H', H)
    fs.release()

load()


# Bucle principal

euclidean = np.array([
    [-1.0, 0.0, 250.0],
    [0.0, -1.0, 500.0],
    [0.0,  0.0,   1.0]
], np.float32)

bins = det.Bins(maxDistance=1500, verbose=True)
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
    lines = fld.detect(imGray[hui.limit:])
    if(lines is None):
        # no lines, skip
        continue

    lines += [0, hui.limit, 0, hui.limit]
    segments = det.Segments(lines)
    segments.computeLengths()   # used for votes

    zenithals = det.zenithalSegmentsFactory(segments, H @ euclidean)
    zenithals.computeAnglesAndDistances()   # Hough variables
    
    bins.assignToBins(zenithals)
    houghSpacePerspectiveWeigthed = bins.makeHoughSpace(segments.lengths, 'Hough space perspective weigth')
    houghSpacePerspectiveWeigthed.show(showMax=True)
    
    #houghSpaceZenithalWeigthed = bins.makeHoughSpace(zenithals.lengths, 'Hough space zenithal weigth')
    #houghSpaceZenithalWeigthed.show(showMax=True)

    imAnnotated = cv.cvtColor(imGray//2, cv.COLOR_GRAY2BGR)
    det.drawSegments(imAnnotated, segments.coords, color=(0,0,255))
    winnerValue = houghSpacePerspectiveWeigthed.maxVal
    winnerBin = houghSpacePerspectiveWeigthed.maxLoc
    mainSegmentsIndices = bins.getIndicesFromBin(winnerBin)
    det.drawSegments(imAnnotated, segments.coords[mainSegmentsIndices])
    cv.putText(imAnnotated, f'Winner bin: {winnerBin} value: {winnerValue}', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    cv.imshow('Main segments', imAnnotated)

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
                print('roiPoly', hui.roiPoly, type(hui.roiPoly), type(hui.roiPoly[0]), type(hui.roiPoly[0][0]))
                print(f"winnerBin: {winnerBin}")
                print(f"mainSegmentsIndices: {len(mainSegmentsIndices)}")
                print(*mainSegmentsIndices)
                print('houghSpacePerspectiveWeigthed.houghSpace:')
                print(houghSpacePerspectiveWeigthed.houghSpace)
                print('angleBinsIndices, distanceBinsIndices:')
                for a,b,c,d in zip(bins.angleBinsIndices, bins.distanceBinsIndices, zenithals.angles, zenithals.distances):
                    print(a,b,c,d)
                
            case 's':
                save()
            case 'l':
                load()





if(fs):
    fs.release()

cv.destroyAllWindows()