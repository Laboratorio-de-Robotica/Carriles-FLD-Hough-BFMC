import numpy as np
import cv2 as cv
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", help="flujo de video de entrada: la ruta de un archivo de video o el número de cámara", default=0)
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
anchoImagen = video.get(cv.CAP_PROP_FRAME_WIDTH)
altoImagen = video.get(cv.CAP_PROP_FRAME_HEIGHT)
print('Resolución del video:', anchoImagen, altoImagen)

alturaObjetivo = 480
cenitalLado = 500
cenitalTamano = (cenitalLado,cenitalLado)


if(altoImagen > alturaObjetivo):
    anchoObjetivo = int(anchoImagen/altoImagen * alturaObjetivo)
    video.set(cv.CAP_PROP_FRAME_WIDTH,  anchoObjetivo)
    video.set(cv.CAP_PROP_FRAME_HEIGHT, alturaObjetivo)

_, im = video.read()
(altoImagen, anchoImagen, canales) = im.shape
print('Nueva resolución del video:', anchoImagen, altoImagen, type(anchoImagen))
tamanoObjetivo = None
if(altoImagen > alturaObjetivo):
    altoImagen = alturaObjetivo
    anchoImagen = anchoObjetivo
    tamanoObjetivo = (int(anchoImagen), int(altoImagen))

print('Resolución de la imagen ajustada:', anchoImagen, altoImagen, type(anchoImagen))

# GUI
def mouse(event,x,y,flags,param):
    global comando
    if(event == cv.EVENT_LBUTTONDOWN):
        if(comando == Comando.HORIZONTAL):
            global horizonte
            horizonte = int(y)
            calcularRoi()
        elif(comando == Comando.TOPE):
            global tope
            tope = int(y)
            calcularRoi()

cv.namedWindow('video')
cv.setMouseCallback('video', mouse)

class Comando:
    HORIZONTAL = 'h'
    TOPE = 't'
    VACIO = ''

comando = ''

horizonte = int(altoImagen * 0.5)
tope = int(altoImagen * 0.75)

def dibujarLineaHorizontal(im, y, texto, seleccionada=False):
    y = int(y)
    cv.putText(im, texto, (0, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255,255,255))
    if seleccionada:
        # Línea seleccionada
        cv.line(imAnotada, (0, y), (anchoImagen, y), (255,255,255), 2)
    else:
        cv.line(imAnotada, (0, y), (anchoImagen, y), (128,128,128))

    return

# ROI
# Cuadrilátero, comenzando por vértice superior izquierdo, en sentido horario
# Si se proporciona una homografía, la usa para obtener horizonte y tope
def calcularRoi(H_ = None):
    global roiPoly, fuga, H

    if(H_ is not None):
        # Homografía suministrada, se recalculan horizonte y tope
        global horizonte, tope
        H = H_.astype(np.float32)
        puntosClave = np.array(((0,-1,0),(0,0,1)), np.float32)
        print('puntosClave', puntosClave)
        puntosEnPerspectiva = np.matmul(np.linalg.inv(H.astype(np.float32)), puntosClave.T).T
        print('puntosEnPerspectiva', puntosEnPerspectiva)
        horizonte = int(puntosEnPerspectiva[0,1]/puntosEnPerspectiva[0,2])
        tope      = int(puntosEnPerspectiva[1,1]/puntosEnPerspectiva[1,2])
        print('horizonte y tope', horizonte, tope)

    medio = int(anchoImagen/2)
    fuga = (medio, horizonte)
    xProyectado = int(medio*(altoImagen-tope)/(altoImagen-horizonte))
    roiVertices = np.array([
        [xProyectado,tope],
        [anchoImagen-xProyectado,tope],
        [anchoImagen,altoImagen],
        [0,altoImagen]
    ], np.int32)
    # polylines int32, shape: (n,1,2): n, -, (x,y)
    roiPoly = roiVertices.reshape((-1,1,2))

    cenitalVertices = np.array([
        [0,0],
        [cenitalLado,0],
        [cenitalLado,cenitalLado],
        [0,cenitalLado]
    ], np.float32)

    if(H_ is None):
        H = cv.getPerspectiveTransform(roiVertices.astype(np.float32), cenitalVertices)
        print('Homografía:\n', H, '\n')


calcularRoi()

umbralCanny = 160
def umbralCannyTrackbar(valor):
    global umbralCanny, detector
    umbralCanny = valor
    detector = cv.ximgproc.createFastLineDetector(canny_th1 = umbralCanny, canny_th2 = umbralCanny*3)


#cv.namedWindow('Canny')
#cv.createTrackbar('umbral', 'Canny' , 0, 255, umbralCannyTrackbar)

umbralBinario = 160
def umbralBinarioTrackbar(valor):
    global umbralBinario
    umbralBinario = valor

cv.namedWindow('Binario')
cv.createTrackbar('umbral', 'Binario' , 0, 255, umbralBinarioTrackbar)

cv.namedWindow('Hough', cv.WINDOW_NORMAL)

# Carga la matriz desde yaml, y actualiza la anotación.  El nombre del archivo está en args.load. Si el archivo no existe no hace nada.
def load():
    fs = cv.FileStorage(args.load, cv.FILE_STORAGE_READ)
    if(fs.isOpened()):
        global H
        H = fs.getNode('H').mat()
        calcularRoi(H)
        fs.release()

# Guarda la matriz en yaml.  El nombre del archivo está en args.save.  Si el archivo no existe lo crea.
def save():
    fs = cv.FileStorage(args.save, cv.FILE_STORAGE_WRITE)
    fs.write('H', H)
    fs.release()


'''
Recibe segmentos detectados por Fast Line Detector: shape (n,1,4), dtype float32
Cada segmento tiene este formato: [x1,y1,x2,y2]
Se procesan todos los segmentos, cualquier ajuste a ROI se debe realizar antes.
'''
base = cenitalTamano[1]
centro = np.array([base//2, 0], dtype=np.float32)   # coordenadas cenitales
histoBins = (20,20)
histoRange = np.array([[-np.pi/2,np.pi/2],[-2*base, 3*base]]) # coordenadas cenitales
#histoRange2 = histoRange + np.array([[np.pi/40], [base/8]])
#print('histoRange, histoRange2', histoRange, histoRange2)
def analizarSegmentos(segmentos):
    n = segmentos.shape[0]
    print('n', n)
    #print('n, segmentos.dtype, segmentos.shape, type(segmentos)', n, segmentos.dtype, segmentos.shape, type(segmentos))

    puntos = segmentos.reshape(-1, 2)           # Shape 2n,2: float32 #segmento*2, (x,y)
    ones = np.ones((n*2,1), np.float32)         # Shape 2n,1
    puntosHomogeneos = np.concatenate((puntos, ones), axis=1)       # Shape 2n,3
    puntosProyectadosHomogeneos = puntosHomogeneos @ H.transpose()  # Shape 2n,3

    # Normalización y reducción dimensional
    segmentosProyectados = (puntosProyectadosHomogeneos[:,:2]/puntosProyectadosHomogeneos[:,2:]).reshape((-1,2,2)) # Shape n,2,2

    imCenital = np.zeros(cenitalTamano, np.uint8)
    for segmento in segmentosProyectados:
        pt0 = segmento[0]
        pt1 = segmento[1]
        if(abs(pt0[1])>1e6 or abs(pt1[1])>1e6):
            continue
        cv.line(imCenital, segmento[0].astype(np.int32), segmento[1].astype(np.int32), 255)
        
    cv.imshow('Cenital', imCenital)

    deltas = segmentosProyectados[:,0,:] - segmentosProyectados[:,1,:]    # Shape n,2, formato #segmento, (dx,dy)
    longitudes = np.linalg.norm(deltas, axis=1)                     # Shape n

    # Longitud en píxeles sobre el segmento original, no cenital, usado luego para la ponderación en el histograma
    pixeles = np.linalg.norm(segmentos[:,:,:2]-segmentos[:,:,2:], axis=2).flatten()     # Shape n

    # Ángulo para Hough, en radianes, atan2 usa (y,x), por conveniencia invierto los ejes y 0 es la vertical
    angulos = np.arctan2(deltas[:,0], deltas[:,1])
    #print('angulos', angulos)

    print('centro', centro)
    print('puntosProyectados.shape, centro.shape', segmentosProyectados.shape, centro.shape)
    distancias = np.sum((deltas[:,::-1]/longitudes[:,None]) * (segmentosProyectados[:,0,:] - centro), axis=1)

    # Pendiente inversa: x/y, es infinito para horizontal, que es la que menos interesa
    # Shape n
    #pendientes = np.where(deltas[:,1], deltas[:,0] / deltas[:,1], np.inf)

    # intersección de la recta en el eje horizontal base
    #intersecciones = np.where(pendientes==np.inf, np.inf, puntosProyectados[:,0,1]-puntosProyectados[:,0,0]*pendientes)

    '''
    pendientesDiscretas = np.digitize(pendientes, range(-1.0, 1.0, 0.2))
    base = cenitalTamano[1]
    interseccionesDiscretas = np.digitize(intersecciones, range(-base, 2*base, base/10))
    '''

    #hist, _, _ = np.histogram2d(pendientes, intersecciones, bins=(50,50), range=[[-2.0,2.0],[-base, 2*base]], weights=longitudes)
    hist , xedges, yedges = np.histogram2d(angulos, distancias, weights=pixeles, bins=histoBins, range=histoRange)
    histCombinado = hist
    #histCombinado = hist[:-1, :-1] + hist[1:, :-1] + hist[:-1, 1:] + hist[1:, 1:]

    maxLoc = np.unravel_index(np.argmax(histCombinado), histCombinado.shape)
    maxVal = histCombinado[maxLoc[0], maxLoc[1]]
    print('maxVal', maxVal)

    top_two_indices = np.argpartition(hist.flatten(), -2)[-2:]
    top_two_coords = np.unravel_index(top_two_indices, hist.shape)
    


    histGray = (histCombinado * 255/maxVal) if maxVal>0 else histCombinado
    histColor = cv.applyColorMap(histGray.astype(np.uint8), cv.COLORMAP_DEEPGREEN)
    cv.imshow('Hough', histColor)

load()

detector = cv.ximgproc.createFastLineDetector(canny_th1 = umbralCanny, canny_th2 = umbralCanny*3, do_merge=True)
while(True):
    if(play):
        _, im = video.read()
        if(tamanoObjetivo):
            im = cv.resize(im, tamanoObjetivo)

    imGris = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    # Vista cenital
    cenitalIm = cv.warpPerspective(im, H, cenitalTamano)
    cv.imshow('cenital', cenitalIm)


    # Fast Line Detector - ROI: horizonte hacia abajo
    lineas = detector.detect(imGris[horizonte:])
    lineas += [0, horizonte, 0, horizonte]
    analizarSegmentos(lineas)
    
    # Anotaciones
    imAnotada = im.copy()
    dibujarLineaHorizontal(imAnotada, horizonte, 'horizonte', comando == Comando.HORIZONTAL)
    dibujarLineaHorizontal(imAnotada, tope, 'tope', comando == Comando.TOPE)
    cv.line(imAnotada, roiPoly[3,0], fuga, (128,128,128))
    cv.line(imAnotada, roiPoly[2,0], fuga, (128,128,128))
    cv.polylines(imAnotada, [roiPoly], True, (0,255,0))
    cv.imshow('video', imAnotada)

    # Canny
    #imCanny = cv.Canny(imGris, umbralCanny, umbralCanny*3)
    #cv.imshow('Canny', imCanny)

    # Adaptativo: muy lento
    #imBinaria = cv.adaptiveThreshold(imGris, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, 0)
    #cv.imshow('Adaptativo', imBinaria)

    # Thinning
    imBinaria = cv.inRange(imGris, umbralBinario, 255)
    #imSkeleton = cv.ximgproc.thinning(imBinaria)
    #cv.imshow('Skeleton', imSkeleton)

    # Anotación de segmentos
    imBinaria = imBinaria // 2
    imFLD = detector.drawSegments(imBinaria, lineas)
    cv.imshow('Binario', imFLD)


    tecla = cv.waitKey(30)
    match tecla:
        case -1:
            continue
        case 27:
            break

    match chr(tecla):
        case ' ':
            play = not play
        case 'h':
            # Ajustar horizonte
            if(comando == Comando.HORIZONTAL):
                comando = ''
            else:
                comando = Comando.HORIZONTAL
        case 't':
            # Ajustar límite
            if(comando == Comando.TOPE):
                comando = ''
            else:
                comando = Comando.TOPE
        case 'p':
            print(roiPoly, type(roiPoly), type(roiPoly[0]), type(roiPoly[0][0]))
        
        case 's':
            save()
        case 'l':
            load()


if(fs):
    fs.release()

cv.destroyAllWindows()