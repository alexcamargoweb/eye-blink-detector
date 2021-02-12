# Eye blink detector - https://github.com/alexcamargoweb/eye-blink-detector
# Detecção de piscadas em olhos humanos com Python, OpenCV e dlib.
# Referência: Adrian Rosebrock, Eye blink detection with OpenCV, Python, and dlib. PyImageSearch.
# Disponível em: https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/.
# Acessado em: 25/01/2021.
# Arquivo: blink_detector.py
# Execução via PyCharm/Linux (Python 3.8)

# importa os pacotes necessários
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2

# função que calcula a proporção dos olhos
def eye_aspect_ratio(eye):
    # calcula as distâncias Euclidianas entre os dois
    # conjuntos verticais de coordenadas (x e y)
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # calcula a distância Euclidiana entre a marcação horizontal do olho (x e y)
    C = dist.euclidean(eye[0], eye[3])
    # calcula a proporção do olho
    ear = (A + B) / (2.0 * C)
    # retorna a proporção do olho
    return ear

# detector dlib pré-treinado para partes de um rosto
shape_predictor = 'predictor/shape_predictor_68_face_landmarks.dat'

# define duas constantes, uma para a proporção do olho indicando
# a piscada, e uma segunda constante para o número de
# frames que (o olho) deve estar abaixo do limite
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 2
# inicializa os contadores de quadros e o número total de piscadas
COUNTER = 0
TOTAL = 0

# inicializa o detector de rosto dlib (baseado em HOG)
detector = dlib.get_frontal_face_detector()
# em seguida, cria o preditor das partes do rosto
predictor = dlib.shape_predictor(shape_predictor)

# pega os índices dos pontos de referência faciais para
# o olho esquerdo e direito, respectivamente
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# inicializa o stream de vídeo
vs = VideoStream(0).start()
time.sleep(2.0)

# faz um loop sobre os frames do vídeo
while True:
    # carrega o frame, redimensiona e converte para uma escala de cinza
    frame = vs.read()
    frame = imutils.resize(frame, width = 450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detecta as faces no frame em escala de cinza
    rects = detector(gray, 0)

    # faz um loop nos rostos detectados
    for rect in rects:
        # determina os pontos de referência faciais para a região do rosto
        shape = predictor(gray, rect)
        # converte as coordenadas (x, y) do ponto de referência facial em uma matriz NumPy
        shape = face_utils.shape_to_np(shape)
        # extrai as coordenadas do olho esquerdo e direito e
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        # usa as coordenadas para calcular a proporção do olho para ambos os olhos
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # calcula a média da proporção dos olhos (juntos) para ambos os olhos
        ear = (leftEAR + rightEAR) / 2.0

        # calcula o casco convexo para o olho esquerdo e direito, então
        # visualiza cada um dos olhos
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # verifica se a proporção do olho está abaixo do limite de piscar
        # em caso afirmativo, incrementa o contador de quadros intermitente
        if ear < EYE_AR_THRESH:
            COUNTER += 1
        # caso contrário, a proporção do olho não fica abaixo do limite de piscar
        else:
            # se os olhos estiverem fechados por um número suficiente,
            # aumenta o número total de piscadas
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            # zera o contador do frame do olho
            COUNTER = 0

        # mostra o número total de piscadas junto com
        # a proporção do olho calculada para o frame
        cv2.putText(frame, "Piscadas: {}".format(TOTAL), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # mostra o frame de vídeo
    cv2.imshow("Detector de piscadas", frame)
    key = cv2.waitKey(1) & 0xFF

    # se a tecla 'q' for pressionada, fecha o programa
    if key == ord("q"):
        break

# limpa a execução
cv2.destroyAllWindows()
vs.stop()