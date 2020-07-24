# Código baseado no artigo do Adrian Rosebrock
# https://bit.ly/2CYC7Gf

# importar pacotes necessários
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import imutils
import time
import dlib
import cv2
import matplotlib.pyplot as plt


# definir constantes
ALARM = "alarm.wav"
WEBCAM = 1
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 40
COUNTER = 0
ALARM_ON = False


def sound_alarm(path=ALARM):
    # play an alarm sound
    playsound.playsound(ALARM)


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


# dlib's face detector (HOG-based)
print("[INFO] carregando o preditor de landmark...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# pegar os índices do previsor, para olhos esquerdo e direito
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# inicializar vídeo
print("[INFO] inicializando streaming de vídeo...")
vs = VideoStream(src=WEBCAM).start()
time.sleep(1.0)

# desenhar um objeto do tipo figure
y = [None] * 100
x = np.arange(0,100)
fig = plt.figure()
ax = fig.add_subplot(111)
li, = ax.plot(x, y)

# loop sobre os frames do vídeo
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detectar faces (grayscale)
    rects = detector(gray, 0)

    # loop nas detecções de faces
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extrair coordenadas dos olhos e calcular a proporção de abertura
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # ratio média para os dois olhos
        ear = (leftEAR + rightEAR) / 2.0

        # convex hull para os olhos
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # salvar historico para plot
        y.pop(0)
        y.append(ear)

        # update canvas
        plt.xlim([0, 100])
        plt.ylim([0, 0.4])
        ax.relim()
        ax.autoscale_view(True, True, True)
        fig.canvas.draw()
        plt.show(block=False)
        li.set_ydata(y)
        fig.canvas.draw()
        time.sleep(0.01)

        # checar ratio x threshold
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            # dentro dos critérios, soar o alarme
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # ligar alarme
                if not ALARM_ON:
                    ALARM_ON = True
                    t = Thread(target=sound_alarm)
                    t.deamon = True
                    t.start()

                cv2.putText(frame, "[ALERTA] FADIGA!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # caso acima do threshold, resetar o contador e desligar o alarme
        else:
            COUNTER = 0
            ALARM_ON = False

            # desenhar a proporção de abertura dos olhos
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # tecla para sair do script "q"
    if key == ord("q"):
        break

# clean
cv2.destroyAllWindows()
vs.stop()
