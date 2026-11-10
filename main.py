import cv2
import mediapipe as mp
import numpy as np


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Камера жок")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=5,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils
draw_frame = np.zeros((480, 640, 3), dtype=np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Кадр жок")
        exit()

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for i in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, i, mp_hands.HAND_CONNECTIONS)
            # h, w, c = frame.shape
            # x = int(i.landmark[0].x * w)
            # y = int(i.landmark[0].y * h)
            #
            # cv2.circle(frame, (x, y), 15, (0, 255, 0), cv2.FILLED)

            if i.landmark[3].y > i.landmark[4].y:
                cv2.putText(frame, "Paper", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            elif i.landmark[19].y > i.landmark[20].y:
                cv2.putText(frame, "Paper", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            elif i.landmark[8].x < i.landmark[7].x:
                cv2.putText(frame, "Scissors", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            elif i.landmark[11].x > i.landmark[12].x:
                cv2.putText(frame, "Scissors", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            elif i.landmark[3].x > i.landmark[4].x:
                cv2.putText(frame, "Rock", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            elif i.landmark[7].x < i.landmark[8].x:
                cv2.putText(frame, "Rock", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)



    image = cv2.add(frame, draw_frame)

    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

