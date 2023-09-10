import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from usefunc import move, x_rotation, y_rotation, z_rotation
from math import pi,cos,sin

video = cv2.VideoCapture(0)

hand = mp.solutions.hands
Hand = hand.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils
rot_angle_x = 0
rot_angle_y = 0
rot_angle_z = 0



while True:
    check, img = video.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Hand.process(imgRGB)
    handPoints = results.multi_hand_landmarks
    h,w,d = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    pontos = []
    

    if handPoints:
        hand_plot = np.array([[0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0]])

        for points in handPoints:
            mpDraw.draw_landmarks(img,points,hand.HAND_CONNECTIONS)
            for id, cord in enumerate(points.landmark):
                cx,cy,cz = int(cord.x*w), int(cord.y*h), int(cord.z*d)
                cv2.putText(img, str(id), (cx,cy+10), font, 0.3, (0,0,0), 2)
                pontos.append((cx,cy))
                hand_plot[id][0] = cord.x*w
                hand_plot[id][1] = cord.y*h
                hand_plot[id][2] = cord.z*d
        
        hand_plot = np.transpose(hand_plot)

        num_columns = np.size(hand_plot,1)
        ones_line = np.ones(num_columns)

        hand_plot = np.vstack([hand_plot, ones_line])
        
        T = move(-hand_plot[0][0], -hand_plot[1][0], -hand_plot[2][0])
        hand_plot = np.dot(T,hand_plot)
        
        Ry = y_rotation(rot_angle_y)
        Rx = x_rotation(rot_angle_x)
        Rz = z_rotation(rot_angle_z)
        hand_plot = Rz@Ry@Rx@hand_plot
        
        print(hand_plot)

        figure = plt.figure(figsize=(100,100))
        ax0 = plt.axes(projection='3d')
        ax0.scatter(hand_plot[0,:], hand_plot[1,:], hand_plot[2,:], c='red', marker='o')
        ax0.set_xlabel("X")
        ax0.set_ylabel("Y")
        ax0.set_zlabel("Z")
        plt.show()
    

    cv2.imshow("hands", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




