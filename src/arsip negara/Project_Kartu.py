# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 14:27:28 2025

@author: User
"""


import cv2
import numpy as np

cam = cv2.VideoCapture(0)
bg = cv2.imread("../img/kartu.png")
cv2.namedWindow("Mikel suka kartu")

def nothing():
    pass

cv2.createTrackbar("H Low", "Mikel suka kartu", 0, 180, nothing)
cv2.createTrackbar("H High", "Mikel suka kartu", 180, 180, nothing)
cv2.createTrackbar("S Low", "Mikel suka kartu", 0, 255, nothing)
cv2.createTrackbar("S High", "Mikel suka kartu", 255, 255, nothing)
cv2.createTrackbar("V Low", "Mikel suka kartu", 0, 255, nothing)
cv2.createTrackbar("V High", "Mikel suka kartu", 255, 255, nothing)

ret, frame = cam.read()
if not ret:
    print("Failed to grab frame from webcam.")
    cam.release()
    exit()

# Resize background to match webcam frame size
bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]))


while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow("Citra Asli", frame)
    key = cv2.waitKey(1) & 0xFF

    citra_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow("Citra HSV", citra_HSV)

    hLow = cv2.getTrackbarPos("H Low", "Mikel suka kartu")
    hHigh = cv2.getTrackbarPos("H High", "Mikel suka kartu")
    sLow = cv2.getTrackbarPos("S Low", "Mikel suka kartu")
    sHigh = cv2.getTrackbarPos("S High", "Mikel suka kartu")
    vLow = cv2.getTrackbarPos("V Low", "Mikel suka kartu")
    vHigh = cv2.getTrackbarPos("V High", "Mikel suka kartu")

    lower_ijo = np.array([hLow, sLow, vLow])
    upper_ijo = np.array([hHigh, sHigh, vHigh])

    mask_ijo = cv2.inRange(citra_HSV, lower_ijo, upper_ijo)
    mask_ijo_terbalik = cv2.bitwise_not(mask_ijo)

    cv2.imshow("Topeng ijo", mask_ijo)
    cv2.imshow("Topeng ijo terbalik", mask_ijo_terbalik)

    fg = cv2.bitwise_and(frame, frame, mask=mask_ijo_terbalik)
    bg_jadi = cv2.bitwise_and(bg, bg, mask=mask_ijo)

    hasil_akhir = cv2.add(fg, bg_jadi)
    cv2.imshow("Mikel suka kartu", hasil_akhir)

    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()