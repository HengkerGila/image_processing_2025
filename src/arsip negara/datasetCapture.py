import cv2
import numpy as np
import os

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def nothing(x):
    pass

cap = cv2.VideoCapture(0)

kernel = np.ones((5,5),np.uint8)

# Sliders
cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 60, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 30, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 40, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 100, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

# Dataset Config
DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

# Label
CURRENT_LABEL = "Q_club"  
SAVE_PATH = os.path.join(DATASET_DIR, CURRENT_LABEL)
os.makedirs(SAVE_PATH, exist_ok=True)

while True:
    ret, img = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")
    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")

    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])

    mask_green = cv2.inRange(hsv, lower_bound, upper_bound)
    mask_cards = cv2.bitwise_not(mask_green)

    mask_opened = cv2.morphologyEx(mask_cards, cv2.MORPH_OPEN, kernel)
    mask_opclo = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)

    res_opclo = cv2.bitwise_and(img, img, mask=mask_opclo)

    contours, _ = cv2.findContours(mask_opclo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    card_value = img.copy()
    grayscale_value_mask = img.copy()

    c_contour = []
    for contour in contours:
        if cv2.contourArea(contour) > 3000:
            c_contour.append(contour)

    if c_contour:
        for contour in c_contour: 
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                src_pts = order_points(approx.reshape(4, 2))
                width, height = 200, 300
                dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
                matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warped = cv2.warpPerspective(img, matrix, (width, height))
                cv2.imshow("Extracted Card", warped)

            cv2.drawContours(card_value, [contour], -1, (0, 255, 0), 2)
    
    cv2.imshow('Mask', mask_opclo)
    cv2.imshow('Opened-Closed', cv2.flip(res_opclo, 1))
    cv2.imshow('Nilai', cv2.flip(card_value, 1))
    cv2.imshow('Value Mask', cv2.flip(grayscale_value_mask, 1))

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('s') and warped is not None:
        count = len(os.listdir(SAVE_PATH))
        filename = f"img_{count+1:04d}.jpg"
        filepath = os.path.join(SAVE_PATH, filename)
        cv2.imwrite(filepath, warped)
        print(f"[SAVED] {filepath}")

cap.release()
cv2.destroyAllWindows()