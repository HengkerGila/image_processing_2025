import cv2
import numpy as np
import keyboard as kb

def nothing(x):
    pass

img = cv2.imread('../img/kartuEdit.png')
h, w, d = img.shape
kernel = np.ones((5, 5),np.uint8)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
'''
cv2.namedWindow("Slider")
cv2.createTrackbar("LH", "Slider", 60, 255, nothing)
cv2.createTrackbar("LS", "Slider", 30, 255, nothing)
cv2.createTrackbar("LV", "Slider", 40, 255, nothing)
cv2.createTrackbar("UH", "Slider", 100, 255, nothing)
cv2.createTrackbar("US", "Slider", 255, 255, nothing)
cv2.createTrackbar("UV", "Slider", 255, 255, nothing)
'''

def boundary_extraction(img):
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    eroded = cv2.erode(img, kernel)
    boundary = cv2.subtract(img, eroded)
    return boundary

bound_ex = boundary_extraction(img)
'''
l_h = cv2.getTrackbarPos("LH", "Slider")
l_s = cv2.getTrackbarPos("LS", "Slider")
l_v = cv2.getTrackbarPos("LV", "Slider")

u_h = cv2.getTrackbarPos("UH", "Slider")
u_s = cv2.getTrackbarPos("US", "Slider")
u_v = cv2.getTrackbarPos("UV", "Slider")
'''

lower_bound = np.array((60, 30, 40))
upper_bound = np.array((100, 255, 255))
#lower_bound = np.array([l_h, l_s, l_v])
#upper_bound = np.array([u_h, u_s, u_v])
mask = cv2.inRange(hsv, lower_bound, upper_bound)
no_background = cv2.bitwise_or(img, img, mask=mask)

mask_cards = cv2.bitwise_not(mask)
mask_eroded = cv2.erode(mask_cards, kernel, iterations=1)
mask_dilated = cv2.dilate(mask_eroded, kernel, iterations=1)
card_only = cv2.bitwise_and(img, img, mask=mask_dilated)

output = cv2.connectedComponentsWithStats(mask_dilated)
labelCount, labeled, stat, centroid = output

labelList = list()
labelIndexList = list()
segmentedImage = list()
lbM = np.zeros(shape=(h,w))

for i in range(1, labelCount):
    count = 0
    area = stat[i, cv2.CC_STAT_AREA]
    if area > 300:
        maskLabel = np.zeros(shape=mask_cards.shape, dtype=np.float32)
        label = np.float32(labeled==i)*1
        maskLabel = cv2.bitwise_or(maskLabel, label)
        maskLabel = np.uint8(maskLabel)*255
        segmented = cv2.bitwise_and(img, img, mask=maskLabel)
        labelList.append(label)
        labelIndexList.append(i)
        segmentedImage.append(segmented)

for i in range(len(labelList)):
    lb = labelList[i]
    lb[lb!=0] = 1*((i+1)/(len(labelList)))
    lbM += lb
        
counter = 0
while True:
    # mask_eroded = cv2.erode(mask_cards, kernel, iterations=1)
    # mask_opened = cv2.morphologyEx(mask_cards, cv2.MORPH_OPEN, kernel)
    # mask_closed = cv2.morphologyEx(mask_cards, cv2.MORPH_CLOSE, kernel)
    # mask_opclo = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)
    # mask_cloop = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    # res_original = cv2.bitwise_and(img, img, mask=mask_cards)
    # res_eroded = cv2.bitwise_and(img, img, mask=mask_eroded)
    # res_dilated = cv2.bitwise_and(img, img, mask=mask_dilated)
    # res_opened = cv2.bitwise_and(img, img, mask=mask_opened)
    # res_closed = cv2.bitwise_and(img, img, mask=mask_closed)
    # res_opclo = cv2.bitwise_and(img, img, mask=mask_opclo)
    # res_cloop = cv2.bitwise_and(img, img, mask=mask_cloop)


    #cv2.imshow('Boundary Extraction', bound_ex)
    #cv2.imshow('Tracking', res_closed)
    cv2.imshow('Original', img)
    cv2.imshow('Card Only', card_only)
    #cv2.imshow('Masking', mask)
    #cv2.imshow('No Card', no_background)    
    cv2.imshow('Mask Inverted', mask_cards)
    cv2.imshow('Mask Opening', mask_dilated)
    cv2.imshow('Segmented', segmentedImage[counter])
    cv2.imshow('Labeled List', labelList[counter])
    cv2.imshow('Labled Mask', lbM)
    #cv2.imshow('Eroded', res_eroded)
    #cv2.imshow('Dilated', res_dilated)
    #cv2.imshow('Opened', res_opened)
    #cv2.imshow('Closed', res_closed)
    #cv2.imshow('open-closed', res_opclo)
    #cv2.imshow('closed-open', res_cloop)


    cv2.waitKey(0)
    if kb.is_pressed('q'):
        break
    elif kb.is_pressed('w'):
        counter += 1
        if counter > 54:
            counter = 0

cv2.destroyAllWindows()