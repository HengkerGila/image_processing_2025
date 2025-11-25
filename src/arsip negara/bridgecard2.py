import cv2 as cv
import numpy as np
import keyboard as kb

'''
def nothing():
    pass
'''
cardSet = cv.imread("../img/kartuEdit.png")
h, w, d = cardSet.shape

rh = np.int32(h/3)
rw = np.int32(w/3)

cv.namedWindow("ori", cv.WINDOW_NORMAL)
cv.namedWindow("filtered", cv.WINDOW_NORMAL)
cv.namedWindow("mask", cv.WINDOW_NORMAL)
cv.namedWindow("segmented", cv.WINDOW_NORMAL)
cv.namedWindow("segmented mask", cv.WINDOW_NORMAL)

cardSetHSV = cv.cvtColor(cardSet, cv.COLOR_BGR2HSV)

dilateKernel = np.array(([0,1,0],[1,1,1],[0,1,0]), dtype=np.uint8)
erodeKernel = np.ones((9,9), dtype=np.uint8)

'''
cv.createTrackbar("H L", "mask", 0, 180, nothing)
cv.createTrackbar("H V", "mask", 0, 180, nothing)
cv.createTrackbar("S L", "mask", 0, 255, nothing)
cv.createTrackbar("S H", "mask", 0, 255, nothing)
cv.createTrackbar("V L", "mask", 0, 255, nothing)
cv.createTrackbar("V H", "mask", 0, 255, nothing)
'''

lower = np.array((72, 70, 0))
upper = np.array((88, 255, 220))
'''
barHL = cv.getTrackbarPos("H L", "mask")
barHV = cv.getTrackbarPos("H V", "mask")
barSL = cv.getTrackbarPos("S L", "mask")
barSH = cv.getTrackbarPos("S H", "mask")
barVL = cv.getTrackbarPos("V L", "mask")
barVH = cv.getTrackbarPos("V H", "mask")

lower = np.array((barHL, barSL, barVL))
upper = np.array((barHV, barSH, barVH))
'''
mask = cv.inRange(cardSetHSV, lower, upper)
maskInv = cv.bitwise_not(mask)

maskInv = cv.dilate(maskInv, dilateKernel)
maskInv = cv.erode(maskInv, erodeKernel)

filtered = cv.bitwise_and(cardSet, cardSet, mask=maskInv)
output = cv.connectedComponentsWithStats(maskInv)
labelCount, labeled, stat, centroid = output

labelList = list()
labelIndexList = list()
segmentedImage = list()
lbM = np.zeros(shape=(h,w))

for i in range(1, labelCount):
    count = 0
    area = stat[i, cv.CC_STAT_AREA]
    if area > 300:
        maskLabel = np.zeros(shape=maskInv.shape, dtype=np.float32)
        label = np.float32(labeled==i)*1
        maskLabel = cv.bitwise_or(maskLabel, label)
        maskLabel = np.uint8(maskLabel)*255
        segmented = cv.bitwise_and(cardSet, cardSet, mask=maskLabel)
        labelList.append(label)
        labelIndexList.append(i)
        segmentedImage.append(segmented)

for i in range(len(labelList)):
    lb = labelList[i]
    lb[lb!=0] = 1*((i+1)/(len(labelList)))
    lbM += lb

counter = 0
while True:
    cv.imshow("ori", cardSet)
    cv.imshow("filtered", filtered)
    cv.imshow("mask", lbM)
    cv.imshow("segmented", segmentedImage[counter])
    cv.imshow("segmented mask", labelList[counter])
    cv.resizeWindow("ori", rw, rh)
    cv.resizeWindow("mask", rw, rh)
    cv.resizeWindow("filtered", rw, rh)
    cv.resizeWindow("segmented", rw, rh)
    cv.resizeWindow("segmented mask", rw, rh)

    cv.waitKey(0)
    if kb.is_pressed('q'):
        break
    elif kb.is_pressed('w'):
        counter += 1
        if counter > 54:
            counter = 0

cv.destroyAllWindows()