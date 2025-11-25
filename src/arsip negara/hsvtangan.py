import cv2
import numpy as np

image = cv2.imread('../img/tangan.jpg')
b, k, d = image.shape

image2 = cv2.imread('../img/piano.jpg')
rimage = cv2.resize(image2, (k, b))

# Convert the image from BGR to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range of green color in HSV
lower= np.array([0, 20, 0])
upper = np.array([180, 255, 255])

# Create a mask that includes all shades of green
mask = cv2.inRange(hsv, lower, upper)
mask2= cv2.bitwise_not(mask)




# Invert the mask to get the foreground
foreground_mask = cv2.bitwise_not(mask)

# Use the mask to extract the foreground
foreground = cv2.bitwise_and(image, image, mask=mask)
background= cv2.bitwise_and(rimage, rimage, mask=mask2)

# Save the result

# Display the original and result images
cv2.imshow('Original', image)
cv2.imshow('Mask', mask)
cv2.imshow('bacground', image2)


#cv2.imshow('Foreground Mask', foreground_mask)
cv2.imshow('Foreground', foreground)
cv2.imshow('rimage', rimage)
cv2.imshow('mask2', mask2)
cv2.imshow('Bacground 2', background)

imgres =background +foreground 
cv2.imshow('res', imgres)

# Write the result2 to a file
cv2.imwrite("../result2/Original.jpg", image)
cv2.imwrite("../result2/Mask.jpg", mask)
cv2.imwrite("../result2/Bacground 2.jpg", background)
cv2.imwrite("../result2/Foreground.jpg", foreground)
cv2.imwrite("../result2/rimage.jpg", rimage)
cv2.imwrite("../result2/mask2.jpg", mask2)
cv2.imwrite("../result2/result.jpg", imgres)


cv2.waitKey(0)
cv2.destroyAllWindows()