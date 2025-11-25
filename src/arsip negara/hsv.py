import cv2
import numpy as np

# Load the image
image = cv2.imread('../img/gb.jpeg')
b,k,d= image.shape
image2 = cv2.imread('../img/piramid.png')
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




cv2.waitKey(0)
cv2.destroyAllWindows()