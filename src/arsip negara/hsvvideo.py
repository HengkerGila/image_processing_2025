import cv2
import numpy as np

cap = cv2.VideoCapture(0)
image2 = cv2.imread('../img/montenegro.jpg')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    cv2.imshow('Webcam Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    b,k,d= frame.shape

    #image = cv2.imread('../img/images.jpg')
    #b, k, d = image.shape

   
    rimage = cv2.resize(image2, (k, b))

    
    # Image to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    # Define the range of blue color in HSV
    lower= np.array([65, 10, 20])
    upper = np.array([130, 255, 255])

    # Create a mask that includes all shades of green
    mask = cv2.inRange(hsv, lower, upper)
    mask2= cv2.bitwise_not(mask)

    # Invert the mask to get the foreground
    foreground_mask = cv2.bitwise_not(mask)




    # Use the mask to extract the foreground
    foreground = cv2.bitwise_and(frame, frame, mask=mask2)
    background= cv2.bitwise_and(rimage, rimage, mask=mask)

    # Save the result

    # Display the original and result images
    cv2.imshow('Mask', mask)
    #cv2.imshow('bacground', image2)


        #cv2.imshow('Foreground Mask', foreground_mask)
    cv2.imshow('Foreground', foreground)
    #cv2.imshow('rimage', rimage)
    cv2.imshow('mask2', mask2)
    #cv2.imshow('Bacground 2', background)

    imgres =background +foreground 
    cv2.imshow('res', imgres)



cap.release()
cv2.destroyAllWindows()
