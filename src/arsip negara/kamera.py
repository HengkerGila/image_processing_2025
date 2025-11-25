import cv2

# Create a VideoCapture object for the default webcam
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # If frame is not read successfully, break the loop
    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the captured frame
    cv2.imshow('Webcam Feed', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()