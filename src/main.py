import cv2

# Open a connection to the default camera (index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop to continuously capture frames
while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Display the frame
    cv2.imshow('Webcam Feed', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
