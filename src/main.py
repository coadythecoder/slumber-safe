import cv2
import numpy as np
import neurokit2 as nk

# Initial settings
colour_breath = (0, 200, 0)  # Initial color for Breath
colour_pulse = (0, 200, 0)   # Initial color for Pulse
ekg_width, ekg_height = 180, 50  # Dimensions of the EKG graph
ekg_graph = np.zeros((ekg_height, ekg_width), dtype=np.uint8)  # EKG graph buffer

# Generate simulated EKG data
ekg_signal = nk.ecg_simulate(duration=10, noise=0.2, heart_rate=480)
ekg_signal = ekg_signal * 1.5  # Increase amplitude for taller pulses
ekg_signal = np.interp(ekg_signal, (ekg_signal.min(), ekg_signal.max()), (0, ekg_height - 1))  # Normalize
ekg_signal = ekg_signal.astype(int)  # Ensure integer values

# Open a connection to the default camera (index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_count = 0

# Loop to continuously capture frames
while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Add a legend box to the frame
    box_x, box_y, box_w, box_h = 10, 10, 200, 220
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (150, 150, 150), -1)  # Light grey box
    # cv2.putText(frame, "Slumber Safe:", (box_x + 10, box_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
    # Add a styled "Legend" text
    cv2.putText(frame, "Slumber Safe", (box_x + 10, box_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, lineType=cv2.LINE_AA)  # Shadow (Black outline)
    cv2.putText(frame, "Slumber Safe", (box_x + 10, box_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, lineType=cv2.LINE_AA)  # Foreground (White text)
    cv2.putText(frame, "q: Quit", (box_x + 10, box_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

    

    # Add text for Breath and Pulse with corresponding circles
    cv2.putText(frame, "Breath:", (box_x + 10, box_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
    cv2.circle(frame, (box_x + 150, box_y + 85), 10, colour_breath, -1)  # Circle for Breath

    cv2.putText(frame, "Pulse:", (box_x + 10, box_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
    cv2.circle(frame, (box_x + 150, box_y + 115), 10, colour_pulse, -1)  # Circle for Pulse

    # Update the EKG graph
    ekg_value = ekg_signal[frame_count % len(ekg_signal)]  # Get the current EKG value
    ekg_graph[:, :-1] = ekg_graph[:, 1:]  # Shift graph left
    ekg_graph[:, -1] = 0  # Clear the last column
    ekg_graph[ekg_height - ekg_value - 1, -1] = 255  # Draw the current EKG value

    # Overlay the EKG graph on the frame
    ekg_y_offset = box_y + 150
    ekg_colored = cv2.cvtColor(ekg_graph, cv2.COLOR_GRAY2BGR)  # Convert to BGR for overlay
    frame[ekg_y_offset:ekg_y_offset + ekg_height, box_x:box_x + ekg_width] = ekg_colored

    # Display the frame
    cv2.imshow('Webcam Feed', frame)

    # Press 'q' to exit the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # Press keys to update colors
    if key == ord('b'):  # Change Breath color
        colour_breath = (0, 0, 255) if colour_breath == (0, 200, 0) else (0, 200, 0)
    if key == ord('p'):  # Change Pulse color
        colour_pulse = (0, 0, 255) if colour_pulse == (0, 200, 0) else (0, 200, 0)

    frame_count += 2  # Increase frame count faster for faster EKG updates

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
