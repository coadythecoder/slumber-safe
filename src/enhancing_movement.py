import cv2
import numpy as np
import time
import neurokit2 as nk
from baby_position import compute_baby_upper_half_position  

baby_position_update_interval = 60 # seconds
baby_video_path = "../media/baby_video.mp4"

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

frame_count = 0


# cap = cv2.VideoCapture(0)

# Read baby video
cap = cv2.VideoCapture(baby_video_path)

if not cap.isOpened():
    print(f"Error: Cannot open the webcam feed.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# # filepath of saved (processed) video
# output_path = "output_abs_video.avi"
# out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))



ret, prev_frame = cap.read()

# image_path = "../media/baby_picture_3.jpeg"
# prev_frame = cv2.imread(image_path)

x, y, w, h = 0, 0, 0, 0
first_frame = True
status = None
displayed_status = None

if not ret:
    print("Error: Cannot read frames from webcam.")
    cap.release()
    exit()

# Track time to update the ROI and breathing indication
last_roi_update_time = time.time()
last_breathing_update_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # # Change the frame that's analyzed
    # frame = cv2.imread(image_path)


    # Update the ROI at specified seconds interval and for first frame
    current_time = time.time()
    if current_time - last_roi_update_time >= baby_position_update_interval or first_frame:  
        x, y, w, h = compute_baby_upper_half_position(frame, x, y, w, h)
        last_roi_update_time = current_time
        first_frame = False

    # convert to grayscale for simplicity
    if not prev_frame is None and x != 0 and y != 0 and w != 0 and h != 0:

        # Compute Displayed Frame
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_diff = gray_frame.astype(int) - gray_prev.astype(int)
        frame_abs_diff = np.abs(frame_diff).astype(np.uint8)


        # Focus on the ROI
        roi_diff = frame_abs_diff[y:y+h, x:x+w]

        # Calculate the amount of movement in the ROI
        movement_intensity = np.sum(roi_diff) / 255  # Normalize to count white pixels

        # Set a movement threshold
        roi_area = w * h
        movement_threshold = 0.0056 * roi_area

        # Determine if movement is detected
        if movement_intensity > movement_threshold:
            status = "Breathing"
            displayed_status = "Breathing"
            last_breathing_update_time = current_time
            colour_breath = (0, 200, 0)
        else:
            status = "Not Breathing"


        # Update the displayed to not breathing if no movement detected for 10 seconds
        if current_time - last_breathing_update_time >= 10:
            displayed_status = "Not Breathing"
            last_breathing_update_time = current_time
            colour_breath = (0, 0, 255)  # Change the color to red

    

    displayed_frame = frame.copy()

    # Check if the status is not None any apply the status to the frame
    if status is not None:
        # Draw the ROI on the frame
        cv2.rectangle(displayed_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the status on the frame
        cv2.putText(displayed_frame, f"Current status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0) if status == "Breathing" else (0, 0, 255), 2)
        cv2.putText(displayed_frame, f"Estimated status: {displayed_status}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0) if displayed_status == "Breathing" else (0, 0, 255), 2)
    
    # Add a legend box to the frame
    box_x, box_y, box_w, box_h = 10, 10, 200, 220
    cv2.rectangle(displayed_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (150, 150, 150), -1)  # Light grey box
    # cv2.putText(frame, "Slumber Safe:", (box_x + 10, box_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
    # Add a styled "Legend" text
    cv2.putText(displayed_frame, "Slumber Safe", (box_x + 10, box_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, lineType=cv2.LINE_AA)  # Shadow (Black outline)
    cv2.putText(displayed_frame, "Slumber Safe", (box_x + 10, box_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, lineType=cv2.LINE_AA)  # Foreground (White text)
    cv2.putText(displayed_frame, "q: Quit", (box_x + 10, box_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

    # Add text for Breath and Pulse with corresponding circles
    cv2.putText(displayed_frame, "Breath:", (box_x + 10, box_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
    cv2.circle(displayed_frame, (box_x + 150, box_y + 85), 10, colour_breath, -1)  # Circle for Breath

    cv2.putText(displayed_frame, "Pulse:", (box_x + 10, box_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
    cv2.circle(displayed_frame, (box_x + 150, box_y + 115), 10, colour_pulse, -1)  # Circle for Pulse

    # Update the EKG graph
    ekg_value = ekg_signal[frame_count % len(ekg_signal)]  # Get the current EKG value
    ekg_graph[:, :-1] = ekg_graph[:, 1:]  # Shift graph left
    ekg_graph[:, -1] = 0  # Clear the last column
    ekg_graph[ekg_height - ekg_value - 1, -1] = 255  # Draw the current EKG value

    # Overlay the EKG graph on the frame
    ekg_y_offset = box_y + 150
    ekg_colored = cv2.cvtColor(ekg_graph, cv2.COLOR_GRAY2BGR)  # Convert to BGR for overlay
    displayed_frame[ekg_y_offset:ekg_y_offset + ekg_height, box_x:box_x + ekg_width] = ekg_colored
        

    # Display the frame
    cv2.imshow('Live Feed w/ Movement Detection', displayed_frame)

    prev_frame = frame
    frame_count += 2  # Increase frame count faster for faster EKG updates

    # Break on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()

# print(f"Processed video saved at {output_path}")
