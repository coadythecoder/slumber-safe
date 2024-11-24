import cv2
import numpy as np
import time
import neurokit2 as nk
from baby_position import compute_baby_upper_half_position  
from collections import defaultdict  # For grouping detections
import matplotlib.pyplot as plt  # For plotting

baby_position_update_interval = 10 # seconds
baby_video_path = "../media/baby_video.MP4"

# Initial settings
colour_breath = (0, 200, 0)  # Initial color for Breath
ekg_width, ekg_height = 180, 50  # Dimensions of the EKG graph
ekg_graph = np.zeros((ekg_height, ekg_width), dtype=np.uint8)  # EKG graph buffer

# Generate simulated EKG data
ekg_signal = nk.ecg_simulate(duration=10, noise=0.2, heart_rate=480)
ekg_signal = ekg_signal * 1.5  # Increase amplitude for taller pulses
ekg_signal = np.interp(ekg_signal, (ekg_signal.min(), ekg_signal.max()), (0, ekg_height - 1))  # Normalize
ekg_signal = ekg_signal.astype(int)  # Ensure integer values

frame_count = 0

# Read baby video
cap = cv2.VideoCapture(baby_video_path)

if not cap.isOpened():
    print(f"Error: Cannot open the webcam feed.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize variables
start_time = time.time()
detection_intervals = []

ret, prev_frame = cap.read()

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

# Calculate the duration each frame should last in seconds
frame_duration = 1 / fps
prev_frame_time = time.time()  # Initialize the start time for the first frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Update the ROI at specified seconds interval and for first frame
    current_time = time.time()
    if current_time - last_roi_update_time >= baby_position_update_interval or first_frame:  
        x, y, w, h = compute_baby_upper_half_position(frame, x, y, w, h)
        last_roi_update_time = current_time
        if first_frame:
            first_frame = False
            prev_frame = frame

    # Compute Displayed Frame
    gray_prev = cv2.cvtColor(prev_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    frame_diff = gray_frame.astype(int) - gray_prev.astype(int)
    frame_abs_diff = np.abs(frame_diff).astype(np.uint8)

    # Focus on the ROI
    roi_diff = frame_abs_diff[y:y+h, x:x+w]

    # Calculate the amount of movement in the ROI
    movement_intensity = np.sum(roi_diff) / 255  # Normalize to count white pixels

    # Set a movement threshold
    roi_area = w * h
    movement_threshold = 0.00057 * roi_area
    movement_detected = movement_intensity > movement_threshold

    # Determine if movement is detected
    if movement_detected:
        status = "Breathing"
        displayed_status = "Breathing"
        last_breathing_update_time = current_time
        colour_breath = (0, 200, 0)
    else:
        status = "Not Breathing"

    detection_intervals.append(1 if movement_detected else 0)


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

        # Display the status on the frame at the bottom left
        cv2.putText(displayed_frame, f"Current status: {status}", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0) if status == "Breathing" else (0, 0, 255), 2)

    
    # Add a legend box to the frame
    box_x, box_y, box_w, box_h = 10, 10, 200, 220
    cv2.rectangle(displayed_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (150, 150, 150), -1)  # Light grey box
    # cv2.putText(frame, "Slumber Safe:", (box_x + 10, box_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
    # Add a styled "Legend" text
    cv2.putText(displayed_frame, "Slumber Safe", (box_x + 10, box_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, lineType=cv2.LINE_AA)  # Shadow (Black outline)
    cv2.putText(displayed_frame, "Slumber Safe", (box_x + 10, box_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, lineType=cv2.LINE_AA)  # Foreground (White text)
    cv2.putText(displayed_frame, "q: Quit", (box_x + 10, box_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

    # Add text for Breath corresponding circles
    cv2.putText(displayed_frame, "Breath:", (box_x + 10, box_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
    cv2.circle(displayed_frame, (box_x + 150, box_y + 85), 10, colour_breath, -1)  # Circle for Breath

    # Update the EKG graph
    ekg_value = ekg_signal[frame_count % len(ekg_signal)]  # Get the current EKG value
    ekg_graph[:, :-1] = ekg_graph[:, 1:]  # Shift graph left
    ekg_graph[:, -1] = 0  # Clear the last column
    ekg_graph[ekg_height - ekg_value - 1, -1] = 255  # Draw the current EKG value

    # Overlay the EKG graph on the frame
    ekg_y_offset = box_y + 150
    ekg_colored = cv2.cvtColor(ekg_graph, cv2.COLOR_GRAY2BGR)  # Convert to BGR for overlay
    displayed_frame[ekg_y_offset:ekg_y_offset + ekg_height, box_x:box_x + ekg_width] = ekg_colored

    # Calculate the elapsed time for processing the current frame
    current_time = time.time()
    elapsed_time = current_time - prev_frame_time

    # Sleep if the processing was faster than the frame duration
    if elapsed_time < frame_duration:
        time.sleep(frame_duration - elapsed_time)

    # Update the previous frame time
    prev_frame_time = time.time()

    # Display the frame
    cv2.imshow('Live Feed Movement', frame_abs_diff)
    cv2.imshow('Live Feed w/ Movement Detection', displayed_frame)


    prev_frame = frame
    frame_count += 2  # Increase frame count faster for faster EKG updates

    # Break on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

# Define FPS and desired resolution
intervals_per_second = 10  # Number of data points per second for the graph
frames_per_interval = int(fps // intervals_per_second)  # Frames per 0.1-second interval

# Aggregate detection data into 0.1-second intervals
aggregated_detections = [
    sum(detection_intervals[i:i + frames_per_interval]) / frames_per_interval
    for i in range(0, len(detection_intervals), frames_per_interval)
]

# Generate the time axis for the graph
time_axis = [i * (1 / intervals_per_second) for i in range(len(aggregated_detections))]

# Detect "Up" impulses
ups = []
i = 0

while i < len(aggregated_detections):
    if aggregated_detections[i] >= 0.66:
        ups.append(i)  # Record the start of the "up"
        i += 5  # Skip the next 4 data points as part of the same "up"
    else:
        i += 1  # Continue to the next data point

# Generate time axis for "up" impulses
up_times = [time_axis[i] for i in ups]

# Calculate the result as breaths per minute
total_impulses = len(up_times)
total_time = time_axis[-1] - time_axis[0]  # Total time in seconds
breaths_per_minute = total_impulses / total_time * 60  # Convert to breaths per minute

# Define the common x-axis range
x_start, x_end = 0, 30  # Start and end time for the x-axis
x_ticks = range(x_start, x_end + 1, 5)  # Tick marks every 5 seconds

# Plot the first graph (detailed breathing detection)
plt.figure(figsize=(12, 10), num="Baby Breathing Detection Analysis")
plt.subplot(2, 1, 1)
plt.plot(time_axis, aggregated_detections, marker='o', linestyle='-', label='Breathing Detection')
plt.xlabel('Time (seconds)')
plt.ylabel('Average Cluster Detection (Cluster Size = 3)')
plt.title('Movement Detection Over Time')
plt.grid(True)
plt.legend()
plt.xlim(x_start, x_end)
plt.xticks(x_ticks)  # Set consistent tick positions

# Plot the second graph (count of "ups")
plt.subplot(2, 1, 2)
plt.scatter(up_times, [1] * len(up_times), color='r', label='Breath Impulses')
plt.xlabel('Time (seconds)')
plt.title('Detected Breaths Over Time')
plt.yticks([1], ["Impulse"])
plt.grid(True)
plt.legend()
plt.xlim(x_start, x_end)
plt.xticks(x_ticks)  # Set consistent tick positions

# Add the breaths per minute as text to the second graph
plt.text(
    x_end - 5, 1.05,  # Position inside the second graph near the top-right
    f"Rate: {breaths_per_minute:.2f} breaths/min",  # Display breaths per minute
    fontsize=12,
    color='blue',
    horizontalalignment='right',  # Align to the right of the position
    verticalalignment='top'  # Align text to the top of the position
)

plt.tight_layout()
plt.show()
