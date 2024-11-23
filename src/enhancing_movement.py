import cv2
import numpy as np


video_path = "../media/baby2.mp4"
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(f"Error: Cannot open video at {video_path}")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# filepath of saved (processed) video
# output_path = "output_abs_video.avi"
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

ret, prev_frame = cap.read()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # convert to grayscale for simplicity
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   
    frame_diff = gray_frame.astype(int) - gray_prev.astype(int)
    abs_diff = np.abs(frame_diff).astype(np.uint8)
    abs_diff_bgr = cv2.cvtColor(abs_diff, cv2.COLOR_GRAY2BGR)

    #uncomment if you want to save video
    # out.write(abs_diff_bgr)

    #this is the part that actively displays processed frames 
    cv2.imshow('Absolute Difference', abs_diff)

    prev_frame = frame

    # Break on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()

# print(f"Processed video saved at {output_path}")
