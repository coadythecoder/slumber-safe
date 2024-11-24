import cv2
import numpy as np
from collections import deque
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

# Initialize video capture
cap = cv2.VideoCapture("../media/IMG_2209.mov")
if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

# Video parameters
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Parameters for motion detection
TEMPORAL_WINDOW = 15  # Frames to average for motion
BREATHING_WINDOW = int(fps * 10)  # 10 seconds of data
motion_buffer = deque(maxlen=TEMPORAL_WINDOW)
breathing_signal = deque(maxlen=BREATHING_WINDOW)
breathing_rates = deque(maxlen=30)

# Initialize plots
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
line1, = ax1.plot([], [], 'b-', label='Motion Signal')
line2, = ax1.plot([], [], 'r-', label='Filtered Signal')
line3, = ax2.plot([], [], 'g-', label='Breathing Rate')
ax1.set_title('Breathing Signal')
ax1.set_ylabel('Amplitude')
ax1.grid(True)
ax1.legend()
ax2.set_title('Breathing Rate Over Time')
ax2.set_ylabel('BPM')
ax2.set_ylim(0, 50)
ax2.grid(True)

def enhance_motion(frame, prev_frames):
    """Enhanced motion detection with multiple techniques"""
    # 1. Convert to grayscale and float
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
    
    # 2. Apply contrast enhancement
    gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # 3. Calculate temporal difference with multiple previous frames
    diffs = []
    weights = np.linspace(1.0, 0.5, len(prev_frames))  # Weighted contributions
    for prev, weight in zip(prev_frames, weights):
        diff = cv2.absdiff(gray_norm, prev)
        diffs.append(diff * weight)
    
    if diffs:
        motion = np.mean(diffs, axis=0)
    else:
        motion = np.zeros_like(gray_norm)
    
    # 4. Apply spatial filtering
    motion_blur = cv2.GaussianBlur(motion, (5, 5), 0)
    
    # 5. Enhance subtle movements
    motion_enhanced = cv2.normalize(motion_blur, None, 0, 255, cv2.NORM_MINMAX)
    
    # 6. Apply adaptive thresholding
    motion_binary = cv2.adaptiveThreshold(
        motion_enhanced.astype(np.uint8),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    
    return motion_enhanced, motion_binary

def calculate_breathing_rate(signal, fps):
    if len(signal) < BREATHING_WINDOW // 2:
        return None
    
    # Normalize and detrend signal
    signal_array = np.array(signal)
    signal_norm = signal_array - np.mean(signal_array)
    
    # Apply bandpass filter (0.1-0.8 Hz = 6-48 breaths per minute)
    filtered = butter_bandpass_filter(signal_norm, 0.1, 0.35, fps)
    
    # Find peaks
    peaks, _ = find_peaks(filtered, distance=int(fps/2))  # Minimum distance between peaks
    
    if len(peaks) < 2:
        return None, filtered
    
    # Calculate breathing rate
    intervals = np.diff(peaks)
    avg_interval = np.mean(intervals)
    breathing_rate = (fps * 60) / avg_interval
    
    # Validate rate
    if not (6 <= breathing_rate <= 48):
        return None, filtered
        
    return breathing_rate, filtered

# Main processing loop
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Initial frame processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
    gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    motion_buffer.append(gray_norm)
    
    if len(motion_buffer) >= TEMPORAL_WINDOW:
        # Enhanced motion detection
        motion_enhanced, motion_binary = enhance_motion(frame, list(motion_buffer)[:-1])
        
        # Calculate motion intensity
        motion_intensity = np.mean(motion_enhanced)
        breathing_signal.append(motion_intensity)
        
        # Calculate breathing rate every second
        if frame_count % fps == 0 and len(breathing_signal) >= BREATHING_WINDOW // 2:
            rate, filtered_signal = calculate_breathing_rate(breathing_signal, fps)
            if rate is not None:
                breathing_rates.append(rate)
                
                # Update plots
                x_data = np.arange(len(breathing_signal))
                line1.set_data(x_data, breathing_signal)
                line2.set_data(x_data, filtered_signal)
                ax1.set_xlim(0, len(breathing_signal))
                ax1.set_ylim(min(breathing_signal), max(breathing_signal))
                
                line3.set_data(range(len(breathing_rates)), breathing_rates)
                ax2.set_xlim(0, max(30, len(breathing_rates)))
                
                fig.canvas.draw()
                fig.canvas.flush_events()
        
        # Create visualization
        motion_color = cv2.applyColorMap(
            motion_enhanced.astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        # Add breathing rate text if available
        if breathing_rates:
            cv2.putText(
                frame,
                f'Breathing Rate: {breathing_rates[-1]:.1f} BPM',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
        
        # Display results
        cv2.imshow('Original', frame)
        cv2.imshow('Motion Enhanced', motion_color)
        cv2.imshow('Motion Binary', motion_binary)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.close()