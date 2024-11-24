import numpy as np
import cv2

def compute_baby_upper_half_position(frame, prev_x, prev_y, prev_w, prev_h):
    # Path to the YOLOv3 files
    yolo_config_path = "../yolo_config/"
    config_path = yolo_config_path + "cfg/yolov3.cfg"
    weights_path = yolo_config_path + "yolov3.weights"

    # Load YOLO
    net = cv2.dnn.readNet(weights_path, config_path)

    # Load Coco names
    classes = []
    with open(yolo_config_path + 'coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]


    # Process the copied image for object detection
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)

    # Analyze and count objects on the copied image
    class_ids = []
    confidences = []
    boxes = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression for overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

   # Return the bounding box of the first detected 'person' (or None if no detection)
    if len(indexes) > 0:
        # Assume the first valid bounding box is the baby
        i = indexes.flatten()[0]
        x, y, w, h =  boxes[i]  # Returns [x, y, w, h]
        return x, y, w, h

    # If no person detected, return previous values
    return prev_x, prev_y, prev_w, prev_h
