import cv2
import numpy as np


def detect_faces(net, frame, confidence_threshold=0.7):
    """
    Detects faces in a given frame.

    Args:
        net: The loaded face detection model.
        frame: The input image/frame.
        confidence_threshold: The minimum probability to filter weak detections.

    Returns:
        A list of bounding boxes for detected faces.
    """
    # Get the frame's dimensions
    (h, w) = frame.shape[:2]

    # Create a blob from the image and pass it through the network
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > confidence_threshold:
            # Compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding box is within the frame dimensions
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)

            # Add the bounding box to our list of faces
            if startX < endX and startY < endY:
                faces.append((startX, startY, endX, endY))

    return faces


def extract_face_roi(frame, bbox):
    """
    Extracts the face Region of Interest (ROI) from a frame given a bounding box.
    """
    startX, startY, endX, endY = bbox
    return frame[startY:endY, startX:endX]
