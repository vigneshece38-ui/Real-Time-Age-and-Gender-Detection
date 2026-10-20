from flask import Flask, render_template, Response
import cv2
import os
from model_loader import load_models
from face_detector import detect_faces
from classifier import classify_age_gender

app = Flask(__name__)

# --- Pre-load Models ---
# Load all the pre-trained models once when the application starts.
try:
    face_net, age_net, gender_net = load_models()
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    face_net, age_net, gender_net = None, None, None

# --- Constants from Model Training ---
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']


def process_frame(frame):
    """
    Processes a single frame from the video stream:
    1. Detects faces.
    2. For each face, classifies age and gender.
    3. Draws the results on the frame.
    Returns the annotated frame.
    """
    if face_net is None:
        return frame

    faces = detect_faces(face_net, frame)
    for (startX, startY, endX, endY) in faces:
        face_roi = frame[startY:endY, startX:endX]
        if face_roi.size == 0:
            continue

        age, gender = classify_age_gender(
            age_net, gender_net, face_roi,
            MODEL_MEAN_VALUES, AGE_LIST, GENDER_LIST
        )
        label = f"{gender}, {age}"

        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(
            frame, label, (startX, max(25, startY - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
        )

    return frame


def generate_frames():
    """A generator function that yields frames from the webcam."""
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                annotated_frame = process_frame(frame)
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        camera.release()


# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Provides the video stream from the webcam."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == '__main__':
    app.run(debug=True)
