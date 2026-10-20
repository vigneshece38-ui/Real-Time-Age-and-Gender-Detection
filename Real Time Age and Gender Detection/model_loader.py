import cv2
import os

# --- Constants from Model Training ---
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']


def load_models():
    """
    Loads the face detection, age, and gender classification models.
    Returns the network objects for each model.
    """
    # Define paths to the model files.
    # These files should be in a 'models' directory in your project.
    face_proto = os.path.join('models', 'opencv_face_detector.pbtxt')
    face_model = os.path.join('models', 'opencv_face_detector_uint8.pb')
    age_proto = os.path.join('models', 'age_deploy.prototxt')
    age_model = os.path.join('models', 'age_net.caffemodel')
    gender_proto = os.path.join('models', 'gender_deploy.prototxt')
    gender_model = os.path.join('models', 'gender_net.caffemodel')

    # Check if model files exist
    required_files = [face_proto, face_model, age_proto, age_model, gender_proto, gender_model]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Model file not found: {file_path}. Please download the models and place them in the 'models' directory."
            )

    # Load the networks using OpenCV's DNN module
    print("Loading face detection model...")
    face_net = cv2.dnn.readNet(face_model, face_proto)

    print("Loading age classification model...")
    age_net = cv2.dnn.readNet(age_model, age_proto)

    print("Loading gender classification model...")
    gender_net = cv2.dnn.readNet(gender_model, gender_proto)

    return face_net, age_net, gender_net


# Alias for backward compatibility
load_all_models = load_models


if __name__ == '__main__':
    try:
        f_net, a_net, g_net = load_models()
        print("All models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
