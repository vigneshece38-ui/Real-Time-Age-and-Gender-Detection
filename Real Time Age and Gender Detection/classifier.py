import cv2


def classify_age_gender(age_net, gender_net, face_roi, model_mean_values, age_list, gender_list):
    """
    Classifies age and gender for a given face Region of Interest (ROI).

    Args:
        age_net: The loaded age classification model.
        gender_net: The loaded gender classification model.
        face_roi: The cropped image of the face.
        model_mean_values: Mean values for image normalization.
        age_list: List of age bracket labels.
        gender_list: List of gender labels.

    Returns:
        A tuple containing the predicted age and gender strings: (age, gender).
    """
    # Create a blob from the face ROI. This preprocesses the image
    # to the format the Caffe models expect (227x227 pixels, mean subtraction).
    blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), model_mean_values, swapRB=False)

    # --- Predict Gender ---
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    # Get the gender with the highest confidence
    gender = gender_list[gender_preds[0].argmax()]

    # --- Predict Age ---
    age_net.setInput(blob)
    age_preds = age_net.forward()
    # Get the age bracket with the highest confidence
    age = age_list[age_preds[0].argmax()]

    return age, gender
