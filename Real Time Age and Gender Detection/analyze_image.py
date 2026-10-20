import cv2
import argparse
import os
from model_loader import load_models
from face_detector import detect_faces
from classifier import classify_age_gender

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']


def main(args):
    # --- 1. Load Models ---
    try:
        face_net, age_net, gender_net = load_models()
        print("Models loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the model files are in the 'models' directory.")
        return

    # --- 2. Read Input Image ---
    if not os.path.exists(args.image):
        print(f"Error: Input image not found at '{args.image}'")
        return

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Error: Could not read the image file '{args.image}'")
        return

    # --- 3. Detect Faces ---
    faces = detect_faces(face_net, frame)
    if not faces:
        print("No faces detected in the image.")
        return

    print(f"Detected {len(faces)} faces.")

    # --- 4. Classify and Annotate ---
    h_frame = frame.shape[0]
    font_scale = max(0.6, h_frame / 750.0)
    thickness = max(1, int(h_frame / 350.0))
    box_thickness = max(2, int(h_frame / 250.0))

    for (startX, startY, endX, endY) in faces:
        face_roi = frame[startY:endY, startX:endX]
        if face_roi.size == 0:
            continue

        age, gender = classify_age_gender(
            age_net, gender_net, face_roi,
            MODEL_MEAN_VALUES, AGE_LIST, GENDER_LIST
        )
        label = f"{gender}, {age}"

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), box_thickness)
        cv2.putText(
            frame, label, (startX, max(25, startY - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA
        )

    # --- 5. Save and/or Display Output ---
    # Construct the output path
    dirname = os.path.dirname(args.image)
    output_filename = "output_" + os.path.basename(args.image)
    output_path = os.path.join(dirname, output_filename) if dirname else output_filename

    # Save the annotated image
    cv2.imwrite(output_path, frame)
    print(f"Result saved to: {output_path}")

    # Display the result in a window
    if not getattr(args, 'no_display', False):
        try:
            cv2.imshow("Age and Gender Detection", frame)
            print("Press any key to close the image window.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Note: Display window skipped ({e}). Result was saved to: {output_path}")


if __name__ == '__main__':
    # --- Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(description='Age and Gender Detection from an image.')
    parser.add_argument('-i', '--image', type=str, required=True, help='Path to the input image file.')
    parser.add_argument('--no-display', action='store_true', help='Skip cv2.imshow GUI window (useful for headless/scripts).')
    args = parser.parse_args()
    main(args)
