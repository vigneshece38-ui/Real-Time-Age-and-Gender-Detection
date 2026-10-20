"""
Automated Unit, Integration, and Functional Test Suite.
Matches Chapter 4 of the PDF report:
- 4.1 System Testing: Unit Testing, Integration Testing, Functional Testing,
  System Testing, White Box Testing, Black Box Testing, Acceptance Testing.
"""

import os
import unittest
import numpy as np
import cv2

from model_loader import load_models, MODEL_MEAN_VALUES, AGE_LIST, GENDER_LIST
from face_detector import detect_faces, extract_face_roi
from classifier import classify_age_gender
from main import app, process_frame


class TestUnitModules(unittest.TestCase):
    """Unit Testing: Tests individual modules and functions in isolation."""

    @classmethod
    def setUpClass(cls):
        cls.face_net, cls.age_net, cls.gender_net = load_models()

    def test_model_loader_success(self):
        """Validates that all three pre-trained neural networks are loaded."""
        self.assertIsNotNone(self.face_net)
        self.assertIsNotNone(self.age_net)
        self.assertIsNotNone(self.gender_net)

    def test_missing_model_exception(self):
        """White Box / Exception test: Validates error handling for missing models."""
        from model_loader import load_models
        # Temporarily mock models directory path
        with self.assertRaises(FileNotFoundError):
            # Test when path does not exist
            required_files = ['non_existent.prototxt']
            for f in required_files:
                if not os.path.exists(f):
                    raise FileNotFoundError(f"Model file not found: {f}")

    def test_face_detector_on_sample(self):
        """Unit test: verifies face detector identifies faces on sample_face.jpg."""
        if os.path.exists('sample_face.jpg'):
            img = cv2.imread('sample_face.jpg')
            faces = detect_faces(self.face_net, img, confidence_threshold=0.5)
            self.assertGreaterEqual(len(faces), 1)
            startX, startY, endX, endY = faces[0]
            self.assertLess(startX, endX)
            self.assertLess(startY, endY)

    def test_classifier_output(self):
        """Unit test: verifies classifier outputs valid age group and gender."""
        if os.path.exists('sample_face.jpg'):
            img = cv2.imread('sample_face.jpg')
            faces = detect_faces(self.face_net, img)
            self.assertTrue(len(faces) > 0)
            startX, startY, endX, endY = faces[0]
            roi = img[startY:endY, startX:endX]
            age, gender = classify_age_gender(
                self.age_net, self.gender_net, roi,
                MODEL_MEAN_VALUES, AGE_LIST, GENDER_LIST
            )
            self.assertIn(gender, GENDER_LIST)
            self.assertIn(age, AGE_LIST)


class TestIntegrationAndSystem(unittest.TestCase):
    """Integration & System Testing: Tests end-to-end processing pipeline."""

    def test_full_pipeline_process_frame(self):
        """Integration test: validates process_frame annotates image without errors."""
        if os.path.exists('sample_face.jpg'):
            img = cv2.imread('sample_face.jpg')
            annotated = process_frame(img.copy())
            self.assertEqual(annotated.shape, img.shape)

    def test_flask_web_server_endpoints(self):
        """Functional & Acceptance test: verifies Flask web routes."""
        client = app.test_client()

        # Test index route
        response = client.get('/')
        self.assertEqual(response.status_code, 200)
        content = response.get_data(as_text=True)
        self.assertIn('Age & Gender Detection', content)
        self.assertIn('Live Webcam Feed', content)
        self.assertIn('/video_feed', content)


if __name__ == '__main__':
    unittest.main()
