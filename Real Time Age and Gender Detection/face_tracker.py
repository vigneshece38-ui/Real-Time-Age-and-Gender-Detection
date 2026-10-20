"""
Real-Time Face Tracking & Temporal Prediction Smoothing Module.

This module maintains temporal consistency across video frames by associating
detected faces and applying Exponential Moving Average (EMA) smoothing to
bounding box coordinates and predicted gender/age probability distributions.
"""

import time
from typing import List, Dict, Tuple, Optional
import numpy as np


class TrackedFace:
    """Represents a tracked face with temporal state and smoothed predictions."""

    def __init__(self, face_id: int, bbox: Tuple[int, int, int, int]):
        self.face_id = face_id
        self.bbox = bbox
        self.smooth_bbox = list(bbox)
        self.last_seen = time.time()
        self.frames_tracked = 1

        # Probability history / EMA state
        self.gender_probs: Optional[np.ndarray] = None
        self.age_probs: Optional[np.ndarray] = None

        self.gender: str = "Unknown"
        self.age: str = "Unknown"
        self.gender_conf: float = 0.0
        self.age_conf: float = 0.0

    def update(
        self,
        new_bbox: Tuple[int, int, int, int],
        raw_gender: str,
        raw_age: str,
        raw_gender_conf: float,
        raw_age_conf: float,
        gender_list: List[str],
        age_list: List[str],
        alpha: float = 0.35
    ):
        """
        Updates the face state using EMA smoothing for coordinates and probabilities.
        """
        self.last_seen = time.time()
        self.frames_tracked += 1

        # 1. Smooth Bounding Box
        box_alpha = 0.55  # Box position responsiveness
        for i in range(4):
            self.smooth_bbox[i] = int(box_alpha * new_bbox[i] + (1.0 - box_alpha) * self.smooth_bbox[i])
        self.bbox = tuple(self.smooth_bbox)

        # 2. Represent current prediction as pseudo-probability vector
        curr_g_probs = np.zeros(len(gender_list))
        g_idx = gender_list.index(raw_gender) if raw_gender in gender_list else 0
        curr_g_probs[g_idx] = raw_gender_conf
        other_idx = 1 - g_idx if len(gender_list) == 2 else 0
        curr_g_probs[other_idx] = max(0.01, 1.0 - raw_gender_conf)

        if self.gender_probs is None:
            self.gender_probs = curr_g_probs
        else:
            self.gender_probs = alpha * curr_g_probs + (1.0 - alpha) * self.gender_probs

        best_g_idx = int(np.argmax(self.gender_probs))
        self.gender = gender_list[best_g_idx]
        self.gender_conf = float(self.gender_probs[best_g_idx] / np.sum(self.gender_probs))

        # 3. Smooth Age Group
        curr_a_probs = np.zeros(len(age_list))
        a_idx = age_list.index(raw_age) if raw_age in age_list else 0
        curr_a_probs[a_idx] = raw_age_conf
        curr_a_probs += (1.0 - raw_age_conf) / len(age_list)

        if self.age_probs is None:
            self.age_probs = curr_a_probs
        else:
            self.age_probs = alpha * curr_a_probs + (1.0 - alpha) * self.age_probs

        best_a_idx = int(np.argmax(self.age_probs))
        self.age = age_list[best_a_idx]
        self.age_conf = float(self.age_probs[best_a_idx] / np.sum(self.age_probs))


def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = float(boxA_area + boxB_area - inter_area)

    return inter_area / union_area if union_area > 0 else 0.0


class FaceTracker:
    """Manages tracking multiple faces and smooths classifications over time."""

    def __init__(self, max_idle_time: float = 1.0, iou_threshold: float = 0.25):
        self.next_id = 1
        self.tracked_faces: Dict[int, TrackedFace] = {}
        self.max_idle_time = max_idle_time
        self.iou_threshold = iou_threshold

    def update(
        self,
        detections: List[Dict],
        gender_list: List[str],
        age_list: List[str]
    ) -> List[Dict]:
        """
        Matches incoming frame detections with existing tracked faces and applies smoothing.
        """
        now = time.time()
        # Clean up stale faces
        stale_ids = [fid for fid, tf in self.tracked_faces.items() if now - tf.last_seen > self.max_idle_time]
        for fid in stale_ids:
            del self.tracked_faces[fid]

        if not detections:
            return []

        matched_fids = set()
        matched_detections = set()
        enhanced_detections = []

        # Greedy IoU matching
        for det_idx, det in enumerate(detections):
            bbox = det["bbox"]
            best_iou = 0.0
            best_fid = None

            for fid, tf in self.tracked_faces.items():
                if fid in matched_fids:
                    continue
                iou = compute_iou(bbox, tf.bbox)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_fid = fid

            if best_fid is not None:
                # Update existing face track
                tf = self.tracked_faces[best_fid]
                tf.update(
                    new_bbox=bbox,
                    raw_gender=det["gender"],
                    raw_age=det["age"],
                    raw_gender_conf=det["gender_confidence"] / 100.0,
                    raw_age_conf=det["age_confidence"] / 100.0,
                    gender_list=gender_list,
                    age_list=age_list
                )
                matched_fids.add(best_fid)
                matched_detections.add(det_idx)

                enhanced_detections.append({
                    "face_id": best_fid,
                    "gender": tf.gender,
                    "age": tf.age,
                    "gender_confidence": round(tf.gender_conf * 100, 1),
                    "age_confidence": round(tf.age_conf * 100, 1),
                    "bbox": tf.bbox,
                    "label": f"{tf.gender}, {tf.age}"
                })
            else:
                # Create new face track
                fid = self.next_id
                self.next_id += 1
                tf = TrackedFace(face_id=fid, bbox=bbox)
                tf.update(
                    new_bbox=bbox,
                    raw_gender=det["gender"],
                    raw_age=det["age"],
                    raw_gender_conf=det["gender_confidence"] / 100.0,
                    raw_age_conf=det["age_confidence"] / 100.0,
                    gender_list=gender_list,
                    age_list=age_list
                )
                self.tracked_faces[fid] = tf
                matched_fids.add(fid)
                matched_detections.add(det_idx)

                enhanced_detections.append({
                    "face_id": fid,
                    "gender": tf.gender,
                    "age": tf.age,
                    "gender_confidence": round(tf.gender_conf * 100, 1),
                    "age_confidence": round(tf.age_conf * 100, 1),
                    "bbox": tf.bbox,
                    "label": f"{tf.gender}, {tf.age}"
                })

        return enhanced_detections
