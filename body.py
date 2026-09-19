"""
Holistic Skeleton Tracking and Biomechanical Analysis Application
Built with MediaPipe (Pose, Face, Hands), OpenCV, and Tkinter.

Features:
- Full Body Skeleton: 33-point pose estimation, joint angles (elbows, knees), posture classification.
- Face Skeleton & Identification: Facial contour mesh (oval, eyes, eyebrows, lips, nose),
  facial expression analysis, and explicit "[Face: Detected]" semantic labeling.
- Hands Skeleton & Gesture Recognition: 21 articulated joints per hand (Left & Right),
  finger-coded anatomical bone segments, and real-time gesture classification.
- Simultaneous multi-model inference via MediaPipe Tasks API with graceful fallbacks.
- Multiple view modes: Camera Overlay, Pure Skeleton (Canvas), and Split View.
- Cohesive visual color themes (Anatomical Coded, Neon Cyberpunk, Matrix Green, Classic White).
- Individual layer toggles for Body, Face, and Hands tracking.
- Interactive Tkinter GUI with live metric dashboards, snapshot capture, and synthetic mannequin demo.
"""

import os
import sys
import time
import math
import urllib.request
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

# ==============================================================================
# Skeleton Connections Definitions
# ==============================================================================

# 1. Full Body Pose Connections (MediaPipe Standard 33 Landmarks)
BODY_CONNECTIONS = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left Arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right Arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Left Leg
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    # Right Leg
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
    # Neck to Head Connection
    (11, 0), (12, 0)
]

BODY_SEGMENTS = {
    'head': [(11, 0), (12, 0)],
    'torso': [(11, 12), (11, 23), (12, 24), (23, 24)],
    'left_arm': [(11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19)],
    'right_arm': [(12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20)],
    'left_leg': [(23, 25), (25, 27), (27, 29), (29, 31), (27, 31)],
    'right_leg': [(24, 26), (26, 28), (28, 30), (30, 32), (28, 32)]
}

MAJOR_BODY_JOINTS = {11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}

# 2. Hand Connections (MediaPipe Standard 21 Landmarks)
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky finger
    (13, 17), (17, 18), (18, 19), (19, 20),
    # Palm base
    (0, 17)
]

HAND_SEGMENTS = {
    'thumb': [(0, 1), (1, 2), (2, 3), (3, 4)],
    'index': [(0, 5), (5, 6), (6, 7), (7, 8)],
    'middle': [(5, 9), (9, 10), (10, 11), (11, 12)],
    'ring': [(9, 13), (13, 14), (14, 15), (15, 16)],
    'pinky': [(13, 17), (17, 18), (18, 19), (19, 20)],
    'palm': [(0, 17)]
}

# 3. Face Contours (Key landmark indices forming canonical facial features)
# Face Oval (Jawline to temples)
FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
]

# Eyebrows
LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [336, 296, 334, 293, 300]

# Eyes
LEFT_EYE = [33, 160, 158, 133, 153, 144, 33]
RIGHT_EYE = [362, 385, 387, 263, 373, 380, 362]

# Lips Outer & Inner
LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]

# Nose Bridge & Tip
NOSE_BRIDGE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]

# Build pairs for fast line drawing
def _indices_to_pairs(idx_list):
    return [(idx_list[i], idx_list[i + 1]) for i in range(len(idx_list) - 1)]

FACE_CONTOUR_PAIRS = (
    _indices_to_pairs(FACE_OVAL_INDICES) +
    _indices_to_pairs(LEFT_EYEBROW) +
    _indices_to_pairs(RIGHT_EYEBROW) +
    _indices_to_pairs(LEFT_EYE) +
    _indices_to_pairs(RIGHT_EYE) +
    _indices_to_pairs(LIPS_OUTER) +
    _indices_to_pairs(LIPS_INNER) +
    _indices_to_pairs(NOSE_BRIDGE)
)

# Key points of interest on face
FACE_KEY_POINTS = [1, 4, 10, 33, 133, 152, 263, 362, 61, 291]


# ==============================================================================
# Visual Themes (BGR Colors for OpenCV)
# ==============================================================================

HOLISTIC_THEMES = {
    'Anatomical Coded': {
        # Body
        'torso': (30, 190, 255),      # Amber / Gold
        'left_arm': (40, 90, 255),    # Vibrant Orange-Red
        'right_arm': (230, 50, 210),  # Magenta / Violet
        'left_leg': (70, 225, 40),    # Emerald Green
        'right_leg': (255, 140, 30),  # Dodger Sky Blue
        'joint': (255, 255, 255),     # White core
        'joint_border': (20, 20, 20),
        # Face
        'face_mesh': (255, 220, 0),   # Bright Cyan/Aqua
        'face_points': (255, 255, 255),
        'face_tag_bg': (10, 50, 50),
        'face_tag_border': (255, 220, 0),
        # Hands
        'hand_thumb': (40, 90, 255),
        'hand_index': (20, 200, 255),
        'hand_middle': (80, 225, 40),
        'hand_ring': (255, 130, 40),
        'hand_pinky': (240, 50, 210),
        'hand_palm': (240, 230, 0),
        'hand_joint': (255, 255, 255),
        'hand_tag_bg': (20, 20, 30),
        'hand_tag_border': (0, 220, 255),
    },
    'Neon Cyberpunk': {
        'torso': (255, 240, 0),
        'left_arm': (255, 240, 0),
        'right_arm': (255, 240, 0),
        'left_leg': (255, 240, 0),
        'right_leg': (255, 240, 0),
        'joint': (255, 50, 230),      # Neon Magenta
        'joint_border': (15, 15, 20),
        'face_mesh': (255, 50, 230),  # Neon Pink / Magenta
        'face_points': (255, 255, 255),
        'face_tag_bg': (40, 10, 40),
        'face_tag_border': (255, 50, 230),
        'hand_thumb': (255, 240, 0),
        'hand_index': (255, 240, 0),
        'hand_middle': (255, 240, 0),
        'hand_ring': (255, 240, 0),
        'hand_pinky': (255, 240, 0),
        'hand_palm': (200, 200, 0),
        'hand_joint': (255, 50, 230),
        'hand_tag_bg': (30, 10, 35),
        'hand_tag_border': (255, 50, 230),
    },
    'Matrix Green': {
        'torso': (40, 220, 60),
        'left_arm': (60, 255, 80),
        'right_arm': (60, 255, 80),
        'left_leg': (50, 240, 70),
        'right_leg': (50, 240, 70),
        'joint': (190, 255, 210),
        'joint_border': (0, 60, 15),
        'face_mesh': (60, 255, 90),
        'face_points': (210, 255, 220),
        'face_tag_bg': (10, 35, 15),
        'face_tag_border': (60, 255, 90),
        'hand_thumb': (50, 255, 80),
        'hand_index': (50, 255, 80),
        'hand_middle': (50, 255, 80),
        'hand_ring': (50, 255, 80),
        'hand_pinky': (50, 255, 80),
        'hand_palm': (30, 180, 60),
        'hand_joint': (180, 255, 200),
        'hand_tag_bg': (10, 35, 15),
        'hand_tag_border': (60, 255, 90),
    },
    'Classic White': {
        'torso': (240, 240, 240),
        'left_arm': (240, 240, 240),
        'right_arm': (240, 240, 240),
        'left_leg': (240, 240, 240),
        'right_leg': (240, 240, 240),
        'joint': (255, 180, 50),
        'joint_border': (40, 40, 40),
        'face_mesh': (230, 230, 230),
        'face_points': (255, 255, 255),
        'face_tag_bg': (30, 30, 35),
        'face_tag_border': (230, 230, 230),
        'hand_thumb': (240, 240, 240),
        'hand_index': (240, 240, 240),
        'hand_middle': (240, 240, 240),
        'hand_ring': (240, 240, 240),
        'hand_pinky': (240, 240, 240),
        'hand_palm': (200, 200, 200),
        'hand_joint': (255, 180, 50),
        'hand_tag_bg': (30, 30, 35),
        'hand_tag_border': (220, 220, 220),
    }
}


# ==============================================================================
# Biomechanics, Posture, and Gesture Recognition Utilities
# ==============================================================================

def calculate_angle(a, b, c):
    """Computes 2D interior angle at joint 'b' formed by (a, b) and (b, c)."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(ba[0], ba[1])
    mag_bc = math.hypot(bc[0], bc[1])
    if mag_ba * mag_bc == 0:
        return 0.0
    cosine = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cosine))


def analyze_pose(landmarks):
    """
    Analyzes body landmarks to extract joint angles and classify full body posture.
    Returns: (posture_name, emoji, angles_dict)
    """
    if len(landmarks) < 33:
        return "Unknown", "", {}

    l_shoulder = (landmarks[11][0], landmarks[11][1])
    r_shoulder = (landmarks[12][0], landmarks[12][1])
    l_elbow = (landmarks[13][0], landmarks[13][1])
    r_elbow = (landmarks[14][0], landmarks[14][1])
    l_wrist = (landmarks[15][0], landmarks[15][1])
    r_wrist = (landmarks[16][0], landmarks[16][1])
    l_hip = (landmarks[23][0], landmarks[23][1])
    r_hip = (landmarks[24][0], landmarks[24][1])
    l_knee = (landmarks[25][0], landmarks[25][1])
    r_knee = (landmarks[26][0], landmarks[26][1])
    l_ankle = (landmarks[27][0], landmarks[27][1])
    r_ankle = (landmarks[28][0], landmarks[28][1])

    angles = {
        'left_elbow': int(calculate_angle(l_shoulder, l_elbow, l_wrist)),
        'right_elbow': int(calculate_angle(r_shoulder, r_elbow, r_wrist)),
        'left_knee': int(calculate_angle(l_hip, l_knee, l_ankle)),
        'right_knee': int(calculate_angle(r_hip, r_knee, r_ankle)),
        'left_shoulder': int(calculate_angle(l_elbow, l_shoulder, l_hip)),
        'right_shoulder': int(calculate_angle(r_elbow, r_shoulder, r_hip))
    }

    l_arm_raised = l_wrist[1] < l_shoulder[1]
    r_arm_raised = r_wrist[1] < r_shoulder[1]
    squatting = angles['left_knee'] < 130 and angles['right_knee'] < 130

    if l_arm_raised and r_arm_raised:
        return "Both Arms Raised", "🙌", angles
    elif l_arm_raised and not r_arm_raised:
        return "Left Arm Raised", "🙋‍♂️", angles
    elif r_arm_raised and not l_arm_raised:
        return "Right Arm Raised", "🙋‍♀️", angles
    elif squatting:
        return "Squatting / Sitting", "🧘", angles
    elif 80 <= angles['left_shoulder'] <= 110 and 80 <= angles['right_shoulder'] <= 110:
        return "T-Pose / Arms Open", "🧍", angles
    else:
        return "Standing / Neutral", "🚶", angles


def detect_hand_gesture(landmarks, label="Right"):
    """
    Robust hand gesture classifier based on finger extensions and thumb relative position.
    Returns: (gesture_name, emoji)
    """
    if len(landmarks) < 21:
        return "Tracking", "✋"

    tips = [4, 8, 12, 16, 20]
    pips = [2, 6, 10, 14, 18]
    fingers_open = [False] * 5

    # Thumb detection
    if label == "Right":
        fingers_open[0] = landmarks[tips[0]][0] < landmarks[pips[0]][0]
    else:
        fingers_open[0] = landmarks[tips[0]][0] > landmarks[pips[0]][0]

    # 4 fingers: tip higher than PIP joint
    for i in range(1, 5):
        fingers_open[i] = landmarks[tips[i]][1] < landmarks[pips[i]][1]

    open_count = sum(fingers_open)
    thumb, index, middle, ring, pinky = fingers_open

    if open_count == 5:
        return "Open Palm", "✋"
    elif open_count == 0:
        return "Fist", "✊"
    elif thumb and not index and not middle and not ring and not pinky:
        if landmarks[4][1] < landmarks[3][1]:
            return "Thumbs Up", "👍"
        else:
            return "Thumbs Down", "👎"
    elif index and not middle and not ring and not pinky:
        return "Pointing", "☝️"
    elif index and middle and not ring and not pinky:
        return "Peace / Victory", "✌️"
    elif index and middle and ring and not pinky and not thumb:
        return "Three Fingers", "3️⃣"
    elif index and pinky and not middle and not ring:
        return "Rock On", "🤘"
    elif thumb and pinky and not index and not middle and not ring:
        return "Call Me", "🤙"
    elif not index and middle and ring and pinky and thumb:
        # Distance between thumb and index tip
        dist = math.hypot(landmarks[4][0] - landmarks[8][0], landmarks[4][1] - landmarks[8][1])
        if dist < 30:
            return "OK Sign", "👌"
        return "Gesture", "✋"
    else:
        return f"{open_count} Fingers", "🖐"


def analyze_face(landmarks_px, w, h):
    """
    Analyzes face landmarks to calculate bounding box, center, and facial state.
    Returns dict: {'bbox': (x1, y1, x2, y2), 'state': str, 'center': (cx, cy)}
    """
    if not landmarks_px:
        return None

    xs = [p[0] for p in landmarks_px]
    ys = [p[1] for p in landmarks_px]

    x1, x2 = max(0, min(xs) - 15), min(w, max(xs) + 15)
    y1, y2 = max(0, min(ys) - 25), min(h, max(ys) + 15)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    state = "Detected"
    # Mouth openness: check distance between upper lip (13) and lower lip (14)
    if len(landmarks_px) > 14:
        lip_dist = abs(landmarks_px[13][1] - landmarks_px[14][1])
        mouth_w = max(1, abs(landmarks_px[61][0] - landmarks_px[291][0]))
        ratio = lip_dist / mouth_w
        if ratio > 0.28:
            state = "Speaking / Open"
        elif ratio > 0.15:
            state = "Smiling"
        else:
            state = "Neutral"

    return {
        'bbox': (x1, y1, x2, y2),
        'state': state,
        'center': (cx, cy)
    }


# ==============================================================================
# Unified Holistic Detector (MediaPipe Tasks + Solutions API)
# ==============================================================================

class UnifiedHolisticDetector:
    """
    Simultaneously runs:
    1. PoseLandmarker: 33 Full-Body Pose Landmarks
    2. HandLandmarker: Up to 2 Hands, 21 Landmarks each with Handedness
    3. FaceLandmarker: 478 Face Mesh Landmarks
    Automatically manages and downloads missing task models.
    """

    POSE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    HAND_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    FACE_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

    POSE_FILE = "pose_landmarker_lite.task"
    HAND_FILE = "hand_landmarker.task"
    FACE_FILE = "face_landmarker.task"

    def __init__(self, min_confidence=0.5):
        self.min_confidence = min_confidence
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        self.pose_detector = None
        self.hand_detector = None
        self.face_detector = None
        self.backend = 'tasks'

        self._init_all_detectors()

    def _get_path(self, filename):
        return os.path.join(self.script_dir, filename)

    def _ensure_model(self, filename, url):
        path = self._get_path(filename)
        if not os.path.exists(path):
            print(f"[HolisticDetector] Downloading {filename} from Google MediaPipe...")
            urllib.request.urlretrieve(url, path)
            print(f"[HolisticDetector] {filename} successfully downloaded.")
        return path

    def _init_all_detectors(self):
        import mediapipe as mp

        # Check legacy solutions API
        if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
            try:
                self.backend = 'solutions'
                self.pose_detector = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=self.min_confidence,
                    min_tracking_confidence=self.min_confidence
                )
                self.hand_detector = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=self.min_confidence,
                    min_tracking_confidence=self.min_confidence
                )
                self.face_detector = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    min_detection_confidence=self.min_confidence,
                    min_tracking_confidence=self.min_confidence
                )
                print("[HolisticDetector] Initialized with MediaPipe Solutions API")
                return
            except Exception as e:
                print(f"[HolisticDetector] Solutions API fallback to Tasks API: {e}")

        # Modern Tasks API
        self.backend = 'tasks'
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        # 1. Pose Model
        pose_path = self._ensure_model(self.POSE_FILE, self.POSE_URL)
        p_opts = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=pose_path),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence
        )
        self.pose_detector = vision.PoseLandmarker.create_from_options(p_opts)

        # 2. Hand Model
        hand_path = self._ensure_model(self.HAND_FILE, self.HAND_URL)
        h_opts = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=hand_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(h_opts)

        # 3. Face Model
        face_path = self._ensure_model(self.FACE_FILE, self.FACE_URL)
        f_opts = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=face_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence
        )
        self.face_detector = vision.FaceLandmarker.create_from_options(f_opts)

        print("[HolisticDetector] Initialized Pose, Hands, and Face via MediaPipe Tasks API")

    def process(self, image_bgr, enable_pose=True, enable_face=True, enable_hands=True):
        """
        Process a single BGR frame and return synchronized holistic detections:
        {
           'poses': list of pose dicts,
           'hands': list of hand dicts,
           'faces': list of face landmark lists
        }
        """
        h, w, _ = image_bgr.shape
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        poses_data = []
        hands_data = []
        faces_data = []

        if self.backend == 'solutions':
            # Pose
            if enable_pose and self.pose_detector:
                p_res = self.pose_detector.process(img_rgb)
                if p_res.pose_landmarks:
                    lms = [(int(lm.x * w), int(lm.y * h), lm.z, getattr(lm, 'visibility', 1.0))
                           for lm in p_res.pose_landmarks.landmark]
                    poses_data.append({'landmarks': lms})

            # Hands
            if enable_hands and self.hand_detector:
                h_res = self.hand_detector.process(img_rgb)
                if h_res.multi_hand_landmarks:
                    for idx, hand_lms in enumerate(h_res.multi_hand_landmarks):
                        lbl = "Hand"
                        if h_res.multi_handedness and idx < len(h_res.multi_handedness):
                            lbl = h_res.multi_handedness[idx].classification[0].label
                        lms = [(int(lm.x * w), int(lm.y * h), lm.z) for lm in hand_lms.landmark]
                        hands_data.append({'label': lbl, 'landmarks': lms})

            # Face
            if enable_face and self.face_detector:
                f_res = self.face_detector.process(img_rgb)
                if f_res.multi_face_landmarks:
                    for f_lms in f_res.multi_face_landmarks:
                        lms = [(int(lm.x * w), int(lm.y * h), lm.z) for lm in f_lms.landmark]
                        faces_data.append(lms)

        elif self.backend == 'tasks':
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            # 1. Pose
            if enable_pose and self.pose_detector:
                try:
                    p_res = self.pose_detector.detect(mp_image)
                    if p_res.pose_landmarks:
                        for pose_lms in p_res.pose_landmarks:
                            lms = [(int(lm.x * w), int(lm.y * h), lm.z, getattr(lm, 'visibility', 1.0))
                                   for lm in pose_lms]
                            poses_data.append({'landmarks': lms})
                except Exception as e:
                    pass

            # 2. Hands
            if enable_hands and self.hand_detector:
                try:
                    h_res = self.hand_detector.detect(mp_image)
                    if h_res.hand_landmarks:
                        for idx, hand_lms in enumerate(h_res.hand_landmarks):
                            lbl = "Hand"
                            if h_res.handedness and idx < len(h_res.handedness):
                                cats = h_res.handedness[idx]
                                if cats:
                                    lbl = cats[0].category_name or cats[0].display_name or "Hand"
                            lms = [(int(lm.x * w), int(lm.y * h), lm.z) for lm in hand_lms]
                            hands_data.append({'label': lbl, 'landmarks': lms})
                except Exception as e:
                    pass

            # 3. Face
            if enable_face and self.face_detector:
                try:
                    f_res = self.face_detector.detect(mp_image)
                    if f_res.face_landmarks:
                        for face_lms in f_res.face_landmarks:
                            lms = [(int(lm.x * w), int(lm.y * h), lm.z) for lm in face_lms]
                            faces_data.append(lms)
                except Exception as e:
                    pass

        return {
            'poses': poses_data,
            'hands': hands_data,
            'faces': faces_data
        }


# ==============================================================================
# Holistic Skeleton Drawing Utilities
# ==============================================================================

def draw_holistic_skeleton(image, detection_data, theme_name="Anatomical Coded",
                           show_body=True, show_face=True, show_hands=True,
                           show_labels=True, show_indices=False,
                           show_angles=True, show_bbox=True, min_vis=0.4):
    """
    Renders Full Body, Face Skeleton, and Hands Skeleton harmoniously on the canvas.
    """
    theme = HOLISTIC_THEMES.get(theme_name, HOLISTIC_THEMES['Anatomical Coded'])
    h, w, _ = image.shape

    poses = detection_data.get('poses', [])
    faces = detection_data.get('faces', [])
    hands = detection_data.get('hands', [])

    # --------------------------------------------------------------------------
    # 1. Full Body Skeleton Drawing
    # --------------------------------------------------------------------------
    if show_body and poses:
        for pose in poses:
            lms = pose['landmarks']
            if len(lms) < 33:
                continue

            posture_name, emoji, angles = analyze_pose(lms)

            # Draw Body Bones
            for seg_name, connections in BODY_SEGMENTS.items():
                color = theme.get(seg_name, (220, 220, 220))
                for start_idx, end_idx in connections:
                    pt1 = lms[start_idx]
                    pt2 = lms[end_idx]

                    if pt1[3] >= min_vis and pt2[3] >= min_vis:
                        p1 = (pt1[0], pt1[1])
                        p2 = (pt2[0], pt2[1])
                        cv2.line(image, p1, p2, (10, 10, 15), 5, cv2.LINE_AA)
                        cv2.line(image, p1, p2, color, 3, cv2.LINE_AA)

            # Draw Body Joint Nodes (exclude face points 0-10 if face mesh is active to avoid clutter)
            joint_color = theme['joint']
            border_color = theme['joint_border']

            start_joint = 11 if (show_face and faces) else 0
            for i in range(start_joint, len(lms)):
                pt = lms[i]
                if pt[3] < min_vis:
                    continue

                center = (pt[0], pt[1])
                is_major = i in MAJOR_BODY_JOINTS
                radius = 6 if is_major else 4

                cv2.circle(image, center, radius + 2, border_color, -1, cv2.LINE_AA)
                cv2.circle(image, center, radius, joint_color, -1, cv2.LINE_AA)

                if show_indices:
                    cv2.putText(image, str(i), (center[0] + 6, center[1] - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

            # Joint Angles
            if show_angles and angles:
                angle_targets = [
                    (13, f"{angles['left_elbow']} deg"),
                    (14, f"{angles['right_elbow']} deg"),
                    (25, f"{angles['left_knee']} deg"),
                    (26, f"{angles['right_knee']} deg")
                ]
                for j_idx, ang_str in angle_targets:
                    if lms[j_idx][3] >= min_vis:
                        cx, cy = lms[j_idx][0], lms[j_idx][1]
                        cv2.putText(image, ang_str, (cx - 20, cy - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

            # Body Bounding Box & Header Banner
            valid_pts = [p for p in lms if p[3] >= min_vis]
            if valid_pts and (show_bbox or show_labels):
                xs = [p[0] for p in valid_pts]
                ys = [p[1] for p in valid_pts]
                x_min, x_max = max(0, min(xs) - 15), min(w, max(xs) + 15)
                y_min, y_max = max(0, min(ys) - 20), min(h, max(ys) + 15)

                if show_bbox:
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (50, 50, 60), 1, cv2.LINE_AA)

                if show_labels:
                    header_text = f"Body: {posture_name}"
                    (tw, th), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
                    banner_y = max(22, y_min - 6)
                    cv2.rectangle(image, (x_min, banner_y - th - 6), (x_min + tw + 12, banner_y + 4), (18, 18, 22), -1)
                    cv2.rectangle(image, (x_min, banner_y - th - 6), (x_min + tw + 12, banner_y + 4), (0, 229, 255), 1)
                    cv2.putText(image, header_text, (x_min + 6, banner_y - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # 2. Face Skeleton & Facial Feature Contours Drawing
    # --------------------------------------------------------------------------
    if show_face and faces:
        face_color = theme['face_mesh']
        face_point_color = theme['face_points']

        for face_lms in faces:
            if len(face_lms) < 468:
                # Handle synthetic face landmarks with custom pairs
                continue

            face_analysis = analyze_face(face_lms, w, h)

            # Draw Contour Lines
            for start_idx, end_idx in FACE_CONTOUR_PAIRS:
                if start_idx < len(face_lms) and end_idx < len(face_lms):
                    p1 = (face_lms[start_idx][0], face_lms[start_idx][1])
                    p2 = (face_lms[end_idx][0], face_lms[end_idx][1])
                    cv2.line(image, p1, p2, (15, 15, 20), 3, cv2.LINE_AA)
                    cv2.line(image, p1, p2, face_color, 1, cv2.LINE_AA)

            # Draw Key Joint Nodes on the Face
            for idx in FACE_KEY_POINTS:
                if idx < len(face_lms):
                    pt = (face_lms[idx][0], face_lms[idx][1])
                    cv2.circle(image, pt, 3, (15, 15, 20), -1, cv2.LINE_AA)
                    cv2.circle(image, pt, 2, face_point_color, -1, cv2.LINE_AA)

            # Explicit Face Recognition Header & Bounding Box
            if face_analysis and show_labels:
                bx1, by1, bx2, by2 = face_analysis['bbox']
                state_desc = face_analysis['state']
                face_label = f"Face: Detected ({state_desc})"

                (tw, th), _ = cv2.getTextSize(face_label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                tag_y = max(18, by1 - 6)

                # Face Bounding Box outline
                if show_bbox:
                    cv2.rectangle(image, (bx1, by1), (bx2, by2), theme['face_tag_border'], 1, cv2.LINE_AA)

                # Header Tag Banner
                cv2.rectangle(image, (bx1, tag_y - th - 5), (bx1 + tw + 10, tag_y + 3), theme['face_tag_bg'], -1)
                cv2.rectangle(image, (bx1, tag_y - th - 5), (bx1 + tw + 10, tag_y + 3), theme['face_tag_border'], 1)
                cv2.putText(image, face_label, (bx1 + 5, tag_y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # 3. Hands Skeleton (21 Landmarks each) Drawing
    # --------------------------------------------------------------------------
    if show_hands and hands:
        for hand in hands:
            lms = hand['landmarks']
            label = hand.get('label', 'Hand')
            if len(lms) < 21:
                continue

            gesture_name, gesture_emoji = detect_hand_gesture(lms, label)

            # Draw Finger Bones
            for finger_name, connections in HAND_SEGMENTS.items():
                seg_color = theme.get(f'hand_{finger_name}', theme['hand_thumb'])
                for start_idx, end_idx in connections:
                    p1 = (lms[start_idx][0], lms[start_idx][1])
                    p2 = (lms[end_idx][0], lms[end_idx][1])
                    cv2.line(image, p1, p2, (15, 15, 20), 4, cv2.LINE_AA)
                    cv2.line(image, p1, p2, seg_color, 2, cv2.LINE_AA)

            # Draw Hand Joint Nodes
            for idx, pt in enumerate(lms):
                center = (pt[0], pt[1])
                is_tip = idx in (4, 8, 12, 16, 20)
                radius = 4 if is_tip else 3

                cv2.circle(image, center, radius + 2, (15, 15, 20), -1, cv2.LINE_AA)
                cv2.circle(image, center, radius, theme['hand_joint'], -1, cv2.LINE_AA)

                if show_indices:
                    cv2.putText(image, str(idx), (center[0] + 4, center[1] - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)

            # Hand Identification Banner & Gesture Tag
            if show_labels:
                wrist_pt = lms[0]
                hand_text = f"{label} Hand: {gesture_name}"
                (tw, th), _ = cv2.getTextSize(hand_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)

                tag_x = max(10, min(w - tw - 12, wrist_pt[0] - tw // 2))
                tag_y = min(h - 10, max(22, wrist_pt[1] + 28))

                cv2.rectangle(image, (tag_x, tag_y - th - 5), (tag_x + tw + 10, tag_y + 3), theme['hand_tag_bg'], -1)
                cv2.rectangle(image, (tag_x, tag_y - th - 5), (tag_x + tw + 10, tag_y + 3), theme['hand_tag_border'], 1)
                cv2.putText(image, hand_text, (tag_x + 5, tag_y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


# ==============================================================================
# Synthetic Holistic Demo Generator (Offline Mannequin Mode)
# ==============================================================================

def generate_synthetic_holistic(width, height, angle):
    """
    Generates an animated humanoid skeleton incorporating:
    - 33 Body landmarks with breathing sway and waving arm
    - 468 Face mesh contours (face oval, eyes blinking, eyebrows, smiling mouth)
    - 2x 21 Hand skeletons (articulated left waving hand and right gesturing hand)
    """
    cx = width // 2
    cy = height // 2 - 25
    scale = height * 0.40

    sway = math.sin(angle * 0.8) * 15
    arm_wave = math.sin(angle * 2.2) * 0.35
    blink = max(0.15, abs(math.sin(angle * 3.0)))

    # 1. Body Landmarks
    body_lms = [
        # 0: nose
        (cx + sway * 0.4, cy - scale * 0.72),
        # 1-6: eyes
        (cx + 8 + sway * 0.4, cy - scale * 0.75), (cx + 15 + sway * 0.4, cy - scale * 0.75),
        (cx + 22 + sway * 0.4, cy - scale * 0.74), (cx - 8 + sway * 0.4, cy - scale * 0.75),
        (cx - 15 + sway * 0.4, cy - scale * 0.75), (cx - 22 + sway * 0.4, cy - scale * 0.74),
        # 7-8: ears
        (cx + 35 + sway * 0.4, cy - scale * 0.72), (cx - 35 + sway * 0.4, cy - scale * 0.72),
        # 9-10: mouth corners
        (cx + 14 + sway * 0.4, cy - scale * 0.67), (cx - 14 + sway * 0.4, cy - scale * 0.67),
        # 11-12: shoulders
        (cx + 65 + sway * 0.6, cy - scale * 0.50), (cx - 65 + sway * 0.6, cy - scale * 0.50),
        # 13: left elbow (waving upward)
        (cx + 115 + sway * 0.8, cy - scale * (0.55 + arm_wave * 0.4)),
        # 14: right elbow
        (cx - 110 + sway * 0.8, cy - scale * 0.25),
        # 15: left wrist (waving)
        (cx + 145 + sway * 0.9, cy - scale * (0.75 + arm_wave * 0.6)),
        # 16: right wrist
        (cx - 130 + sway * 0.9, cy - scale * 0.05),
        # 17-22: body hand tip approximations
        (cx + 155, cy - scale * (0.80 + arm_wave * 0.6)), (cx - 140, cy - scale * 0.00),
        (cx + 150, cy - scale * (0.82 + arm_wave * 0.6)), (cx - 135, cy - scale * 0.02),
        (cx + 140, cy - scale * (0.78 + arm_wave * 0.6)), (cx - 125, cy - scale * 0.04),
        # 23-24: hips
        (cx + 45 + sway * 0.4, cy + scale * 0.02), (cx - 45 + sway * 0.4, cy + scale * 0.02),
        # 25-26: knees
        (cx + 50 + sway * 0.2, cy + scale * 0.48), (cx - 50 + sway * 0.2, cy + scale * 0.48),
        # 27-28: ankles
        (cx + 55, cy + scale * 0.90), (cx - 55, cy + scale * 0.90),
        # 29-32: heels & foot tips
        (cx + 55, cy + scale * 0.94), (cx - 55, cy + scale * 0.94),
        (cx + 72, cy + scale * 0.95), (cx - 72, cy + scale * 0.95),
    ]
    formatted_body = [(int(p[0]), int(p[1]), 0.0, 1.0) for p in body_lms]

    # 2. Face Landmarks (Populate full 478 points with synthetic facial geometry)
    head_cx = cx + sway * 0.4
    head_cy = cy - scale * 0.72
    face_lms = [(int(head_cx), int(head_cy), 0.0)] * 478

    # Assign key facial points
    face_lms[10] = (int(head_cx), int(head_cy - 48), 0.0)      # Forehead top
    face_lms[152] = (int(head_cx), int(head_cy + 42), 0.0)     # Chin bottom
    face_lms[234] = (int(head_cx - 36), int(head_cy - 2), 0.0) # Left cheek
    face_lms[454] = (int(head_cx + 36), int(head_cy - 2), 0.0) # Right cheek
    face_lms[1] = (int(head_cx), int(head_cy + 2), 0.0)        # Nose tip
    face_lms[4] = (int(head_cx), int(head_cy - 12), 0.0)

    # Face Oval points interpolation
    num_oval = len(FACE_OVAL_INDICES)
    for i, idx in enumerate(FACE_OVAL_INDICES):
        theta = (2 * math.pi * i) / (num_oval - 1) - math.pi / 2
        rx = 36 * math.cos(theta)
        ry = 45 * math.sin(theta)
        face_lms[idx] = (int(head_cx + rx), int(head_cy + ry), 0.0)

    # Eyes & Eyebrows
    for i, idx in enumerate(LEFT_EYEBROW):
        face_lms[idx] = (int(head_cx - 24 + i * 4), int(head_cy - 20), 0.0)
    for i, idx in enumerate(RIGHT_EYEBROW):
        face_lms[idx] = (int(head_cx + 8 + i * 4), int(head_cy - 20), 0.0)

    # Eye contours with blinking
    for i, idx in enumerate(LEFT_EYE):
        face_lms[idx] = (int(head_cx - 20 + (i % 4) * 4), int(head_cy - 10 + (i % 2) * 4 * blink), 0.0)
    for i, idx in enumerate(RIGHT_EYE):
        face_lms[idx] = (int(head_cx + 8 + (i % 4) * 4), int(head_cy - 10 + (i % 2) * 4 * blink), 0.0)

    # Lips with smile curve
    smile_curve = math.sin(angle * 1.5) * 4
    for i, idx in enumerate(LIPS_OUTER):
        dx = (i - len(LIPS_OUTER) // 2) * 1.4
        dy = 22 + abs(dx) * 0.2 - smile_curve * 0.5
        face_lms[idx] = (int(head_cx + dx), int(head_cy + dy), 0.0)
    for i, idx in enumerate(LIPS_INNER):
        dx = (i - len(LIPS_INNER) // 2) * 1.1
        dy = 23 - smile_curve * 0.4
        face_lms[idx] = (int(head_cx + dx), int(head_cy + dy), 0.0)

    # Key points
    face_lms[61] = (int(head_cx - 15), int(head_cy + 22 - smile_curve), 0.0)
    face_lms[291] = (int(head_cx + 15), int(head_cy + 22 - smile_curve), 0.0)
    face_lms[13] = (int(head_cx), int(head_cy + 20), 0.0)
    face_lms[14] = (int(head_cx), int(head_cy + 24), 0.0)

    # 3. Left Hand Skeleton (Waving Hand at Left Wrist landmark 15)
    lw_x = formatted_body[15][0]
    lw_y = formatted_body[15][1]
    left_hand_lms = []
    # Wrist
    left_hand_lms.append((lw_x, lw_y, 0.0))

    # 5 fingers spread
    finger_angles = [-0.6, -0.25, 0.0, 0.25, 0.5]
    for fa in finger_angles:
        base_x = lw_x + math.sin(fa + arm_wave * 0.5) * 12
        base_y = lw_y - math.cos(fa + arm_wave * 0.5) * 12
        for seg in range(1, 5):
            fx = base_x + math.sin(fa + arm_wave * 0.5) * (seg * 9)
            fy = base_y - math.cos(fa + arm_wave * 0.5) * (seg * 9)
            left_hand_lms.append((int(fx), int(fy), 0.0))

    # 4. Right Hand Skeleton (Resting / Gesture Hand at Right Wrist landmark 16)
    rw_x = formatted_body[16][0]
    rw_y = formatted_body[16][1]
    right_hand_lms = []
    right_hand_lms.append((rw_x, rw_y, 0.0))

    # 5 fingers pointing down / curled
    for fa in finger_angles:
        base_x = rw_x + math.sin(fa) * 10
        base_y = rw_y + math.cos(fa) * 10
        for seg in range(1, 5):
            fx = base_x + math.sin(fa) * (seg * 7)
            fy = base_y + math.cos(fa) * (seg * 7)
            right_hand_lms.append((int(fx), int(fy), 0.0))

    return {
        'poses': [{'landmarks': formatted_body}],
        'faces': [face_lms],
        'hands': [
            {'label': 'Left', 'landmarks': left_hand_lms[:21]},
            {'label': 'Right', 'landmarks': right_hand_lms[:21]}
        ]
    }


# ==============================================================================
# Tkinter GUI Application
# ==============================================================================

class HolisticBodyTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("⚡ Holistic Skeleton Tracker: Full Body • Face • Hands")
        self.root.geometry("1240x840")
        self.root.minsize(1020, 700)
        self.root.configure(bg="#121214")

        # Application State
        self.is_running = False
        self.cap = None
        self.camera_id = 0
        self.view_mode = tk.StringVar(value="Overlay")          # Overlay, Skeleton Only, Split View
        self.color_theme = tk.StringVar(value="Anatomical Coded")
        self.flip_video = tk.BooleanVar(value=True)

        # Layer Toggles
        self.track_body = tk.BooleanVar(value=True)
        self.track_face = tk.BooleanVar(value=True)
        self.track_hands = tk.BooleanVar(value=True)

        # Display Toggles
        self.show_labels = tk.BooleanVar(value=True)
        self.show_angles = tk.BooleanVar(value=True)
        self.show_indices = tk.BooleanVar(value=False)
        self.show_bbox = tk.BooleanVar(value=True)

        # Simulation
        self.demo_angle = 0.0
        self.is_demo_mode = False

        # FPS calculation
        self.prev_time = time.time()
        self.fps = 0.0

        # Holistic detector instance
        self.detector = None

        # Build UI layout
        self._setup_styles()
        self._build_ui()

        # Window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Initialize detector and start camera
        self.root.after(100, self._init_detector_and_start)

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        style.configure(".", background="#121214", foreground="#E0E0E0")
        style.configure("TFrame", background="#1a1a1e")
        style.configure("Card.TFrame", background="#1e1e24", relief="flat")
        style.configure("TLabel", background="#1e1e24", foreground="#E0E0E0", font=("Segoe UI", 9))
        style.configure("Header.TLabel", background="#1e1e24", foreground="#00E5FF", font=("Segoe UI", 10, "bold"))
        style.configure("Title.TLabel", background="#121214", foreground="#FFFFFF", font=("Segoe UI", 15, "bold"))

        style.configure("Accent.TButton", background="#00E5FF", foreground="#000000", font=("Segoe UI", 9, "bold"), borderwidth=0)
        style.map("Accent.TButton", background=[('active', '#33EBFF'), ('disabled', '#555555')])

        style.configure("Danger.TButton", background="#FF3D71", foreground="#FFFFFF", font=("Segoe UI", 9, "bold"), borderwidth=0)
        style.map("Danger.TButton", background=[('active', '#FF6690'), ('disabled', '#555555')])

        style.configure("Action.TButton", background="#2d2d38", foreground="#FFFFFF", font=("Segoe UI", 9), borderwidth=0)
        style.map("Action.TButton", background=[('active', '#3d3d4d'), ('disabled', '#555555')])

        style.configure("TCheckbutton", background="#1e1e24", foreground="#E0E0E0", font=("Segoe UI", 9))
        style.map("TCheckbutton", background=[('active', '#1e1e24')])

        style.configure("TRadiobutton", background="#1e1e24", foreground="#E0E0E0", font=("Segoe UI", 9))
        style.map("TRadiobutton", background=[('active', '#1e1e24')])

    def _init_detector_and_start(self):
        try:
            self.status_label.config(text="Loading MediaPipe Holistic (Pose, Face, Hands) models...")
            self.root.update_idletasks()
            self.detector = UnifiedHolisticDetector(min_confidence=0.5)
            self.status_label.config(text=f"Holistic detector active ({self.detector.backend} backend)")
        except Exception as e:
            print(f"Error initializing HolisticDetector: {e}")
            self.status_label.config(text="Model init warning. Running demo mode...")
            self.start_demo_mode()
            return

        self.start_camera()

    def _build_ui(self):
        # 1. Top Header Bar
        top_bar = tk.Frame(self.root, bg="#121214", padx=18, pady=10)
        top_bar.pack(fill=tk.X)

        title_frame = tk.Frame(top_bar, bg="#121214")
        title_frame.pack(side=tk.LEFT)
        tk.Label(title_frame, text="⚡ MediaPipe Holistic Skeleton Tracker",
                 font=("Segoe UI", 15, "bold"), fg="#00E5FF", bg="#121214").pack(anchor="w")
        tk.Label(title_frame, text="Synchronized Real-time Full Body (33), Face Contours (478), and Hand Articulation (21)",
                 font=("Segoe UI", 9), fg="#8A8A9E", bg="#121214").pack(anchor="w")

        # Top Right Badges (FPS, Body, Face, Hands)
        badges_frame = tk.Frame(top_bar, bg="#121214")
        badges_frame.pack(side=tk.RIGHT)

        self.fps_badge = tk.Label(badges_frame, text="FPS: 0", font=("Consolas", 10, "bold"),
                                  fg="#00E5FF", bg="#1E1E24", padx=10, pady=4, relief="flat")
        self.fps_badge.pack(side=tk.LEFT, padx=4)

        self.body_badge = tk.Label(badges_frame, text="Body: None", font=("Segoe UI", 9, "bold"),
                                   fg="#FFB300", bg="#1E1E24", padx=10, pady=4, relief="flat")
        self.body_badge.pack(side=tk.LEFT, padx=4)

        self.face_badge = tk.Label(badges_frame, text="Face: None", font=("Segoe UI", 9, "bold"),
                                   fg="#00E5FF", bg="#1E1E24", padx=10, pady=4, relief="flat")
        self.face_badge.pack(side=tk.LEFT, padx=4)

        self.hands_badge = tk.Label(badges_frame, text="Hands: None", font=("Segoe UI", 9, "bold"),
                                    fg="#FF4081", bg="#1E1E24", padx=10, pady=4, relief="flat")
        self.hands_badge.pack(side=tk.LEFT, padx=4)

        # 2. Main Content Split
        main_content = tk.Frame(self.root, bg="#121214", padx=12, pady=4)
        main_content.pack(fill=tk.BOTH, expand=True)

        # Video Canvas Container
        canvas_container = tk.Frame(main_content, bg="#0A0A0C", relief="flat", bd=1)
        canvas_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=4)

        self.video_canvas = tk.Canvas(canvas_container, bg="#0D0D11", highlightthickness=0)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

        # Right Sidebar Controls
        sidebar = tk.Frame(main_content, bg="#1E1E24", width=335, padx=14, pady=12)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, pady=4)
        sidebar.pack_propagate(False)

        # -- Card 1: Camera Controls --
        self._create_card_header(sidebar, "CAMERA & HARDWARE")

        cam_btn_frame = tk.Frame(sidebar, bg="#1E1E24")
        cam_btn_frame.pack(fill=tk.X, pady=(0, 8))

        self.toggle_cam_btn = ttk.Button(cam_btn_frame, text="Stop Camera",
                                         style="Danger.TButton", command=self.toggle_camera)
        self.toggle_cam_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))

        self.snapshot_btn = ttk.Button(cam_btn_frame, text="📸 Snapshot",
                                       style="Action.TButton", command=self.take_snapshot)
        self.snapshot_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(3, 0))

        cam_opt_frame = tk.Frame(sidebar, bg="#1E1E24")
        cam_opt_frame.pack(fill=tk.X, pady=(0, 8))

        tk.Label(cam_opt_frame, text="Camera Device:", bg="#1E1E24", fg="#B0B0C0", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self.cam_combo = ttk.Combobox(cam_opt_frame, values=["Device 0", "Device 1", "Device 2"], width=9, state="readonly")
        self.cam_combo.current(0)
        self.cam_combo.pack(side=tk.RIGHT)
        self.cam_combo.bind("<<ComboboxSelected>>", self.on_camera_device_changed)

        ttk.Checkbutton(sidebar, text="Mirror Video (Horizontal Flip)", variable=self.flip_video).pack(anchor="w", pady=(0, 10))

        # -- Card 2: Holistic Tracking Layers --
        self._create_card_header(sidebar, "SKELETON LAYERS")

        layers_frame = tk.Frame(sidebar, bg="#1E1E24")
        layers_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Checkbutton(layers_frame, text="🏃 Track Full Body Pose (33)",
                        variable=self.track_body).pack(anchor="w", pady=2)
        ttk.Checkbutton(layers_frame, text="👤 Track Face Skeleton & Mesh",
                        variable=self.track_face).pack(anchor="w", pady=2)
        ttk.Checkbutton(layers_frame, text="✋ Track Hands Skeletons (21x2)",
                        variable=self.track_hands).pack(anchor="w", pady=2)

        # -- Card 3: View Modes & Themes --
        self._create_card_header(sidebar, "VISUALIZATION & THEME")

        mode_frame = tk.Frame(sidebar, bg="#1E1E24")
        mode_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Radiobutton(mode_frame, text="Camera + Skeleton Overlay",
                        variable=self.view_mode, value="Overlay").pack(anchor="w", pady=1)
        ttk.Radiobutton(mode_frame, text="Pure Skeleton (Dark Canvas)",
                        variable=self.view_mode, value="Skeleton Only").pack(anchor="w", pady=1)
        ttk.Radiobutton(mode_frame, text="Split View (Side-by-Side)",
                        variable=self.view_mode, value="Split View").pack(anchor="w", pady=1)

        theme_frame = tk.Frame(sidebar, bg="#1E1E24")
        theme_frame.pack(fill=tk.X, pady=(0, 10))
        theme_combo = ttk.Combobox(theme_frame, textvariable=self.color_theme,
                                   values=list(HOLISTIC_THEMES.keys()), state="readonly")
        theme_combo.pack(fill=tk.X)

        # -- Card 4: Overlays & Labels --
        self._create_card_header(sidebar, "DISPLAY OVERLAYS")

        disp_frame = tk.Frame(sidebar, bg="#1E1E24")
        disp_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Checkbutton(disp_frame, text="Show Semantic Tags ([Face], [Hands], [Body])",
                        variable=self.show_labels).pack(anchor="w", pady=1)
        ttk.Checkbutton(disp_frame, text="Show Joint Angles & Posture",
                        variable=self.show_angles).pack(anchor="w", pady=1)
        ttk.Checkbutton(disp_frame, text="Show Landmark Numerical Indices",
                        variable=self.show_indices).pack(anchor="w", pady=1)

        # -- Card 5: Live Holistic Status Panel --
        self._create_card_header(sidebar, "LIVE HOLISTIC ANALYSIS")

        self.posture_panel = tk.Label(sidebar, text="Body: No person detected",
                                      font=("Segoe UI", 9, "bold"), fg="#FFB300",
                                      bg="#141418", padx=8, pady=5, relief="flat", anchor="w")
        self.posture_panel.pack(fill=tk.X, pady=(0, 4))

        self.face_panel = tk.Label(sidebar, text="Face: No face detected",
                                   font=("Segoe UI", 9, "bold"), fg="#00E5FF",
                                   bg="#141418", padx=8, pady=5, relief="flat", anchor="w")
        self.face_panel.pack(fill=tk.X, pady=(0, 4))

        self.hands_panel = tk.Label(sidebar, text="Hands: No hands detected",
                                    font=("Segoe UI", 9, "bold"), fg="#FF4081",
                                    bg="#141418", padx=8, pady=5, relief="flat", anchor="w")
        self.hands_panel.pack(fill=tk.X, pady=(0, 10))

        # Demo Button
        self.demo_btn = ttk.Button(sidebar, text="▶ Run Synthetic Mannequin Demo",
                                   style="Action.TButton", command=self.toggle_demo_mode)
        self.demo_btn.pack(fill=tk.X, pady=(2, 0))

        # 3. Bottom Status Bar
        status_bar = tk.Frame(self.root, bg="#0E0E10", padx=16, pady=5)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = tk.Label(status_bar, text="Initializing...",
                                     font=("Segoe UI", 9), fg="#8A8A9E", bg="#0E0E10")
        self.status_label.pack(side=tk.LEFT)

        author_label = tk.Label(status_bar, text="Full Body (33) • Face (478) • Hands (21x2) • OpenCV + Tkinter",
                                font=("Segoe UI", 8), fg="#5A5A6E", bg="#0E0E10")
        author_label.pack(side=tk.RIGHT)

        self.photo_image = None
        self.current_frame = None

    def _create_card_header(self, parent, text):
        f = tk.Frame(parent, bg="#1E1E24")
        f.pack(fill=tk.X, pady=(2, 4))
        tk.Label(f, text=text, font=("Segoe UI", 8, "bold"),
                 fg="#00E5FF", bg="#1E1E24").pack(anchor="w")

    def toggle_camera(self):
        if self.is_running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        if self.is_running:
            return

        self.is_demo_mode = False
        cam_idx = self.cam_combo.current()
        self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY)

        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(cam_idx)

        if not self.cap.isOpened():
            self.status_label.config(text=f"Camera {cam_idx} unavailable. Running synthetic mannequin demo.")
            self.start_demo_mode()
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.is_running = True
        self.toggle_cam_btn.config(text="Stop Camera", style="Danger.TButton")
        self.status_label.config(text=f"Camera {cam_idx} active • Tracking Body, Face & Hands simultaneously...")
        self.update_frame()

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.toggle_cam_btn.config(text="Start Camera", style="Accent.TButton")
        self.status_label.config(text="Camera stopped.")
        self.fps_badge.config(text="FPS: 0")

    def on_camera_device_changed(self, event=None):
        if self.is_running:
            self.stop_camera()
            self.start_camera()

    def toggle_demo_mode(self):
        if self.is_demo_mode:
            self.stop_camera()
            self.start_camera()
            self.demo_btn.config(text="▶ Run Synthetic Mannequin Demo")
        else:
            self.start_demo_mode()

    def start_demo_mode(self):
        self.stop_camera()
        self.is_demo_mode = True
        self.is_running = True
        self.demo_btn.config(text="⏹ Stop Synthetic Demo")
        self.status_label.config(text="Running Synthetic Holistic Mannequin Demo (Animated Body + Face + Hands)")
        self.update_frame()

    def update_frame(self):
        if not self.is_running:
            return

        frame = None
        detection_data = {'poses': [], 'faces': [], 'hands': []}

        if self.is_demo_mode:
            frame = np.zeros((650, 850, 3), dtype=np.uint8)
            # Subtle background grid
            for x in range(0, 850, 45):
                cv2.line(frame, (x, 0), (x, 650), (18, 18, 22), 1)
            for y in range(0, 650, 45):
                cv2.line(frame, (0, y), (850, y), (18, 18, 22), 1)

            self.demo_angle += 0.04
            detection_data = generate_synthetic_holistic(850, 650, self.demo_angle)
        else:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    self.status_label.config(text="Camera frame read error.")
                    self.root.after(30, self.update_frame)
                    return

                if self.flip_video.get():
                    frame = cv2.flip(frame, 1)

                if self.detector:
                    detection_data = self.detector.process(
                        frame,
                        enable_pose=self.track_body.get(),
                        enable_face=self.track_face.get(),
                        enable_hands=self.track_hands.get()
                    )

        if frame is not None:
            # FPS calculation
            curr_time = time.time()
            dt = curr_time - self.prev_time
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
            self.prev_time = curr_time

            # Update Badges
            poses = detection_data.get('poses', [])
            faces = detection_data.get('faces', [])
            hands = detection_data.get('hands', [])

            self.fps_badge.config(text=f"FPS: {int(self.fps)}")
            self.body_badge.config(text="Body: Tracked" if poses else "Body: None")
            self.face_badge.config(text="Face: Tracked" if faces else "Face: None")
            self.hands_badge.config(text=f"Hands: {len(hands)}" if hands else "Hands: None")

            # Update Live Status Panel
            if poses:
                posture_name, emoji, angles = analyze_pose(poses[0]['landmarks'])
                self.posture_panel.config(text=f"🏃 Body: {posture_name} {emoji}", fg="#FFB300")
            else:
                self.posture_panel.config(text="🏃 Body: No person detected", fg="#7A7A8E")

            if faces:
                f_analysis = analyze_face(faces[0], frame.shape[1], frame.shape[0])
                state = f_analysis['state'] if f_analysis else "Detected"
                self.face_panel.config(text=f"👤 Face: Detected ({state})", fg="#00E5FF")
            else:
                self.face_panel.config(text="👤 Face: No face detected", fg="#7A7A8E")

            if hands:
                hand_summaries = []
                for h_item in hands:
                    lbl = h_item.get('label', 'Hand')
                    g_name, g_emo = detect_hand_gesture(h_item['landmarks'], lbl)
                    hand_summaries.append(f"{lbl}: {g_name} {g_emo}")
                self.hands_panel.config(text="✋ " + " | ".join(hand_summaries), fg="#FF4081")
            else:
                self.hands_panel.config(text="✋ Hands: No hands detected", fg="#7A7A8E")

            # Render According to Visual Mode
            mode = self.view_mode.get()
            theme = self.color_theme.get()
            h, w, _ = frame.shape

            if mode == "Overlay":
                display_img = frame.copy()
                draw_holistic_skeleton(display_img, detection_data, theme_name=theme,
                                       show_body=self.track_body.get(),
                                       show_face=self.track_face.get(),
                                       show_hands=self.track_hands.get(),
                                       show_labels=self.show_labels.get(),
                                       show_indices=self.show_indices.get(),
                                       show_angles=self.show_angles.get(),
                                       show_bbox=self.show_bbox.get())

            elif mode == "Skeleton Only":
                display_img = np.zeros((h, w, 3), dtype=np.uint8)
                for x in range(0, w, 50):
                    cv2.line(display_img, (x, 0), (x, h), (18, 18, 24), 1)
                for y in range(0, h, 50):
                    cv2.line(display_img, (0, y), (w, y), (18, 18, 24), 1)

                draw_holistic_skeleton(display_img, detection_data, theme_name=theme,
                                       show_body=self.track_body.get(),
                                       show_face=self.track_face.get(),
                                       show_hands=self.track_hands.get(),
                                       show_labels=self.show_labels.get(),
                                       show_indices=self.show_indices.get(),
                                       show_angles=self.show_angles.get(),
                                       show_bbox=self.show_bbox.get())

            elif mode == "Split View":
                half_w = w // 2
                cam_half = cv2.resize(frame, (half_w, h))

                skel_canvas = np.zeros((h, half_w, 3), dtype=np.uint8)
                # Rescale detection points for half canvas
                scaled_data = {'poses': [], 'faces': [], 'hands': []}
                scale_x = half_w / w

                for pose in detection_data.get('poses', []):
                    scaled_lms = [(int(p[0] * scale_x), p[1], p[2], p[3]) for p in pose['landmarks']]
                    scaled_data['poses'].append({'landmarks': scaled_lms})

                for face_lms in detection_data.get('faces', []):
                    scaled_flms = [(int(p[0] * scale_x), p[1], p[2]) for p in face_lms]
                    scaled_data['faces'].append(scaled_flms)

                for hand_item in detection_data.get('hands', []):
                    scaled_hlms = [(int(p[0] * scale_x), p[1], p[2]) for p in hand_item['landmarks']]
                    scaled_data['hands'].append({'label': hand_item.get('label', 'Hand'), 'landmarks': scaled_hlms})

                draw_holistic_skeleton(skel_canvas, scaled_data, theme_name=theme,
                                       show_body=self.track_body.get(),
                                       show_face=self.track_face.get(),
                                       show_hands=self.track_hands.get(),
                                       show_labels=self.show_labels.get(),
                                       show_indices=self.show_indices.get(),
                                       show_angles=self.show_angles.get(),
                                       show_bbox=False)

                display_img = np.hstack([cam_half, skel_canvas])
                cv2.line(display_img, (half_w, 0), (half_w, h), (0, 229, 255), 2)

            self.current_frame = display_img
            self._render_to_canvas(display_img)

        self.root.after(15, self.update_frame)

    def _render_to_canvas(self, bgr_img):
        canvas_w = self.video_canvas.winfo_width()
        canvas_h = self.video_canvas.winfo_height()

        if canvas_w < 10 or canvas_h < 10:
            return

        img_h, img_w, _ = bgr_img.shape
        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))

        resized = cv2.resize(bgr_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_img)

        self.photo_image = ImageTk.PhotoImage(image=pil_image)

        x_offset = (canvas_w - new_w) // 2
        y_offset = (canvas_h - new_h) // 2

        self.video_canvas.delete("all")
        self.video_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.photo_image)

    def take_snapshot(self):
        """Captures and saves current frame with holistic skeleton to a PNG file."""
        if self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"holistic_skeleton_{timestamp}.png"
            cv2.imwrite(filename, self.current_frame)
            self.status_label.config(text=f"Snapshot saved: {filename}")
            messagebox.showinfo("Snapshot Saved", f"Holistic skeleton saved as:\n{os.path.abspath(filename)}")
        else:
            messagebox.showwarning("No Frame", "No active video frame to capture.")

    def on_close(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    root = tk.Tk()
    app = HolisticBodyTrackerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
