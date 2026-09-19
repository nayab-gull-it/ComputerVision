"""
Hand Skeleton Tracking and Visualization Application
Built with MediaPipe, OpenCV, and Tkinter.

Features:
- Detects 21 hand landmarks in real-time.
- Draws smooth, colored hand skeleton connections and joints.
- Supports both legacy MediaPipe solutions and modern MediaPipe Tasks API.
- Multiple view modes: Camera Overlay, Pure Skeleton (Canvas), and Split View.
- Multiple color themes (Finger Coded, Neon Cyberpunk, Matrix, Classic White).
- Real-time gesture recognition (Open Palm, Fist, Peace, Thumbs Up, Pointing, etc.).
- Snapshot capture to save images with drawn skeleton.
- Offline/Demo simulation mode if no webcam is available.
"""

import os
import sys
import time
import math
import urllib.request
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

# ==============================================================================
# MediaPipe Hand Detector Adapter (Compatible with all MediaPipe versions)
# ==============================================================================

# Hand skeleton connections according to MediaPipe standard 21 landmarks
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

# Finger connection groups for color coding
FINGER_CONNECTIONS = {
    'thumb': [(0, 1), (1, 2), (2, 3), (3, 4)],
    'index': [(0, 5), (5, 6), (6, 7), (7, 8)],
    'middle': [(5, 9), (9, 10), (10, 11), (11, 12)],
    'ring': [(9, 13), (13, 14), (14, 15), (15, 16)],
    'pinky': [(13, 17), (17, 18), (18, 19), (19, 20)],
    'palm': [(0, 17)]
}

# Color palettes (BGR format for OpenCV)
COLOR_THEMES = {
    'Finger Coded': {
        'thumb': (40, 90, 255),       # Vibrant Orange-Red
        'index': (20, 200, 255),      # Gold / Yellow
        'middle': (80, 225, 40),      # Emerald Green
        'ring': (255, 130, 40),       # Sky Blue
        'pinky': (240, 50, 210),      # Magenta / Violet
        'palm': (240, 230, 0),        # Cyan
        'joint': (255, 255, 255),     # White joint core
        'joint_border': (30, 30, 30), # Dark outline
    },
    'Neon Cyberpunk': {
        'thumb': (255, 230, 0),       # Cyan
        'index': (255, 230, 0),
        'middle': (255, 230, 0),
        'ring': (255, 230, 0),
        'pinky': (255, 230, 0),
        'palm': (200, 200, 0),
        'joint': (255, 50, 230),      # Neon Magenta
        'joint_border': (20, 20, 20),
    },
    'Matrix Green': {
        'thumb': (50, 255, 80),
        'index': (50, 255, 80),
        'middle': (50, 255, 80),
        'ring': (50, 255, 80),
        'pinky': (50, 255, 80),
        'palm': (30, 180, 60),
        'joint': (180, 255, 200),
        'joint_border': (0, 80, 20),
    },
    'Classic White': {
        'thumb': (240, 240, 240),
        'index': (240, 240, 240),
        'middle': (240, 240, 240),
        'ring': (240, 240, 240),
        'pinky': (240, 240, 240),
        'palm': (200, 200, 200),
        'joint': (255, 180, 50),
        'joint_border': (50, 50, 50),
    }
}


class UnifiedHandDetector:
    """
    Unified Hand Detector supporting both:
    1. Legacy mp.solutions.hands (MediaPipe <= 0.10.14)
    2. Modern mp.tasks.python.vision.HandLandmarker (MediaPipe 0.10.15+ / 1.0+)
    """
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    MODEL_FILENAME = "hand_landmarker.task"

    def __init__(self, max_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.backend = None
        self.detector = None

        self._init_detector()

    def _init_detector(self):
        import mediapipe as mp

        # Check if legacy mp.solutions is present
        if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
            self.backend = 'solutions'
            self.detector = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            print("[HandDetector] Initialized with MediaPipe Solutions API")
        else:
            # Modern MediaPipe Tasks API
            self.backend = 'tasks'
            self._ensure_model_downloaded()
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=self._get_model_path())
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=self.max_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            print("[HandDetector] Initialized with MediaPipe Tasks API")

    def _get_model_path(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(script_dir, self.MODEL_FILENAME)

    def _ensure_model_downloaded(self):
        model_path = self._get_model_path()
        if not os.path.exists(model_path):
            print(f"[HandDetector] Downloading model asset to {model_path}...")
            urllib.request.urlretrieve(self.MODEL_URL, model_path)
            print("[HandDetector] Model downloaded successfully.")

    def find_hands(self, image_bgr):
        """
        Process a BGR image frame and return detected hands with 21 landmarks.
        Returns: list of dicts:
          [
             {
                'label': 'Left' or 'Right',
                'landmarks': [(x_px, y_px, z), ... 21 points],
                'normalized': [(x_norm, y_norm, z), ...]
             }
          ]
        """
        h, w, _ = image_bgr.shape
        hands_data = []

        if self.backend == 'solutions':
            img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            results = self.detector.process(img_rgb)

            if results.multi_hand_landmarks:
                for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                    label = "Hand"
                    if results.multi_handedness and idx < len(results.multi_handedness):
                        label = results.multi_handedness[idx].classification[0].label

                    landmarks_px = []
                    normalized_lms = []
                    for lm in hand_lms.landmark:
                        px = int(lm.x * w)
                        py = int(lm.y * h)
                        landmarks_px.append((px, py, lm.z))
                        normalized_lms.append((lm.x, lm.y, lm.z))

                    hands_data.append({
                        'label': label,
                        'landmarks': landmarks_px,
                        'normalized': normalized_lms
                    })

        elif self.backend == 'tasks':
            import mediapipe as mp
            img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            results = self.detector.detect(mp_image)

            if results.hand_landmarks:
                for idx, hand_lms in enumerate(results.hand_landmarks):
                    label = "Hand"
                    if results.handedness and idx < len(results.handedness):
                        categories = results.handedness[idx]
                        if categories:
                            label = categories[0].category_name or categories[0].display_name or "Hand"

                    landmarks_px = []
                    normalized_lms = []
                    for lm in hand_lms:
                        px = int(lm.x * w)
                        py = int(lm.y * h)
                        landmarks_px.append((px, py, lm.z))
                        normalized_lms.append((lm.x, lm.y, lm.z))

                    hands_data.append({
                        'label': label,
                        'landmarks': landmarks_px,
                        'normalized': normalized_lms
                    })

        return hands_data


# ==============================================================================
# Gesture Recognition and Skeleton Drawing Utilities
# ==============================================================================

def detect_gesture(landmarks, label="Right"):
    """
    Simple, effective rule-based hand gesture classifier based on finger extensions.
    """
    if len(landmarks) < 21:
        return "Unknown", 0

    # Finger tip and PIP landmark indices
    tips = [4, 8, 12, 16, 20]
    pips = [2, 6, 10, 14, 18]

    fingers_open = [False, False, False, False, False]

    # Thumb: Check x-separation depending on hand side
    # For a right hand facing camera, thumb tip is to the left (smaller x) when open
    if label == "Right":
        fingers_open[0] = landmarks[tips[0]][0] < landmarks[pips[0]][0]
    else:
        fingers_open[0] = landmarks[tips[0]][0] > landmarks[pips[0]][0]

    # 4 fingers: Check if tip is higher (smaller y) than PIP joint
    for i in range(1, 5):
        fingers_open[i] = landmarks[tips[i]][1] < landmarks[pips[i]][1]

    open_count = sum(fingers_open)

    # Classify recognizable gestures
    thumb, index, middle, ring, pinky = fingers_open

    if open_count == 5:
        return "Open Palm", "🖐", open_count
    elif open_count == 0:
        return "Fist", "✊", open_count
    elif thumb and not index and not middle and not ring and not pinky:
        # Check if thumb tip is pointing up
        if landmarks[4][1] < landmarks[3][1]:
            return "Thumbs Up", "👍", open_count
        else:
            return "Thumbs Down", "👎", open_count
    elif index and not middle and not ring and not pinky:
        return "Pointing", "☝", open_count
    elif index and middle and not ring and not pinky:
        return "Peace / Victory", "✌", open_count
    elif index and middle and ring and not pinky and not thumb:
        return "Three Fingers", "3️⃣", open_count
    elif index and middle and ring and pinky and not thumb:
        return "Four Fingers", "4️⃣", open_count
    elif index and pinky and not middle and not ring:
        return "Rock On", "🤘", open_count
    elif thumb and pinky and not index and not middle and not ring:
        return "Call Me", "🤙", open_count
    elif thumb and index and not middle and not ring and not pinky:
        # Check distance between thumb tip and index tip for OK sign
        dist = math.hypot(landmarks[4][0] - landmarks[8][0], landmarks[4][1] - landmarks[8][1])
        if dist < 35:
            return "OK Sign", "👌", open_count
        return "Gun / L Sign", "👈", open_count

    return f"{open_count} Fingers Up", "", open_count


def draw_skeleton(image, hands, theme_name="Finger Coded", show_labels=True, show_indices=False, show_bbox=True):
    """
    Renders high-quality hand skeletons, connections, joints, and annotations on an image.
    """
    theme = COLOR_THEMES.get(theme_name, COLOR_THEMES['Finger Coded'])

    for hand in hands:
        lms = hand['landmarks']
        label = hand.get('label', 'Hand')
        gesture_name, emoji, finger_count = detect_gesture(lms, label)

        # 1. Draw Bones / Connections
        for finger_name, connections in FINGER_CONNECTIONS.items():
            color = theme.get(finger_name, (200, 200, 200))
            for start_idx, end_idx in connections:
                pt1 = (lms[start_idx][0], lms[start_idx][1])
                pt2 = (lms[end_idx][0], lms[end_idx][1])

                # Smooth anti-aliased bone line with subtle glow
                cv2.line(image, pt1, pt2, (0, 0, 0), 5, cv2.LINE_AA)        # Outline
                cv2.line(image, pt1, pt2, color, 3, cv2.LINE_AA)            # Core color

        # 2. Draw Joint Nodes
        joint_color = theme['joint']
        border_color = theme['joint_border']

        for i, pt in enumerate(lms):
            center = (pt[0], pt[1])
            # Key joints (tips and wrist) are slightly larger
            radius = 6 if i in [0, 4, 8, 12, 16, 20] else 4

            cv2.circle(image, center, radius + 2, border_color, -1, cv2.LINE_AA)
            cv2.circle(image, center, radius, joint_color, -1, cv2.LINE_AA)

            # Optional joint index numbers
            if show_indices:
                cv2.putText(image, str(i), (center[0] + 6, center[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Hand Bounding Box & Label
        if show_bbox or show_labels:
            xs = [p[0] for p in lms]
            ys = [p[1] for p in lms]
            x_min, x_max = max(0, min(xs) - 15), min(image.shape[1], max(xs) + 15)
            y_min, y_max = max(0, min(ys) - 20), min(image.shape[0], max(ys) + 15)

            if show_bbox:
                # Rounded corner bounding box style
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (70, 70, 70), 1, cv2.LINE_AA)

            if show_labels:
                header_text = f"{label}: {gesture_name}"
                # Background banner for text
                (tw, th), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                banner_y = max(24, y_min - 8)
                cv2.rectangle(image, (x_min, banner_y - th - 6), (x_min + tw + 10, banner_y + 4), (20, 20, 20), -1)
                cv2.rectangle(image, (x_min, banner_y - th - 6), (x_min + tw + 10, banner_y + 4), (0, 229, 255), 1)
                cv2.putText(image, header_text, (x_min + 5, banner_y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ==============================================================================
# Tkinter GUI Application
# ==============================================================================

class HandTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Skeleton Tracker - MediaPipe & Tkinter")
        self.root.geometry("1180x800")
        self.root.minsize(980, 680)
        self.root.configure(bg="#121214")

        # Application state variables
        self.is_running = False
        self.cap = None
        self.camera_id = 0
        self.view_mode = tk.StringVar(value="Overlay")            # "Overlay", "Skeleton Only", "Split View"
        self.color_theme = tk.StringVar(value="Finger Coded")
        self.flip_video = tk.BooleanVar(value=True)
        self.show_indices = tk.BooleanVar(value=False)
        self.show_labels = tk.BooleanVar(value=True)
        self.show_bbox = tk.BooleanVar(value=True)
        self.max_hands_var = tk.IntVar(value=2)
        self.confidence_var = tk.DoubleVar(value=0.5)

        # Simulation animation angle (for fallback mode)
        self.demo_angle = 0.0
        self.is_demo_mode = False

        # FPS calculation
        self.prev_time = time.time()
        self.fps = 0.0

        # Hand Detector initialization
        self.detector = None

        # Build UI layout
        self._setup_styles()
        self._build_ui()

        # Window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Initialize detector and start camera once event loop starts
        self.root.after(100, self._init_detector_and_start)

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        # Configure dark-themed ttk widgets
        style.configure(".", background="#121214", foreground="#E0E0E0")
        style.configure("TFrame", background="#1a1a1e")
        style.configure("Card.TFrame", background="#1e1e24", relief="flat")
        style.configure("TLabel", background="#1e1e24", foreground="#E0E0E0", font=("Segoe UI", 9))
        style.configure("Header.TLabel", background="#1e1e24", foreground="#00E5FF", font=("Segoe UI", 11, "bold"))
        style.configure("Title.TLabel", background="#121214", foreground="#FFFFFF", font=("Segoe UI", 16, "bold"))
        style.configure("Subtitle.TLabel", background="#121214", foreground="#9E9E9E", font=("Segoe UI", 9))

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
            self.status_label.config(text="Loading MediaPipe Hand Model...")
            self.root.update_idletasks()
            self.detector = UnifiedHandDetector(
                max_hands=self.max_hands_var.get(),
                min_detection_confidence=self.confidence_var.get()
            )
            self.status_label.config(text=f"Detector loaded ({self.detector.backend} backend)")
        except Exception as e:
            print(f"Error initializing HandDetector: {e}")
            self.status_label.config(text="Detector init failed. Starting synthetic demo...")
            self.start_demo_mode()
            return

        self.start_camera()

    def _build_ui(self):
        # 1. Top Header Bar
        top_bar = tk.Frame(self.root, bg="#121214", padx=20, pady=12)
        top_bar.pack(fill=tk.X)

        title_frame = tk.Frame(top_bar, bg="#121214")
        title_frame.pack(side=tk.LEFT)
        tk.Label(title_frame, text="⚡ MediaPipe Hand Skeleton Visualizer",
                 font=("Segoe UI", 16, "bold"), fg="#00E5FF", bg="#121214").pack(anchor="w")
        tk.Label(title_frame, text="Real-time 21-point hand tracking & skeleton geometry with Tkinter",
                 font=("Segoe UI", 9), fg="#8A8A9E", bg="#121214").pack(anchor="w")

        # Top Right Badges (FPS & Hands Count)
        badges_frame = tk.Frame(top_bar, bg="#121214")
        badges_frame.pack(side=tk.RIGHT)

        self.fps_badge = tk.Label(badges_frame, text="FPS: 0", font=("Consolas", 10, "bold"),
                                  fg="#00E5FF", bg="#1E1E24", padx=12, pady=5, relief="flat")
        self.fps_badge.pack(side=tk.LEFT, padx=6)

        self.hands_badge = tk.Label(badges_frame, text="Hands: 0", font=("Segoe UI", 10, "bold"),
                                    fg="#FFB300", bg="#1E1E24", padx=12, pady=5, relief="flat")
        self.hands_badge.pack(side=tk.LEFT, padx=6)

        # 2. Main Content Split (Left: Video Canvas, Right: Control Panel)
        main_content = tk.Frame(self.root, bg="#121214", padx=15, pady=5)
        main_content.pack(fill=tk.BOTH, expand=True)

        # Left Video Container
        canvas_container = tk.Frame(main_content, bg="#0A0A0C", relief="flat", bd=1)
        canvas_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12), pady=5)

        self.video_canvas = tk.Canvas(canvas_container, bg="#0D0D11", highlightthickness=0)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

        # Right Sidebar Controls
        sidebar = tk.Frame(main_content, bg="#1E1E24", width=310, padx=16, pady=16)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        sidebar.pack_propagate(False)

        # -- Card 1: Camera Controls --
        self._create_card_header(sidebar, "CAMERA CONTROLS")

        cam_btn_frame = tk.Frame(sidebar, bg="#1E1E24")
        cam_btn_frame.pack(fill=tk.X, pady=(0, 10))

        self.toggle_cam_btn = ttk.Button(cam_btn_frame, text="Stop Camera",
                                         style="Danger.TButton", command=self.toggle_camera)
        self.toggle_cam_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.snapshot_btn = ttk.Button(cam_btn_frame, text="📸 Snapshot",
                                       style="Action.TButton", command=self.take_snapshot)
        self.snapshot_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

        # Camera selection & flip
        cam_opt_frame = tk.Frame(sidebar, bg="#1E1E24")
        cam_opt_frame.pack(fill=tk.X, pady=(0, 12))

        tk.Label(cam_opt_frame, text="Camera Device:", bg="#1E1E24", fg="#B0B0C0", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self.cam_combo = ttk.Combobox(cam_opt_frame, values=["Device 0", "Device 1", "Device 2"], width=9, state="readonly")
        self.cam_combo.current(0)
        self.cam_combo.pack(side=tk.RIGHT)
        self.cam_combo.bind("<<ComboboxSelected>>", self.on_camera_device_changed)

        ttk.Checkbutton(sidebar, text="Mirror Video (Flip Horizontal)", variable=self.flip_video).pack(anchor="w", pady=(0, 14))

        # -- Card 2: View Modes --
        self._create_card_header(sidebar, "VISUALIZATION MODE")

        mode_frame = tk.Frame(sidebar, bg="#1E1E24")
        mode_frame.pack(fill=tk.X, pady=(0, 14))

        ttk.Radiobutton(mode_frame, text="Camera + Skeleton Overlay",
                        variable=self.view_mode, value="Overlay").pack(anchor="w", pady=2)
        ttk.Radiobutton(mode_frame, text="Pure Skeleton (Black Canvas)",
                        variable=self.view_mode, value="Skeleton Only").pack(anchor="w", pady=2)
        ttk.Radiobutton(mode_frame, text="Split View (Side-by-Side)",
                        variable=self.view_mode, value="Split View").pack(anchor="w", pady=2)

        # -- Card 3: Color Themes --
        self._create_card_header(sidebar, "SKELETON COLOR THEME")

        theme_frame = tk.Frame(sidebar, bg="#1E1E24")
        theme_frame.pack(fill=tk.X, pady=(0, 14))

        theme_combo = ttk.Combobox(theme_frame, textvariable=self.color_theme,
                                   values=list(COLOR_THEMES.keys()), state="readonly")
        theme_combo.pack(fill=tk.X)

        # -- Card 4: Display Toggles --
        self._create_card_header(sidebar, "OVERLAY OPTIONS")

        toggles_frame = tk.Frame(sidebar, bg="#1E1E24")
        toggles_frame.pack(fill=tk.X, pady=(0, 14))

        ttk.Checkbutton(toggles_frame, text="Show Hand & Gesture Labels",
                        variable=self.show_labels).pack(anchor="w", pady=2)
        ttk.Checkbutton(toggles_frame, text="Show Bounding Box",
                        variable=self.show_bbox).pack(anchor="w", pady=2)
        ttk.Checkbutton(toggles_frame, text="Show Landmark Indices (0-20)",
                        variable=self.show_indices).pack(anchor="w", pady=2)

        # -- Card 5: Gesture Info Panel --
        self._create_card_header(sidebar, "LIVE GESTURE DETECTED")

        self.gesture_panel = tk.Label(sidebar, text="No hands detected",
                                      font=("Segoe UI", 11, "bold"), fg="#00E5FF",
                                      bg="#141418", padx=10, pady=12, relief="flat", wraplength=260)
        self.gesture_panel.pack(fill=tk.X, pady=(0, 14))

        # Demo Mode Button (fallback)
        self.demo_btn = ttk.Button(sidebar, text="▶ Run Synthetic Demo Mode",
                                   style="Action.TButton", command=self.toggle_demo_mode)
        self.demo_btn.pack(fill=tk.X, pady=(4, 0))

        # 3. Bottom Status Bar
        status_bar = tk.Frame(self.root, bg="#0E0E10", padx=18, pady=6)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = tk.Label(status_bar, text="Initializing...",
                                     font=("Segoe UI", 9), fg="#8A8A9E", bg="#0E0E10")
        self.status_label.pack(side=tk.LEFT)

        author_label = tk.Label(status_bar, text="Computer Vision Project • MediaPipe & Tkinter",
                                font=("Segoe UI", 8), fg="#5A5A6E", bg="#0E0E10")
        author_label.pack(side=tk.RIGHT)

        # Keep reference to photo image
        self.photo_image = None
        self.current_frame = None

    def _create_card_header(self, parent, text):
        f = tk.Frame(parent, bg="#1E1E24")
        f.pack(fill=tk.X, pady=(4, 6))
        tk.Label(f, text=text, font=("Segoe UI", 9, "bold"),
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
            # Try default without backend flag
            self.cap = cv2.VideoCapture(cam_idx)

        if not self.cap.isOpened():
            self.status_label.config(text=f"Could not open camera {cam_idx}. Starting Demo mode.")
            self.start_demo_mode()
            return

        # Request 720p or standard capture resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.is_running = True
        self.toggle_cam_btn.config(text="Stop Camera", style="Danger.TButton")
        self.status_label.config(text=f"Camera {cam_idx} active • Tracking hand skeleton...")
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
            self.demo_btn.config(text="▶ Run Synthetic Demo Mode")
        else:
            self.start_demo_mode()

    def start_demo_mode(self):
        self.stop_camera()
        self.is_demo_mode = True
        self.is_running = True
        self.demo_btn.config(text="⏹ Stop Synthetic Demo")
        self.status_label.config(text="Running Synthetic Hand Animation Demo (No camera required)")
        self.update_frame()

    def _generate_synthetic_hand(self, width, height):
        """
        Creates synthetic moving hand landmarks to demonstrate skeleton rendering
        without requiring a camera.
        """
        if not hasattr(self, 'demo_angle'):
            self.demo_angle = 0.0
        self.demo_angle += 0.04
        cx = width // 2 + int(math.sin(self.demo_angle) * 50)
        cy = height // 2 + int(math.cos(self.demo_angle * 0.7) * 30)

        scale = min(width, height) * 0.35

        # Base 21 landmarks normalized relative to hand center
        base_points = [
            (0.0, 0.5),      # 0: Wrist
            (-0.25, 0.3),    # 1: Thumb CMC
            (-0.38, 0.15),   # 2: Thumb MCP
            (-0.45, 0.0),    # 3: Thumb IP
            (-0.52, -0.15),  # 4: Thumb Tip
            (-0.2, -0.1),    # 5: Index MCP
            (-0.22, -0.3),   # 6: Index PIP
            (-0.23, -0.45),  # 7: Index DIP
            (-0.24, -0.6),   # 8: Index Tip
            (0.0, -0.12),    # 9: Middle MCP
            (0.0, -0.34),    # 10: Middle PIP
            (0.0, -0.52),    # 11: Middle DIP
            (0.0, -0.68),    # 12: Middle Tip
            (0.18, -0.08),   # 13: Ring MCP
            (0.20, -0.28),   # 14: Ring PIP
            (0.21, -0.44),   # 15: Ring DIP
            (0.22, -0.58),   # 16: Ring Tip
            (0.34, -0.02),   # 17: Pinky MCP
            (0.38, -0.18),   # 18: Pinky PIP
            (0.40, -0.30),   # 19: Pinky DIP
            (0.42, -0.44),   # 20: Pinky Tip
        ]

        # Animate finger curl based on sin waves
        lms = []
        for i, (bx, by) in enumerate(base_points):
            wave = math.sin(self.demo_angle * 2 + i * 0.4) * 0.04
            px = int(cx + (bx + wave) * scale)
            py = int(cy + (by + wave) * scale)
            lms.append((px, py, 0.0))

        return [{
            'label': 'Right',
            'landmarks': lms,
            'normalized': [(p[0]/width, p[1]/height, 0.0) for p in lms]
        }]

    def update_frame(self):
        if not self.is_running:
            return

        frame = None
        hands = []

        if self.is_demo_mode:
            # Synthetic canvas frame
            frame = np.zeros((600, 800, 3), dtype=np.uint8)
            # Subtle background grid
            for x in range(0, 800, 40):
                cv2.line(frame, (x, 0), (x, 600), (18, 18, 22), 1)
            for y in range(0, 600, 40):
                cv2.line(frame, (0, y), (800, y), (18, 18, 22), 1)
            hands = self._generate_synthetic_hand(800, 600)
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
                    hands = self.detector.find_hands(frame)

        if frame is not None:
            # Calculate FPS
            curr_time = time.time()
            dt = curr_time - self.prev_time
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
            self.prev_time = curr_time
            self.fps_badge.config(text=f"FPS: {int(self.fps)}")
            self.hands_badge.config(text=f"Hands: {len(hands)}")

            # Update detected gesture panel
            if hands:
                gesture_texts = []
                for h_idx, hand in enumerate(hands):
                    gest_name, emoji, cnt = detect_gesture(hand['landmarks'], hand.get('label', 'Hand'))
                    gesture_texts.append(f"{hand.get('label', 'Hand')}: {gest_name} {emoji}")
                self.gesture_panel.config(text="\n".join(gesture_texts), fg="#00E5FF")
            else:
                self.gesture_panel.config(text="No hands detected\nShow your hand to the camera", fg="#7A7A8E")

            # Render according to visual mode
            mode = self.view_mode.get()
            theme = self.color_theme.get()
            h, w, _ = frame.shape

            if mode == "Overlay":
                display_img = frame.copy()
                draw_skeleton(display_img, hands, theme_name=theme,
                              show_labels=self.show_labels.get(),
                              show_indices=self.show_indices.get(),
                              show_bbox=self.show_bbox.get())

            elif mode == "Skeleton Only":
                display_img = np.zeros((h, w, 3), dtype=np.uint8)
                # Subtle dark grid background
                for x in range(0, w, 50):
                    cv2.line(display_img, (x, 0), (x, h), (20, 20, 26), 1)
                for y in range(0, h, 50):
                    cv2.line(display_img, (0, y), (w, y), (20, 20, 26), 1)

                draw_skeleton(display_img, hands, theme_name=theme,
                              show_labels=self.show_labels.get(),
                              show_indices=self.show_indices.get(),
                              show_bbox=self.show_bbox.get())

            elif mode == "Split View":
                # Side by side: left is raw/overlay, right is dark skeleton canvas
                half_w = w // 2
                cam_half = cv2.resize(frame, (half_w, h))

                skel_canvas = np.zeros((h, half_w, 3), dtype=np.uint8)
                # Scale landmarks for right half
                scaled_hands = []
                for hand in hands:
                    scaled_lms = [(int(pt[0] * (half_w / w)), pt[1], pt[2]) for pt in hand['landmarks']]
                    scaled_hands.append({'label': hand['label'], 'landmarks': scaled_lms})

                draw_skeleton(skel_canvas, scaled_hands, theme_name=theme,
                              show_labels=self.show_labels.get(),
                              show_indices=self.show_indices.get(),
                              show_bbox=False)

                # Combine both
                display_img = np.hstack([cam_half, skel_canvas])
                # Separator line
                cv2.line(display_img, (half_w, 0), (half_w, h), (0, 229, 255), 2)

            self.current_frame = display_img

            # Render to Tkinter Canvas with proper aspect ratio scaling
            self._render_to_canvas(display_img)

        # Schedule next frame update
        self.root.after(15, self.update_frame)

    def _render_to_canvas(self, bgr_img):
        canvas_w = self.video_canvas.winfo_width()
        canvas_h = self.video_canvas.winfo_height()

        if canvas_w < 10 or canvas_h < 10:
            return

        img_h, img_w, _ = bgr_img.shape

        # Calculate scale to fit canvas while preserving aspect ratio
        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))

        resized = cv2.resize(bgr_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_img)

        self.photo_image = ImageTk.PhotoImage(image=pil_image)

        # Center on canvas
        x_offset = (canvas_w - new_w) // 2
        y_offset = (canvas_h - new_h) // 2

        self.video_canvas.delete("all")
        self.video_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.photo_image)

    def take_snapshot(self):
        """Captures and saves the current frame with the hand skeleton."""
        if self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hand_skeleton_{timestamp}.png"
            cv2.imwrite(filename, self.current_frame)
            self.status_label.config(text=f"Snapshot saved successfully: {filename}")
            messagebox.showinfo("Snapshot Saved", f"Hand skeleton image saved as:\n{os.path.abspath(filename)}")
        else:
            messagebox.showwarning("No Frame", "No active video frame to capture.")

    def on_close(self):
        """Clean shutdown handler."""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    root = tk.Tk()
    app = HandTrackerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
