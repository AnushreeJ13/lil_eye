"""
EyeSpy CV Engine — Multi-Signal Drowsiness Detection Pipeline

Uses MediaPipe Face Mesh (478 landmarks) for:
  1. Eye Aspect Ratio (EAR) — eye openness
  2. Mouth Aspect Ratio (MAR) — yawn detection
  3. PERCLOS — percentage of eye closure over time window
  4. Blink Rate — blinks per minute
  5. Head Pose Estimation — pitch/yaw/roll via cv2.solvePnP
  6. Composite Alertness Score — weighted fusion of all signals
"""

import cv2
import numpy as np
import mediapipe as mp
# from scipy.spatial import distance as dist  # Removed unused heavy dependency
from collections import deque
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

_global_face_mesh = None


@dataclass
class DetectionResult:
    """All metrics returned per frame."""
    ear: float = 0.0
    left_ear: float = 0.0
    right_ear: float = 0.0
    mar: float = 0.0
    perclos: float = 0.0
    blink_rate: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    alertness_score: float = 100.0
    is_drowsy: bool = False
    is_yawning: bool = False
    is_distracted: bool = False
    eyes_closed: bool = False
    face_detected: bool = False
    blink_total: int = 0
    frame_count: int = 0

    def to_dict(self):
        """Convert to JSON-safe dict (numpy types → Python native)."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, (np.bool_, np.generic)):
                d[k] = v.item()
        return d


class DrowsinessEngine:
    """
    Production-grade drowsiness detection using MediaPipe Face Mesh.

    Improvements over basic EAR-only approach:
    - 478 landmarks (vs dlib's 68) for higher precision
    - Yawn detection via Mouth Aspect Ratio
    - PERCLOS (industry standard) for sustained drowsiness measurement
    - Blink frequency analysis
    - Full 3D head pose estimation (not a pixel-delta hack)
    - Composite alertness score fusing all signals
    """

    # ── MediaPipe Face Mesh Landmark Indices ─────────────────────────
    # Right eye (6 points for EAR: outer, upper1, upper2, inner, lower2, lower1)
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    # Left eye
    LEFT_EYE = [362, 385, 387, 263, 373, 380]

    # Mouth landmarks for MAR
    # Vertical pairs (top → bottom of inner lip)
    MOUTH_VERT_TOP = [13, 312, 311, 310]
    MOUTH_VERT_BOT = [14, 317, 402, 318]
    # Horizontal (left corner → right corner)
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291

    # 6-point set for solvePnP head pose estimation
    POSE_LANDMARKS = [1, 152, 263, 33, 61, 291]
    # Corresponding 3D model points (generic face model, mm)
    MODEL_POINTS_3D = np.array([
        (0.0, 0.0, 0.0),            # Nose tip
        (0.0, -330.0, -65.0),       # Chin
        (-225.0, 170.0, -135.0),    # Left eye outer corner
        (225.0, 170.0, -135.0),     # Right eye outer corner
        (-150.0, -150.0, -125.0),   # Left mouth corner
        (150.0, -150.0, -125.0),    # Right mouth corner
    ], dtype=np.float64)

    # Eye contour indices for drawing (full outline)
    RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                         173, 157, 158, 159, 160, 161, 246]
    LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263,
                        466, 388, 387, 386, 385, 384, 398]
    # Lip contour for drawing
    LIP_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                 409, 270, 269, 267, 0, 37, 39, 40, 185]

    def __init__(
        self,
        ear_threshold: float = 0.22,
        mar_threshold: float = 0.65,
        consec_frames: int = 15,
        perclos_window_sec: float = 60.0,
        perclos_alert_pct: float = 40.0,
        head_pitch_limit: float = 20.0,
        head_yaw_limit: float = 25.0,
    ):
        # ── Thresholds ──
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.consec_frames = consec_frames
        self.perclos_window_sec = perclos_window_sec
        self.perclos_alert_pct = perclos_alert_pct
        self.head_pitch_limit = head_pitch_limit
        self.head_yaw_limit = head_yaw_limit

        # ── MediaPipe Face Mesh ──
        # Use a global instance to prevent OOM errors on limited memory instances (Render free tier)
        global _global_face_mesh
        try:
            if _global_face_mesh is None:
                print("Initializing MediaPipe FaceMesh singleton...")
                self.mp_face_mesh = mp.solutions.face_mesh
                _global_face_mesh = self.mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                print("MediaPipe FaceMesh successfully initialized.")
            else:
                self.mp_face_mesh = mp.solutions.face_mesh
            
            self.face_mesh = _global_face_mesh
        except Exception as e:
            print(f"FATAL ERROR initializing MediaPipe: {e}")
            self.face_mesh = None

        # ── Tracking State ──
        self.frame_count = 0
        self.closed_frame_counter = 0
        self.is_drowsy = False
        self.drowsiness_events = 0

        # PERCLOS: sliding window of (timestamp, eyes_closed_bool)
        self.perclos_buffer: deque = deque()

        # Blink tracking
        self.blink_total = 0
        self.blink_in_progress = False
        self.blink_timestamps: deque = deque()

        # EAR history for session analytics
        self.ear_history: list = []
        self.mar_history: list = []

        self.session_start = time.time()

    def reset(self):
        """Reset all tracking state for a new session."""
        self.frame_count = 0
        self.closed_frame_counter = 0
        self.is_drowsy = False
        self.drowsiness_events = 0
        self.perclos_buffer.clear()
        self.blink_total = 0
        self.blink_in_progress = False
        self.blink_timestamps.clear()
        self.ear_history.clear()
        self.mar_history.clear()
        self.session_start = time.time()

    # ── Core Metric Calculations ─────────────────────────────────────

    @staticmethod
    def _ear(eye_points: np.ndarray) -> float:
        """
        Eye Aspect Ratio.
        eye_points: 6 (x,y) points [p1..p6]
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        C = dist.euclidean(eye_points[0], eye_points[3])
        return (A + B) / (2.0 * C) if C > 0 else 0.0

    @staticmethod
    def _mar(vert_top: np.ndarray, vert_bot: np.ndarray,
             left: np.ndarray, right: np.ndarray) -> float:
        """
        Mouth Aspect Ratio — yawn indicator.
        Average of vertical distances / horizontal distance.
        """
        vert_dists = [dist.euclidean(t, b) for t, b in zip(vert_top, vert_bot)]
        avg_vert = np.mean(vert_dists)
        horiz = dist.euclidean(left, right)
        return avg_vert / horiz if horiz > 0 else 0.0

    def _compute_perclos(self, eyes_closed: bool) -> float:
        """
        PERCLOS: Percentage of Eye Closure over a time window.
        Industry-standard metric. >40% indicates severe drowsiness.
        """
        now = time.time()
        self.perclos_buffer.append((now, eyes_closed))

        # Trim buffer to window size
        cutoff = now - self.perclos_window_sec
        while self.perclos_buffer and self.perclos_buffer[0][0] < cutoff:
            self.perclos_buffer.popleft()

        if len(self.perclos_buffer) < 10:
            return 0.0

        closed_count = sum(1 for _, closed in self.perclos_buffer if closed)
        return (closed_count / len(self.perclos_buffer)) * 100.0

    def _track_blink(self, eyes_closed: bool) -> None:
        """
        Track blinks: a blink = brief closure (open → closed → open).
        Excludes sustained closures (those are drowsiness, not blinks).
        """
        now = time.time()

        if eyes_closed and not self.blink_in_progress:
            self.blink_in_progress = True
            self._blink_start_frame = self.frame_count
        elif not eyes_closed and self.blink_in_progress:
            self.blink_in_progress = False
            # Only count as blink if closure was brief (< consec_frames)
            duration = self.frame_count - getattr(self, '_blink_start_frame', 0)
            if 2 <= duration < self.consec_frames:
                self.blink_total += 1
                self.blink_timestamps.append(now)

        # Trim to 60-second window
        cutoff = now - 60.0
        while self.blink_timestamps and self.blink_timestamps[0] < cutoff:
            self.blink_timestamps.popleft()

    def _compute_blink_rate(self) -> float:
        """Blinks per minute over a 60-second rolling window."""
        if not self.blink_timestamps:
            return 0.0
        elapsed = time.time() - self.session_start
        window = min(elapsed, 60.0)
        if window < 5.0:
            return 0.0  # Not enough data yet
        return (len(self.blink_timestamps) / window) * 60.0

    def _estimate_head_pose(self, landmarks_px: list, frame_shape: tuple):
        """
        Estimate head pose (pitch, yaw, roll) using cv2.solvePnP.
        Uses a 6-point 3D face model matched to 2D image landmarks.
        Returns (pitch, yaw, roll) in degrees.
        """
        h, w = frame_shape[:2]

        image_points = np.array(
            [landmarks_px[i] for i in self.POSE_LANDMARKS],
            dtype=np.float64
        )

        # Approximate camera intrinsics
        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rvec, tvec = cv2.solvePnP(
            self.MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return 0.0, 0.0, 0.0

        # Convert rotation vector → rotation matrix → Euler angles
        rmat, _ = cv2.Rodrigues(rvec)
        proj_matrix = np.hstack((rmat, tvec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch = float(euler[0][0])
        yaw = float(euler[1][0])
        roll = float(euler[2][0])

        return pitch, yaw, roll

    def _compute_alertness_score(
        self, ear: float, mar: float, perclos: float,
        blink_rate: float, pitch: float, yaw: float
    ) -> float:
        """
        Composite alertness score (0–100). Higher = more alert.

        Weights:
          - PERCLOS:    30%  (strongest single predictor)
          - EAR:        25%  (real-time eye openness)
          - Blink Rate: 15%  (frequency anomaly)
          - MAR/Yawn:   15%  (yawning indicator)
          - Head Pose:  15%  (distraction indicator)
        """
        # EAR component: 100 when fully open, 0 when closed
        ear_norm = np.clip((ear - 0.15) / (0.35 - 0.15), 0, 1) * 100

        # PERCLOS component: 100 when 0% closed, 0 when 50%+ closed
        perclos_norm = np.clip((50 - perclos) / 50, 0, 1) * 100

        # Blink rate component: optimal ~15-20 bpm
        # Too low (<8) = microsleep risk; too high (>30) = fatigue
        if blink_rate < 1:
            blink_norm = 80.0  # Not enough data
        elif 10 <= blink_rate <= 22:
            blink_norm = 100.0
        elif blink_rate < 10:
            blink_norm = max(0, blink_rate / 10 * 100)
        else:
            blink_norm = max(0, (45 - blink_rate) / 23 * 100)

        # MAR component: 100 when not yawning, decreasing with yawn
        mar_norm = np.clip((0.7 - mar) / 0.7, 0, 1) * 100

        # Head pose component: 100 when facing forward
        pose_deviation = (abs(pitch) / self.head_pitch_limit +
                          abs(yaw) / self.head_yaw_limit) / 2.0
        pose_norm = np.clip(1.0 - pose_deviation, 0, 1) * 100

        score = (
            0.30 * perclos_norm +
            0.25 * ear_norm +
            0.15 * blink_norm +
            0.15 * mar_norm +
            0.15 * pose_norm
        )
        return round(np.clip(score, 0, 100), 1)

    # ── Main Processing Pipeline ─────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single video frame through the full detection pipeline.

        Args:
            frame: BGR image (numpy array)

        Returns:
            (annotated_frame, DetectionResult)
        """
        self.frame_count += 1
        result = DetectionResult(frame_count=self.frame_count)
        h, w = frame.shape[:2]

        # Convert BGR → RGB for MediaPipe
        if self.face_mesh is None:
            return frame, result

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        mp_results = self.face_mesh.process(rgb)
        rgb.flags.writeable = True

        if not mp_results.multi_face_landmarks:
            self.closed_frame_counter = 0
            self.is_drowsy = False
            result.face_detected = False
            result.blink_total = self.blink_total
            return frame, result

        face = mp_results.multi_face_landmarks[0]
        result.face_detected = True

        # Convert normalized landmarks → pixel coordinates
        lm_px = [(int(lm.x * w), int(lm.y * h)) for lm in face.landmark]

        # ── 1. EAR ──
        right_pts = np.array([lm_px[i] for i in self.RIGHT_EYE])
        left_pts = np.array([lm_px[i] for i in self.LEFT_EYE])
        result.right_ear = self._ear(right_pts)
        result.left_ear = self._ear(left_pts)
        result.ear = (result.left_ear + result.right_ear) / 2.0

        eyes_closed = result.ear < self.ear_threshold
        result.eyes_closed = eyes_closed

        # ── 2. MAR (Yawn) ──
        vert_top = np.array([lm_px[i] for i in self.MOUTH_VERT_TOP])
        vert_bot = np.array([lm_px[i] for i in self.MOUTH_VERT_BOT])
        mouth_l = np.array(lm_px[self.MOUTH_LEFT])
        mouth_r = np.array(lm_px[self.MOUTH_RIGHT])
        result.mar = self._mar(vert_top, vert_bot, mouth_l, mouth_r)
        result.is_yawning = result.mar > self.mar_threshold

        # ── 3. PERCLOS ──
        result.perclos = self._compute_perclos(eyes_closed)

        # ── 4. Blink Rate ──
        self._track_blink(eyes_closed)
        result.blink_rate = self._compute_blink_rate()
        result.blink_total = self.blink_total

        # ── 5. Head Pose ──
        result.pitch, result.yaw, result.roll = self._estimate_head_pose(
            lm_px, frame.shape
        )
        result.is_distracted = (
            abs(result.pitch) > self.head_pitch_limit or
            abs(result.yaw) > self.head_yaw_limit
        )

        # ── 6. Drowsiness Logic ──
        if eyes_closed:
            self.closed_frame_counter += 1
            if self.closed_frame_counter >= self.consec_frames:
                if not self.is_drowsy:
                    self.drowsiness_events += 1
                self.is_drowsy = True
                result.is_drowsy = True
        else:
            self.closed_frame_counter = 0
            self.is_drowsy = False

        # Also flag drowsy if PERCLOS is critical
        if result.perclos > self.perclos_alert_pct:
            result.is_drowsy = True

        # ── 7. Alertness Score ──
        result.alertness_score = self._compute_alertness_score(
            result.ear, result.mar, result.perclos,
            result.blink_rate, result.pitch, result.yaw
        )

        # ── History ──
        self.ear_history.append(result.ear)
        self.mar_history.append(result.mar)
        if len(self.ear_history) > 2000:
            self.ear_history = self.ear_history[-2000:]
        if len(self.mar_history) > 2000:
            self.mar_history = self.mar_history[-2000:]

        # ── Annotate Frame ──
        annotated = self._draw_annotations(frame.copy(), lm_px, result)
        return annotated, result

    # ── Frame Annotation / Drawing ───────────────────────────────────

    def _draw_annotations(
        self, frame: np.ndarray, lm_px: list, result: DetectionResult
    ) -> np.ndarray:
        """Draw all visual overlays on the frame."""
        h, w = frame.shape[:2]

        # Eye contours
        eye_color = (0, 0, 255) if result.eyes_closed else (0, 255, 100)
        for contour_ids in [self.RIGHT_EYE_CONTOUR, self.LEFT_EYE_CONTOUR]:
            pts = np.array([lm_px[i] for i in contour_ids], np.int32)
            cv2.polylines(frame, [pts], True, eye_color, 1, cv2.LINE_AA)

        # Mouth contour
        mouth_color = (0, 200, 255) if result.is_yawning else (0, 255, 100)
        lip_pts = np.array([lm_px[i] for i in self.LIP_OUTER], np.int32)
        cv2.polylines(frame, [lip_pts], True, mouth_color, 1, cv2.LINE_AA)

        # Head pose axis (nose tip → projected direction)
        nose = lm_px[1]
        pitch_arrow = (nose[0], nose[1] - int(result.pitch * 2))
        yaw_arrow = (nose[0] + int(result.yaw * 2), nose[1])
        cv2.arrowedLine(frame, nose, pitch_arrow, (255, 100, 0), 2, tipLength=0.3)
        cv2.arrowedLine(frame, nose, yaw_arrow, (0, 100, 255), 2, tipLength=0.3)

        # Alert overlays
        if result.is_drowsy:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 200), -1)
            frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
            self._draw_centered_text(frame, "DROWSINESS DETECTED", w, h,
                                     (255, 255, 255), scale=1.2)

        if result.is_yawning and not result.is_drowsy:
            self._draw_banner(frame, "YAWN DETECTED", (0, 180, 255), w)

        if result.is_distracted and not result.is_drowsy:
            direction = ""
            if abs(result.yaw) > self.head_yaw_limit:
                direction = "LEFT" if result.yaw < 0 else "RIGHT"
            elif result.pitch > self.head_pitch_limit:
                direction = "DOWN"
            elif result.pitch < -self.head_pitch_limit:
                direction = "UP"
            self._draw_banner(frame, f"DISTRACTED: LOOKING {direction}",
                              (0, 100, 255), w)

        # Metrics overlay (top-left)
        y_offset = 25
        metrics = [
            (f"EAR: {result.ear:.3f}", (180, 255, 100)),
            (f"MAR: {result.mar:.3f}", (180, 255, 100)),
            (f"PERCLOS: {result.perclos:.1f}%", (180, 255, 100)),
            (f"Blinks/min: {result.blink_rate:.1f}", (180, 255, 100)),
            (f"Pitch: {result.pitch:.1f}  Yaw: {result.yaw:.1f}", (255, 200, 100)),
            (f"Alertness: {result.alertness_score:.0f}/100", self._score_color(result.alertness_score)),
        ]
        for text, color in metrics:
            cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            y_offset += 22

        return frame

    @staticmethod
    def _draw_centered_text(frame, text, w, h, color, scale=1.0):
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)[0]
        x = (w - size[0]) // 2
        y = (h + size[1]) // 2
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                    scale, (0, 0, 0), 4)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                    scale, color, 2)

    @staticmethod
    def _draw_banner(frame, text, color, w):
        cv2.rectangle(frame, (0, 0), (w, 35), color, -1)
        cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

    @staticmethod
    def _score_color(score):
        if score >= 70:
            return (100, 255, 100)
        elif score >= 40:
            return (0, 200, 255)
        else:
            return (0, 0, 255)

    def get_session_summary(self) -> dict:
        """Return summary data for saving a completed session."""
        elapsed = time.time() - self.session_start
        return {
            "total_time": int(elapsed),
            "drowsiness_events": self.drowsiness_events,
            "total_blinks": self.blink_total,
            "avg_ear": float(np.mean(self.ear_history)) if self.ear_history else 0.0,
            "min_ear": float(np.min(self.ear_history)) if self.ear_history else 0.0,
            "max_ear": float(np.max(self.ear_history)) if self.ear_history else 0.0,
            "avg_mar": float(np.mean(self.mar_history)) if self.mar_history else 0.0,
            "ear_history": self.ear_history[-200:],
            "mar_history": self.mar_history[-200:],
        }
