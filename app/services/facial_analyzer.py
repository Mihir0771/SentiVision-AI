"""
Real-time facial emotion detection using OpenCV and FER.

Supports webcam capture, single-image analysis, in-memory byte analysis, and
continuous camera streaming with annotated overlays.

Backends
--------
- **FER** (default): ``fer.FER`` wrapping OpenCV + TensorFlow/MTCNN.
- **DeepFace** (fallback): ``deepface.DeepFace`` for emotion analysis when FER
  is unavailable.

Both backends are lazy-loaded; the module degrades gracefully when neither is
installed (returning empty results with descriptive error messages).
"""

from __future__ import annotations

import logging
import os
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency probes
# ---------------------------------------------------------------------------

try:
    import cv2  # type: ignore

    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CV2_AVAILABLE = False
    logger.warning("OpenCV (cv2) not installed – facial analysis will be limited.")

try:
    import numpy as np  # type: ignore

    _NP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NP_AVAILABLE = False
    logger.warning("numpy not installed – facial analysis will be unavailable.")

try:
    from fer import FER as _FER  # type: ignore

    _FER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FER_AVAILABLE = False
    logger.info("FER not installed – will try DeepFace as fallback.")

try:
    from deepface import DeepFace as _DeepFace  # type: ignore

    _DEEPFACE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _DEEPFACE_AVAILABLE = False
    logger.info("DeepFace not installed.")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EMOTION_LABELS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")

# BGR colours used for annotation overlays
_EMOTION_COLORS: Dict[str, Tuple[int, int, int]] = {
    "angry": (0, 0, 255),       # red
    "disgust": (0, 140, 255),   # dark orange
    "fear": (0, 255, 255),      # yellow
    "happy": (0, 255, 0),       # green
    "sad": (255, 0, 0),         # blue
    "surprise": (255, 0, 255),  # magenta
    "neutral": (200, 200, 200), # light grey
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX if _CV2_AVAILABLE else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timestamp() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _empty_result(error: Optional[str] = None) -> Dict[str, Any]:
    """Return a well-formed but empty analysis result."""
    result: Dict[str, Any] = {
        "faces_detected": 0,
        "faces": [],
        "analyzed_at": _timestamp(),
    }
    if error:
        result["error"] = error
    return result


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class FacialEmotionAnalyzer:
    """Facial emotion detection engine with FER / DeepFace backends.

    Parameters
    ----------
    backend : str, optional
        ``"fer"`` (default) or ``"deepface"``.  Falls back automatically when
        the requested backend is unavailable.
    mtcnn : bool, optional
        Whether to use MTCNN face detector with FER (slower but more accurate).
        Defaults to ``True``.
    """

    def __init__(
        self,
        backend: str = "onnx",
        mtcnn: bool = True,
    ) -> None:
        self._backend = backend.lower()
        self._mtcnn = mtcnn
        self._fer_detector: Any = None
        self._onnx_detector: Any = None
        self._onnx_emotion_net: Any = None
        logger.info(
            "FacialEmotionAnalyzer created (backend=%s, mtcnn=%s).",
            self._backend,
            self._mtcnn,
        )

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _get_onnx_models(self) -> Tuple[Any, Any]:
        """Load or download high-performance YuNet face detector and FERPlus emotion net."""
        if self._onnx_detector is not None and self._onnx_emotion_net is not None:
            return self._onnx_detector, self._onnx_emotion_net

        if not _CV2_AVAILABLE:
            return None, None

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models_weights"))
        os.makedirs(base_dir, exist_ok=True)
        detector_path = os.path.join(base_dir, "face_detection_yunet.onnx")
        emotion_path = os.path.join(base_dir, "emotion_ferplus.onnx")

        # Auto-download detector if missing
        if not os.path.exists(detector_path) or os.path.getsize(detector_path) < 1000:
            try:
                logger.info("Downloading YuNet face detection model...")
                url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
                urllib.request.urlretrieve(url, detector_path)
            except Exception as e:
                logger.warning("Could not download YuNet model: %s", e)

        # Auto-download emotion net if missing
        if not os.path.exists(emotion_path) or os.path.getsize(emotion_path) < 1000:
            try:
                logger.info("Downloading FERPlus emotion recognition model...")
                url = "https://huggingface.co/onnxmodelzoo/emotion-ferplus-8/resolve/main/emotion-ferplus-8.onnx"
                urllib.request.urlretrieve(url, emotion_path)
            except Exception as e:
                logger.warning("Could not download FERPlus model: %s", e)

        try:
            if hasattr(cv2, "FaceDetectorYN_create") and os.path.exists(detector_path):
                self._onnx_detector = cv2.FaceDetectorYN_create(detector_path, "", (320, 240), 0.5, 0.3)
            if hasattr(cv2, "dnn") and hasattr(cv2.dnn, "readNetFromONNX") and os.path.exists(emotion_path):
                self._onnx_emotion_net = cv2.dnn.readNetFromONNX(emotion_path)
            return self._onnx_detector, self._onnx_emotion_net
        except Exception as exc:
            logger.error("Failed to initialise ONNX facial models: %s", exc)
            return None, None

    def _get_fer(self) -> Any:
        """Return the FER detector, initialising on first call."""
        if self._fer_detector is not None:
            return self._fer_detector
        if not _FER_AVAILABLE:
            return None
        try:
            self._fer_detector = _FER(mtcnn=self._mtcnn)
            logger.info("FER detector loaded (mtcnn=%s).", self._mtcnn)
            return self._fer_detector
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to initialise FER: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze_frame(self, frame: "np.ndarray") -> Dict[str, Any]:
        """Analyse a single BGR image frame for facial emotions.

        Parameters
        ----------
        frame : numpy.ndarray
            BGR image array (as returned by ``cv2.imread`` or webcam capture).

        Returns
        -------
        dict
            Analysis result with ``faces_detected``, ``faces`` list, and
            ``analyzed_at`` timestamp.
        """
        if not _NP_AVAILABLE:
            return _empty_result("numpy is not installed.")

        if frame is None or not hasattr(frame, "shape"):
            return _empty_result("Invalid frame provided.")

        # 1. Primary High-Performance Engine (YuNet + FERPlus ONNX)
        onnx_res = self._analyze_with_onnx(frame)
        if onnx_res is not None:
            return onnx_res

        # 2. Fallback to FER library
        if _FER_AVAILABLE:
            result = self._analyze_with_fer(frame)
            if result is not None:
                return result

        # 3. Fallback to DeepFace
        if _DEEPFACE_AVAILABLE:
            result = self._analyze_with_deepface(frame)
            if result is not None:
                return result

        return _empty_result("No facial detection backend available.")

    def _analyze_with_onnx(self, frame: "np.ndarray") -> Optional[Dict[str, Any]]:
        """Run OpenCV YuNet + FERPlus ONNX detection and emotion classification."""
        detector, emotion_net = self._get_onnx_models()
        if detector is None or emotion_net is None:
            return None

        try:
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            h, w = frame.shape[:2]
            detector.setInputSize((w, h))
            detector.setScoreThreshold(0.35)
            detector.setNMSThreshold(0.3)
            retval, detected_faces = detector.detect(frame)

            if detected_faces is None or len(detected_faces) == 0:
                return {
                    "faces_detected": 0,
                    "faces": [],
                    "image_width": w,
                    "image_height": h,
                    "analyzed_at": _timestamp(),
                }

            faces: List[Dict[str, Any]] = []
            for face in detected_faces:
                x = max(0, int(face[0]))
                y = max(0, int(face[1]))
                box_w = min(w - x, int(face[2]))
                box_h = min(h - y, int(face[3]))
                face_score = float(face[14]) if len(face) > 14 else 0.9

                if box_w < 10 or box_h < 10:
                    continue

                # Add 10% padding for facial emotion context
                pad_x = int(box_w * 0.1)
                pad_y = int(box_h * 0.1)
                crop_x1 = max(0, x - pad_x)
                crop_y1 = max(0, y - pad_y)
                crop_x2 = min(w, x + box_w + pad_x)
                crop_y2 = min(h, y + box_h + pad_y)
                face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                if face_crop.size == 0:
                    continue

                # Prepare for FERPlus (64x64 grayscale with adaptive CLAHE contrast enhancement)
                gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced_face = clahe.apply(gray_face)

                blob = cv2.dnn.blobFromImage(enhanced_face, 1.0, (64, 64), (0,))
                emotion_net.setInput(blob)
                out = emotion_net.forward()[0]

                # Extract 7 basic emotions (neutral, happy, surprise, sad, angry, disgust, fear)
                raw_logits = np.array(out[:7], dtype=np.float32)

                # Prior calibration: dampen neutral bias inherent in raw dataset
                calibrated_logits = raw_logits.copy()
                calibrated_logits[0] -= 0.35  # neutral baseline adjustment

                # Multi-modal fusion with YuNet 5-point facial landmark geometry
                if len(face) >= 14:
                    re_x, re_y = float(face[4]), float(face[5])
                    le_x, le_y = float(face[6]), float(face[7])
                    nose_x, nose_y = float(face[8]), float(face[9])
                    rm_x, rm_y = float(face[10]), float(face[11])
                    lm_x, lm_y = float(face[12]), float(face[13])

                    # Inter-ocular distance (baseline scale for face geometry)
                    inter_ocular = max(1.0, float(np.hypot(le_x - re_x, le_y - re_y)))

                    # Mouth width expansion (zygomaticus major activation / smile)
                    mouth_width = float(np.hypot(lm_x - rm_x, lm_y - rm_y))
                    mouth_eye_ratio = mouth_width / inter_ocular

                    # Jaw opening ratio (mouth center to nose distance)
                    mouth_center_y = (rm_y + lm_y) / 2.0
                    jaw_drop_ratio = (mouth_center_y - nose_y) / inter_ocular

                    # Responsive smile boost when mouth widens
                    if mouth_eye_ratio > 0.90:
                        smile_factor = min(1.5, (mouth_eye_ratio - 0.90) / 0.15)
                        calibrated_logits[1] += 1.2 * smile_factor
                        calibrated_logits[0] -= 0.8 * smile_factor

                    # Responsive surprise boost when jaw drops
                    if jaw_drop_ratio > 0.85:
                        surprise_factor = min(1.5, (jaw_drop_ratio - 0.85) / 0.15)
                        calibrated_logits[2] += 1.3 * surprise_factor
                        calibrated_logits[0] -= 0.7 * surprise_factor

                # Temperature scaling (T=0.8 for crisp, decisive confidence peaks)
                temperature = 0.80
                scaled_logits = calibrated_logits / temperature
                exp_scores = np.exp(scaled_logits - np.max(scaled_logits))
                probs = exp_scores / np.sum(exp_scores)

                emotions = {
                    "neutral": round(float(probs[0]), 4),
                    "happy": round(float(probs[1]), 4),
                    "surprise": round(float(probs[2]), 4),
                    "sad": round(float(probs[3]), 4),
                    "angry": round(float(probs[4]), 4),
                    "disgust": round(float(probs[5]), 4),
                    "fear": round(float(probs[6]), 4),
                }

                dominant = max(emotions, key=emotions.get)
                confidence = emotions[dominant]

                faces.append({
                    "box": {
                        "x": x,
                        "y": y,
                        "w": box_w,
                        "h": box_h,
                    },
                    "emotions": emotions,
                    "dominant_emotion": dominant,
                    "confidence": confidence,
                    "face_confidence": round(face_score, 4),
                })

            return {
                "faces_detected": len(faces),
                "faces": faces,
                "image_width": w,
                "image_height": h,
                "analyzed_at": _timestamp(),
            }
        except Exception as exc:
            logger.error("ONNX Face+Emotion analysis error: %s", exc)
            return None

    def _analyze_with_fer(self, frame: "np.ndarray") -> Optional[Dict[str, Any]]:
        """Run FER-based analysis on *frame*."""
        detector = self._get_fer()
        if detector is None:
            return None
        try:
            raw_results = detector.detect_emotions(frame)
            faces: List[Dict[str, Any]] = []
            for face_data in raw_results:
                box = face_data.get("box", (0, 0, 0, 0))
                emotions_raw = face_data.get("emotions", {})
                # Normalise emotion keys to lowercase
                emotions = {k.lower(): round(float(v), 4) for k, v in emotions_raw.items()}
                dominant = max(emotions, key=emotions.get) if emotions else "neutral"  # type: ignore[arg-type]
                confidence = round(emotions.get(dominant, 0.0), 4)
                faces.append(
                    {
                        "box": {
                            "x": int(box[0]),
                            "y": int(box[1]),
                            "w": int(box[2]),
                            "h": int(box[3]),
                        },
                        "emotions": emotions,
                        "dominant_emotion": dominant,
                        "confidence": confidence,
                    }
                )
            return {
                "faces_detected": len(faces),
                "faces": faces,
                "image_width": w,
                "image_height": h,
                "analyzed_at": _timestamp(),
            }
        except Exception as exc:
            logger.error("FER analysis failed: %s", exc)
            return None

    def _analyze_with_deepface(self, frame: "np.ndarray") -> Optional[Dict[str, Any]]:
        """Run DeepFace-based emotion analysis on *frame*."""
        if not _DEEPFACE_AVAILABLE:
            return None
        try:
            results = _DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
            # DeepFace may return a single dict or a list
            if isinstance(results, dict):
                results = [results]

            faces: List[Dict[str, Any]] = []
            for face_data in results:
                region = face_data.get("region", {})
                emotions_raw = face_data.get("emotion", {})
                emotions = {k.lower(): round(float(v) / 100.0, 4) for k, v in emotions_raw.items()}
                dominant = face_data.get("dominant_emotion", "neutral").lower()
                confidence = round(emotions.get(dominant, 0.0), 4)
                faces.append(
                    {
                        "box": {
                            "x": int(region.get("x", 0)),
                            "y": int(region.get("y", 0)),
                            "w": int(region.get("w", 0)),
                            "h": int(region.get("h", 0)),
                        },
                        "emotions": emotions,
                        "dominant_emotion": dominant,
                        "confidence": confidence,
                    }
                )

            return {
                "faces_detected": len(faces),
                "faces": faces,
                "image_width": w,
                "image_height": h,
                "analyzed_at": _timestamp(),
            }
        except Exception as exc:
            logger.error("DeepFace analysis failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # File / bytes helpers
    # ------------------------------------------------------------------

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyse an image file on disk.

        Parameters
        ----------
        image_path : str
            Path to a JPEG / PNG image file.

        Returns
        -------
        dict
            Analysis result.
        """
        if not _CV2_AVAILABLE:
            return _empty_result("OpenCV not installed – cannot read image files.")
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                return _empty_result(f"Failed to read image at '{image_path}'.")
            return self.analyze_frame(frame)
        except Exception as exc:
            logger.error("analyze_image failed for '%s': %s", image_path, exc)
            return _empty_result(str(exc))

    def analyze_image_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """Analyse an image provided as raw bytes (e.g. from a WebSocket).

        Parameters
        ----------
        image_bytes : bytes
            Raw image bytes (JPEG, PNG, etc.).

        Returns
        -------
        dict
            Analysis result.
        """
        if not _CV2_AVAILABLE or not _NP_AVAILABLE:
            return _empty_result("OpenCV/numpy not installed.")
        try:
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return _empty_result("Could not decode image bytes.")
            return self.analyze_frame(frame)
        except Exception as exc:
            logger.error("analyze_image_bytes failed: %s", exc)
            return _empty_result(str(exc))

    # ------------------------------------------------------------------
    # Annotation / drawing
    # ------------------------------------------------------------------

    @staticmethod
    def get_emotion_color(emotion: str) -> Tuple[int, int, int]:
        """Return a BGR colour tuple for the given emotion label.

        Parameters
        ----------
        emotion : str
            Emotion name (case-insensitive).

        Returns
        -------
        tuple of int
            ``(B, G, R)`` colour.
        """
        return _EMOTION_COLORS.get(emotion.lower(), (200, 200, 200))

    def draw_annotations(
        self,
        frame: "np.ndarray",
        results: Dict[str, Any],
    ) -> "np.ndarray":
        """Draw bounding boxes and emotion labels on *frame*.

        Parameters
        ----------
        frame : numpy.ndarray
            The original BGR image.
        results : dict
            Output from :meth:`analyze_frame`.

        Returns
        -------
        numpy.ndarray
            Annotated copy of the frame.
        """
        if not _CV2_AVAILABLE or not _NP_AVAILABLE:
            return frame

        annotated = frame.copy()
        for face in results.get("faces", []):
            box = face.get("box", {})
            x, y, w, h = box.get("x", 0), box.get("y", 0), box.get("w", 0), box.get("h", 0)
            emotion = face.get("dominant_emotion", "neutral")
            confidence = face.get("confidence", 0.0)
            colour = self.get_emotion_color(emotion)

            # Rectangle
            cv2.rectangle(annotated, (x, y), (x + w, y + h), colour, 2)

            # Label
            label = f"{emotion} ({confidence:.0%})"
            label_y = max(y - 10, 20)
            cv2.putText(annotated, label, (x, label_y), _FONT, 0.7, colour, 2, cv2.LINE_AA)

        return annotated

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------

    def start_camera_stream(
        self,
        camera_id: int = 0,
    ) -> Generator[Tuple["np.ndarray", Dict[str, Any]], None, None]:
        """Yield annotated frames and analysis results from a live camera.

        Parameters
        ----------
        camera_id : int
            OpenCV camera index.

        Yields
        ------
        tuple of (numpy.ndarray, dict)
            ``(annotated_frame, analysis_result)`` for each captured frame.
        """
        if not _CV2_AVAILABLE:
            logger.error("OpenCV not installed – cannot start camera stream.")
            return

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error("Camera %d could not be opened.", camera_id)
            return

        try:
            logger.info("Camera stream started (id=%d).", camera_id)
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera %d.", camera_id)
                    break
                result = self.analyze_frame(frame)
                annotated = self.draw_annotations(frame, result)
                yield annotated, result
        finally:
            cap.release()
            logger.info("Camera %d released.", camera_id)

    def capture_snapshot(
        self,
        camera_id: int = 0,
    ) -> Tuple[Optional["np.ndarray"], Dict[str, Any]]:
        """Capture a single frame from the camera and analyse it.

        Parameters
        ----------
        camera_id : int
            OpenCV camera index.

        Returns
        -------
        tuple of (numpy.ndarray or None, dict)
            ``(frame, analysis_result)``.
        """
        if not _CV2_AVAILABLE:
            return None, _empty_result("OpenCV not installed.")

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            return None, _empty_result(f"Camera {camera_id} could not be opened.")

        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                return None, _empty_result("Failed to capture frame.")
            result = self.analyze_frame(frame)
            return frame, result
        finally:
            cap.release()
