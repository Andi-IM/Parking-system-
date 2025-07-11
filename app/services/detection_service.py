import logging
from io import BytesIO

import cv2
import inference
from PIL import Image
from ultralytics import YOLO


class DetectionService:
    """
    A service class for handling vehicle detection and license plate recognition.

    This class provides functionality for:
    - Loading and managing detection models (YOLO and custom inference)
    - License plate detection and recognition
    - Motorcycle detection and tracking
    - Vehicle counting for entry/exit monitoring
    """


def __init__(self):
    # Setup logging
    self.logger = logging.getLogger(__name__)

    try:
        # Load model dari inference platform
        self.model = inference.get_model("sticker-hqw2u/6")
        self.logger.info("Model berhasil dimuat")

        # Load YOLO model untuk deteksi motor
        self.yolo_model = YOLO("best.pt")
        self.logger.info("YOLO model berhasil dimuat")

        # Counting Line Coordinates untuk resolusi 720x480
        # Format: [(x1, y1), (x2, y2)] - defines start and end points of counting lines
        self.ENTRY_LINE = [
            (258, 312),
            (388, 248),
        ]  # Green line for vehicles entering
        self.EXIT_LINE = [(287, 341), (420, 267)]  # Red line for vehicles exiting

        # Tracking variables
        self.tracked_objects = {}  # Dictionary untuk menyimpan objek yang di-track
        self.entry_count = 0  # Counter motor masuk
        self.exit_count = 0  # Counter motor keluar
        self.next_object_id = 1  # ID untuk objek baru

    except Exception as e:
        self.logger.error(f"Error loading model: {str(e)}")
        raise


def detect_plate(self, frame):
    """
    Detects and recognizes license plates in the given frame.

    Args:
        frame: Image frame in either base64 string format or numpy array (OpenCV format)
            If base64, should start with "data:image"
            If numpy array, should be in BGR color format

    Returns:
        str or None: Detected license plate number if found, None otherwise

    Raises:
        Exception: If there is an error during plate detection or recognition
    """
    try:
        # Convert frame ke format yang sesuai untuk model
        if isinstance(frame, str) and frame.startswith("data:image"):
            # Handle base64 image
            img_data = base64.b64decode(frame.split(",")[1])
            img = Image.open(BytesIO(img_data))
            self.logger.debug("Frame diterima dalam format base64")
        else:
            # Handle numpy array (OpenCV format)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.logger.debug("Frame diterima dalam format numpy array")

        # Lakukan inferensi
        self.logger.debug("Memulai inferensi model")
        result = self.model.infer(image=img)
        self.logger.debug(f"Hasil inferensi: {result}")

        # Proses hasil deteksi
        if result and len(result) > 0:
            # Ambil hasil deteksi pertama
            plate_number = result[0]
            self.logger.info(f"Plat nomor terdeteksi: {plate_number}")
            return plate_number

        self.logger.debug("Tidak ada plat nomor terdeteksi")
        return None

    except Exception as e:
        self.logger.error(f"Error in plate detection: {str(e)}")
        return None


def _calculate_distance(self, point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def _get_center_point(self, bbox):
    """Get center point of bounding box"""
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)


def _line_intersection(self, point, line_start, line_end, threshold=20):
    """Check if point is near the line within threshold distance"""
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end

    # Calculate distance from point to line
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2

    distance = abs(A * x0 + B * y0 + C) / math.sqrt(A**2 + B**2)

    return distance <= threshold


def _assign_object_id(self, current_detections):
    """Assign IDs to detected objects using simple distance-based tracking"""
    if not self.tracked_objects:
        # First frame - assign new IDs to all detections
        for i, detection in enumerate(current_detections):
            detection["id"] = self.next_object_id
            detection["crossed_entry"] = False
            detection["crossed_exit"] = False
            self.tracked_objects[self.next_object_id] = detection
            self.next_object_id += 1
    else:
        # Match current detections with tracked objects
        matched_ids = set()

        for detection in current_detections:
            center = self._get_center_point(detection["bbox"])
            best_match_id = None
            min_distance = float("inf")

            # Find closest tracked object
            for obj_id, tracked_obj in self.tracked_objects.items():
                if obj_id in matched_ids:
                    continue

                tracked_center = self._get_center_point(tracked_obj["bbox"])
                distance = self._calculate_distance(center, tracked_center)

                # Match if distance is reasonable (less than 100 pixels)
                if distance < 100 and distance < min_distance:
                    min_distance = distance
                    best_match_id = obj_id

            if best_match_id:
                # Update existing object
                detection["id"] = best_match_id
                detection["crossed_entry"] = self.tracked_objects[best_match_id]["crossed_entry"]
                detection["crossed_exit"] = self.tracked_objects[best_match_id]["crossed_exit"]
                self.tracked_objects[best_match_id] = detection
                matched_ids.add(best_match_id)
            else:
                # New object
                detection["id"] = self.next_object_id
                detection["crossed_entry"] = False
                detection["crossed_exit"] = False
                self.tracked_objects[self.next_object_id] = detection
                self.next_object_id += 1

        # Remove objects that weren't matched (disappeared)
        current_ids = {det["id"] for det in current_detections}
        self.tracked_objects = {k: v for k, v in self.tracked_objects.items() if k in current_ids}


def _check_line_crossing(self, detections):
    """Check if any tracked object crosses entry or exit lines"""
    for detection in detections:
        obj_id = detection["id"]
        center = self._get_center_point(detection["bbox"])

        # Check entry line crossing
        if (
            not detection["crossed_entry"]
            and not detection["crossed_exit"]
            and self._line_intersection(center, self.ENTRY_LINE[0], self.ENTRY_LINE[1])
        ):

            detection["crossed_entry"] = True
            self.tracked_objects[obj_id]["crossed_entry"] = True
            self.entry_count += 1
            self.logger.info(f"Motor masuk terdeteksi! Total masuk: {self.entry_count}")

        # Check exit line crossing
        elif (
            not detection["crossed_exit"]
            and not detection["crossed_entry"]
            and self._line_intersection(center, self.EXIT_LINE[0], self.EXIT_LINE[1])
        ):

            detection["crossed_exit"] = True
            self.tracked_objects[obj_id]["crossed_exit"] = True
            self.exit_count += 1
            self.logger.info(f"Motor keluar terdeteksi! Total keluar: {self.exit_count}")


def detect_motorcycle(self, frame):
    try:
        # Convert frame jika diperlukan
        if isinstance(frame, str) and frame.startswith("data:image"):
            # Handle base64 image
            img_data = base64.b64decode(frame.split(",")[1])
            img = Image.open(BytesIO(img_data))
            # Convert PIL to numpy array
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.logger.debug("Frame diterima dalam format base64")

        # Lakukan inferensi dengan YOLO
        self.logger.debug("Memulai deteksi motor dengan YOLO")
        results = self.yolo_model(frame)

        motorcycles = []

        # Proses hasil deteksi
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Ambil class ID dan confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    # Cek apakah objek yang terdeteksi adalah motor
                    # Asumsi class_id untuk motor (sesuaikan dengan model Anda)
                    if class_id == 3 and confidence > 0.5:  # threshold confidence 0.5
                        # Ambil koordinat bounding box
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        motorcycle_info = {
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": confidence,
                            "class_id": class_id,
                        }
                        motorcycles.append(motorcycle_info)

        # Assign IDs dan track objects
        if motorcycles:
            self._assign_object_id(motorcycles)

            # Check line crossing untuk counting
            self._check_line_crossing(motorcycles)

            self.logger.info(f"Total {len(motorcycles)} motor terdeteksi")

            # Return hasil dengan informasi counting
            return {
                "detections": motorcycles,
                "entry_count": self.entry_count,
                "exit_count": self.exit_count,
                "entry_line": self.ENTRY_LINE,
                "exit_line": self.EXIT_LINE,
            }
        else:
            self.logger.debug("Tidak ada motor terdeteksi")
            return {
                "detections": [],
                "entry_count": self.entry_count,
                "exit_count": self.exit_count,
                "entry_line": self.ENTRY_LINE,
                "exit_line": self.EXIT_LINE,
            }

    except Exception as e:
        self.logger.error(f"Error in motorcycle detection: {str(e)}")
        return {
            "detections": [],
            "entry_count": self.entry_count,
            "exit_count": self.exit_count,
            "entry_line": self.ENTRY_LINE,
            "exit_line": self.EXIT_LINE,
        }


def reset_counters(self):
    """Reset counting statistics"""
    self.entry_count = 0
    self.exit_count = 0
    self.tracked_objects = {}
    self.next_object_id = 1
    self.logger.info("Counter direset")


def get_counting_stats(self):
    """Get current counting statistics"""
    return {
        "entry_count": self.entry_count,
        "exit_count": self.exit_count,
        "net_count": self.entry_count - self.exit_count,
    }
