import base64
import logging
from io import BytesIO

import cv2
import inference
import numpy as np
from PIL import Image


class DetectionService:
    def __init__(self):
        # Setup logging
        self.logger = logging.getLogger(__name__)

        try:
            # Load model dari inference platform
            self.model = inference.get_model("sticker-hqw2u/6")
            self.logger.info("Model berhasil dimuat")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def detect_plate(self, frame):
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
