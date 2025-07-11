import base64
import json
import logging

import cv2
import numpy as np
from flask import Blueprint
from flask_socketio import emit

from app import socketio
from app.services.detection_service import DetectionService

websocket_bp = Blueprint("websocket", __name__)
detection_service = DetectionService()
logger = logging.getLogger(__name__)


@socketio.on("connect")
def handle_connect():
    logger.info("Client connected")


@socketio.on("disconnect")
def handle_disconnect():
    logger.info("Client disconnected")


@socketio.on("message")
def handle_message(message):
    try:
        if isinstance(message, str):
            message = json.loads(message)

        if message.get("type") == "frame":
            logger.debug("Frame diterima dari client")
            # Proses frame dengan model deteksi
            result = detection_service.detect_plate(message["data"])

            if result:
                logger.info(f"Plat nomor terdeteksi: {result}")
                # Kirim hasil deteksi ke client
                emit("detection", {"type": "detection", "license_plate": result})
            else:
                logger.debug("Tidak ada plat nomor terdeteksi")

    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        emit("error", {"type": "error", "message": str(e)})
