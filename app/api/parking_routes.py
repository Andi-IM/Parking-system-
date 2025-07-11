import logging

import cv2
from flask import Blueprint, Response, jsonify, request

from app.services.parking_service import parking_service
from app.utils.validators import validate_license_plate

# Membuat blueprint untuk routing parkir
parking_bp = Blueprint("parking", __name__)
logger = logging.getLogger(__name__)

video_bp = Blueprint("video", __name__)


def gen_frames():
    cap = cv2.VideoCapture(0)  # 0 untuk webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Encode frame ke JPEG
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            # Yield frame dalam format MJPEG
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@video_bp.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@parking_bp.route("/active", methods=["GET"])
def get_active_parking():
    """Mengambil daftar kendaraan yang sedang parkir"""
    try:
        records = parking_service.get_active_parking()
        logger.info(f"Retrieved {len(records)} active parking records")
        return jsonify({"status": "success", "data": records, "count": len(records)})
    except Exception as e:
        logger.error(f"Error in get_active_parking: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@parking_bp.route("/history", methods=["GET"])
def get_parking_history():
    """Mengambil riwayat parkir"""
    try:
        # Mengambil parameter limit dan status
        limit = request.args.get("limit", default=100, type=int)
        status = request.args.get("status", default=None)

        records = parking_service.get_parking_history(limit=limit, status=status)
        logger.info(f"Retrieved {len(records)} parking history records")
        return jsonify({"status": "success", "data": records, "count": len(records)})
    except Exception as e:
        logger.error(f"Error in get_parking_history: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@parking_bp.route("/entry", methods=["POST"])
def record_entry():
    """Mencatat kendaraan masuk"""
    try:
        data = request.get_json()
        if not data or "license_plate" not in data:
            return (
                jsonify(
                    {"status": "error", "message": "Data plat nomor tidak ditemukan"}
                ),
                400,
            )

        license_plate = data["license_plate"].strip().upper()
        if not validate_license_plate(license_plate):
            return (
                jsonify(
                    {"status": "error", "message": "Format plat nomor tidak valid"}
                ),
                400,
            )

        record = parking_service.add_parking_record(license_plate, "MASUK")
        logger.info(f"Recorded entry for plate: {license_plate}")
        return jsonify(
            {
                "status": "success",
                "message": "Kendaraan berhasil dicatat masuk",
                "data": record,
            }
        )
    except Exception as e:
        logger.error(f"Error in record_entry: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@parking_bp.route("/exit", methods=["POST"])
def record_exit():
    """Mencatat kendaraan keluar"""
    try:
        data = request.get_json()
        if not data or "license_plate" not in data:
            return (
                jsonify(
                    {"status": "error", "message": "Data plat nomor tidak ditemukan"}
                ),
                400,
            )

        license_plate = data["license_plate"].strip().upper()
        if not validate_license_plate(license_plate):
            return (
                jsonify(
                    {"status": "error", "message": "Format plat nomor tidak valid"}
                ),
                400,
            )

        record = parking_service.add_parking_record(license_plate, "KELUAR")
        logger.info(f"Recorded exit for plate: {license_plate}")
        return jsonify(
            {
                "status": "success",
                "message": "Kendaraan berhasil dicatat keluar",
                "data": record,
            }
        )
    except Exception as e:
        logger.error(f"Error in record_exit: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@parking_bp.route("/stats", methods=["GET"])
def get_stats():
    """Mengambil statistik parkir"""
    try:
        stats = parking_service.get_parking_stats()
        return jsonify({"status": "success", "data": stats})
    except Exception as e:
        logger.error(f"Error in get_stats: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
