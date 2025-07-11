from flask import Blueprint, jsonify, render_template, request

from app.services.parking_service import parking_service

parking_bp = Blueprint("parking", __name__)


@parking_bp.route("/status", methods=["GET"])
def get_parking_status():
    """Get current parking status"""
    try:
        active_parking = parking_service.get_active_parking()
        return jsonify({"success": True, "data": active_parking})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@parking_bp.route("/history", methods=["GET"])
def get_parking_history():
    """Get parking history"""
    try:
        history = parking_service.get_parking_history()
        return jsonify({"success": True, "data": history})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
