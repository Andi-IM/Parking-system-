# Import library yang diperlukan
from flask import Blueprint, jsonify, request

from app.services.student_service import student_service
from app.utils.validators import (validate_license_plate, validate_nim,
                                  validate_phone)

# Membuat blueprint untuk routing mahasiswa
student_bp = Blueprint("student", __name__)


@student_bp.route("/list", methods=["GET"])
def get_students():
    """Mengambil daftar semua mahasiswa yang terdaftar"""
    try:
        # Mengambil data semua mahasiswa dari database
        students = student_service.get_all_students()
        return jsonify({"status": "success", "data": students, "count": len(students)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@student_bp.route("/<nim>", methods=["GET"])
def get_student(nim):
    """Mengambil detail mahasiswa berdasarkan NIM"""
    try:
        # Mencari data mahasiswa
        student = student_service.get_student_by_nim(nim)
        if not student:
            return (
                jsonify({"status": "error", "message": "Mahasiswa tidak ditemukan"}),
                404,
            )

        return jsonify({"status": "success", "data": student})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@student_bp.route("/plate/<license_plate>", methods=["GET"])
def get_student_by_plate(license_plate):
    """Mencari mahasiswa berdasarkan plat nomor kendaraan"""
    try:
        # Validasi format plat nomor
        if not validate_license_plate(license_plate):
            return (
                jsonify(
                    {"status": "error", "message": "Format plat nomor tidak valid"}
                ),
                400,
            )

        # Mencari data mahasiswa berdasarkan plat nomor
        student = student_service.get_student_by_license_plate(license_plate)
        if not student:
            return (
                jsonify({"status": "error", "message": "Mahasiswa tidak ditemukan"}),
                404,
            )

        return jsonify({"status": "success", "data": student})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@student_bp.route("/register", methods=["POST"])
def register_student():
    """Mendaftarkan mahasiswa baru"""
    try:
        # Mengambil data dari request JSON
        data = request.get_json()

        # Validasi field yang wajib diisi
        required_fields = ["nim", "name", "major", "phone", "license_plate"]
        for field in required_fields:
            if field not in data:
                return (
                    jsonify(
                        {"status": "error", "message": f"Field {field} harus diisi"}
                    ),
                    400,
                )

        # Validasi format data
        if not validate_nim(data["nim"]):
            return (
                jsonify({"status": "error", "message": "Format NIM tidak valid"}),
                400,
            )

        if not validate_phone(data["phone"]):
            return (
                jsonify(
                    {"status": "error", "message": "Format nomor telepon tidak valid"}
                ),
                400,
            )

        if not validate_license_plate(data["license_plate"]):
            return (
                jsonify(
                    {"status": "error", "message": "Format plat nomor tidak valid"}
                ),
                400,
            )

        # Menyimpan data mahasiswa baru ke database
        student = student_service.create_student(data)
        return (
            jsonify(
                {
                    "status": "success",
                    "message": "Mahasiswa berhasil didaftarkan",
                    "data": student,
                }
            ),
            201,
        )

    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@student_bp.route("/<nim>", methods=["PUT"])
def update_student(nim):
    """Mengupdate data mahasiswa yang sudah ada"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "Data tidak ditemukan"}), 400

        # Validasi format untuk field yang diupdate
        if "phone" in data and not validate_phone(data["phone"]):
            return (
                jsonify(
                    {"status": "error", "message": "Format nomor telepon tidak valid"}
                ),
                400,
            )

        if "license_plate" in data and not validate_license_plate(
            data["license_plate"]
        ):
            return (
                jsonify(
                    {"status": "error", "message": "Format plat nomor tidak valid"}
                ),
                400,
            )

        # Update data mahasiswa di database
        student = student_service.update_student(nim, data)
        return jsonify(
            {
                "status": "success",
                "message": "Data mahasiswa berhasil diperbarui",
                "data": student,
            }
        )

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@student_bp.route("/<nim>", methods=["DELETE"])
def delete_student(nim):
    """Menghapus data mahasiswa"""
    try:
        student_service.delete_student(nim)
        return jsonify({"status": "success", "message": "Mahasiswa berhasil dihapus"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
