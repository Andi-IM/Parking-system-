# File inisialisasi aplikasi Flask
# Berisi konfigurasi dasar dan registrasi blueprint

import logging
import os
from datetime import timedelta

from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_cors import CORS
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy

from app.api.parking_routes import parking_bp
from app.api.student_routes import student_bp
from app.extensions import db
from app.models.admin import Admin
from app.models.models import ParkingRecord, Student
from app.routes.auth_routes import auth_bp
from app.services.parking_service import parking_service
from app.services.student_service import student_service
from app.utils.dev_check import warn_if_dev_dependencies_in_production
from flask_session import Session

# Inisialisasi SocketIO
socketio = SocketIO()


def create_app(config_name="default"):
    """
    Fungsi untuk membuat dan mengkonfigurasi aplikasi Flask

    Args:
        config_name (str): Nama konfigurasi yang akan digunakan ('development', 'production', dll)

    Returns:
        Flask: Instance aplikasi Flask yang sudah dikonfigurasi
    """
    # Inisialisasi aplikasi Flask dengan konfigurasi folder static dan template
    app = Flask(__name__)

    # Check for dev dependencies in production
    warn_if_dev_dependencies_in_production()

    # Set folder template dan static dari root project
    app.template_folder = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "templates"
    )
    app.static_folder = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "static"
    )

    # Configure logging
    logging.basicConfig(level=logging.DEBUG)

    # Enable debug mode and template auto-reload for development
    app.config["DEBUG"] = True
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.auto_reload = True

    # Mengaktifkan CORS untuk akses API dari domain berbeda
    CORS(app)

    # Konfigurasi dasar
    app.config["SECRET_KEY"] = "dev-key-123"

    # Konfigurasi database SQLite
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///parking.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # Konfigurasi session
    app.config["SESSION_TYPE"] = "filesystem"
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=30)
    app.config["SESSION_COOKIE_SECURE"] = (
        False  # Set ke True di production dengan HTTPS
    )
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["SESSION_FILE_DIR"] = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "flask_session"
    )

    # Inisialisasi ekstensi
    db.init_app(app)
    Session(app)
    socketio.init_app(app, cors_allowed_origins="*", async_mode="threading")

    # Buat database dan tabel
    with app.app_context():
        db.create_all()

        # Buat admin default jika belum ada
        if not Admin.query.filter_by(username="admin").first():
            admin = Admin(
                username="admin",
                name="Administrator",
                email="admin@example.com",
                is_active=True,
            )
            admin.set_password("admin123")
            db.session.add(admin)
            db.session.commit()
            app.logger.info("Created default admin user")

        # Tambah data contoh jika belum ada
        if not Student.query.first():
            # Tambah beberapa mahasiswa
            students = [
                Student(
                    nim="2021001",
                    name="John Doe",
                    major="Teknik Informatika",
                    phone="081234567890",
                    license_plate="BA 1234 AB",
                ),
                Student(
                    nim="2021002",
                    name="Jane Smith",
                    major="Sistem Informasi",
                    phone="081234567891",
                    license_plate="BA 5678 CD",
                ),
                Student(
                    nim="2021003",
                    name="Bob Johnson",
                    major="Teknik Informatika",
                    phone="081234567892",
                    license_plate="BA 9012 EF",
                ),
                Student(
                    nim="2021004",
                    name="Alice Brown",
                    major="Sistem Informasi",
                    phone="081234567893",
                    license_plate="BA 3456 GH",
                ),
            ]
            db.session.add_all(students)
            db.session.commit()
            app.logger.info("Added sample students")

            # Tambah beberapa record parkir
            parking_records = [
                ParkingRecord(student_nim="2021001", status="MASUK"),
                ParkingRecord(student_nim="2021002", status="MASUK"),
                ParkingRecord(student_nim="2021003", status="MASUK"),
                ParkingRecord(student_nim="2021004", status="KELUAR"),
            ]
            db.session.add_all(parking_records)
            db.session.commit()
            app.logger.info("Added sample parking records")

    # Registrasi blueprint untuk routing API
    app.register_blueprint(student_bp, url_prefix="/api/mahasiswa")
    app.register_blueprint(parking_bp, url_prefix="/api/parking")
    app.register_blueprint(auth_bp, url_prefix="/auth")

    # Registrasi route untuk halaman web
    @app.route("/")
    def index():
        """Halaman utama"""
        try:
            # Mengambil data parkir aktif
            active_parking = parking_service.get_active_parking()
            return render_template("index.html", parking_data=active_parking)
        except Exception as e:
            app.logger.error(f"Error in index route: {str(e)}")
            return render_template("index.html", error=str(e))

    @app.route("/admin")
    def admin():
        """Halaman admin"""
        try:
            # Check authentication
            from app.services.auth_service import AuthService

            auth_result, auth_message = AuthService.require_auth()
            if not auth_result:
                return redirect(url_for("auth.login"))

            # Mengambil ringkasan data untuk admin
            students = student_service.get_all_students()
            app.logger.info(
                f"Admin page: Retrieved {len(students) if students else 0} students"
            )

            active_parking = parking_service.get_active_parking()
            app.logger.info(
                f"Admin page: Retrieved {len(active_parking) if active_parking else 0} active parking records"
            )

            parking_history = parking_service.get_parking_history(limit=10)
            app.logger.info(
                f"Admin page: Retrieved {len(parking_history) if parking_history else 0} parking history records"
            )

            return render_template(
                "admin.html",
                students=students,
                active_parking=active_parking,
                parking_history=parking_history,
            )
        except Exception as e:
            app.logger.error(f"Error in admin route: {str(e)}", exc_info=True)
            return render_template("admin.html", error=str(e))

    @app.route("/mahasiswa")
    def mahasiswa():
        """Halaman daftar mahasiswa"""
        try:
            # Ambil filter dari query string
            selected_major = request.args.get("major", None)
            q = request.args.get("q", "").strip()

            # Ambil semua jurusan unik dan data mahasiswa dalam satu session
            with db.session.begin():
                majors = [
                    m[0]
                    for m in Student.query.with_entities(Student.major).distinct().all()
                ]

                # Query dasar
                query = Student.query
                if selected_major:
                    query = query.filter(Student.major == selected_major)
                if q:
                    q_like = f"%{q}%"
                    query = query.filter(
                        (Student.name.ilike(q_like))
                        | (Student.nim.ilike(q_like))
                        | (Student.license_plate.ilike(q_like))
                    )
                students = [s.to_dict() for s in query.all()]

            app.logger.info(
                f"Retrieved {len(students) if students else 0} students from database"
            )
            return render_template(
                "mahasiswa.html",
                students=students,
                majors=majors,
                selected_major=selected_major,
            )
        except Exception as e:
            app.logger.error(f"Error in mahasiswa route: {str(e)}", exc_info=True)
            return render_template("mahasiswa.html", error=str(e))

    @app.route("/history")
    def history():
        """Halaman riwayat parkir"""
        try:
            # Mengambil riwayat parkir
            parking_history = parking_service.get_parking_history()
            app.logger.info(
                f"History page: Retrieved {len(parking_history) if parking_history else 0} parking records"
            )
            return render_template("history.html", parking_history=parking_history)
        except Exception as e:
            app.logger.error(f"Error in history route: {str(e)}", exc_info=True)
            return render_template("history.html", error=str(e))

    @app.route("/register", methods=["GET", "POST"])
    def register():
        """Halaman pendaftaran mahasiswa baru dan proses tambah mahasiswa"""
        if request.method == "GET":
            try:
                return render_template("register.html")
            except Exception as e:
                app.logger.error(f"Error in register route: {str(e)}", exc_info=True)
                return render_template("register.html", error=str(e))
        else:
            # Handle POST (AJAX)
            try:
                name = request.form.get("name", "").strip()
                nim = request.form.get("nim", "").strip()
                major = request.form.get("major", "").strip()
                phone = request.form.get("phone", "").strip()
                license_plate = request.form.get("license_plate", "").strip().upper()

                # Validasi sederhana
                if not all([name, nim, major, phone, license_plate]):
                    return jsonify(success=False, message="Semua field harus diisi.")
                if not nim.isdigit() or not (8 <= len(nim) <= 15):
                    return jsonify(success=False, message="NIM harus angka 8-15 digit.")
                if not phone.isdigit() or not (10 <= len(phone) <= 13):
                    return jsonify(
                        success=False, message="Nomor HP harus angka 10-13 digit."
                    )

                # Validasi plat nomor sederhana (bisa disesuaikan)
                import re

                if not re.match(r"^[A-Z]{1,2} ?[0-9]{1,4} ?[A-Z]{1,3}$", license_plate):
                    return jsonify(
                        success=False,
                        message="Format plat nomor tidak valid. Contoh: BA 1234 XX",
                    )

                # Cek duplikasi NIM/plat dan simpan ke database dalam satu session
                with db.session.begin():
                    if Student.query.filter_by(nim=nim).first():
                        return jsonify(success=False, message="NIM sudah terdaftar.")
                    if Student.query.filter_by(license_plate=license_plate).first():
                        return jsonify(
                            success=False, message="Plat nomor sudah terdaftar."
                        )

                    # Simpan ke database
                    new_student = Student(
                        nim=nim,
                        name=name,
                        major=major,
                        phone=phone,
                        license_plate=license_plate,
                    )
                    db.session.add(new_student)
                    db.session.commit()

                return jsonify(success=True, message="Pendaftaran berhasil!")
            except Exception as e:
                app.logger.error(f"Error in register POST: {str(e)}", exc_info=True)
                return jsonify(success=False, message="Terjadi kesalahan pada server.")

    # Handler untuk error 404 (Page Not Found)
    @app.errorhandler(404)
    def not_found(e):
        return render_template("404.html"), 404

    # Handler untuk error 500 (Internal Server Error)
    @app.errorhandler(500)
    def server_error(e):
        app.logger.error(f"Server error: {str(e)}")
        return render_template("500.html"), 500

    return app
