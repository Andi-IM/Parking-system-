import logging
from datetime import datetime

from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.extensions import db
from app.models.models import ParkingRecord, Student


class StudentService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def get_all_students():
        """Get all students"""
        try:
            students = Student.query.all()
            return [student.to_dict() for student in students]
        except Exception as e:
            raise Exception(f"Error getting students: {str(e)}")

    @staticmethod
    def get_student_by_nim(nim):
        """Get student by NIM"""
        try:
            student = Student.query.filter_by(nim=nim).first()
            if not student:
                raise Exception("Mahasiswa tidak ditemukan")
            return student.to_dict()
        except Exception as e:
            raise Exception(f"Error getting student: {str(e)}")

    @staticmethod
    def get_student_by_license_plate(license_plate):
        """Get student by license plate"""
        try:
            student = Student.query.filter_by(license_plate=license_plate).first()
            return student.to_dict() if student else None
        except Exception as e:
            raise Exception(f"Error getting student: {str(e)}")

    @staticmethod
    def add_student(nim, name, major, phone, license_plate):
        """Add new student"""
        try:
            # Validasi NIM unik
            if Student.query.filter_by(nim=nim).first():
                raise Exception("NIM sudah terdaftar")

            # Validasi plat nomor unik
            if Student.query.filter_by(license_plate=license_plate).first():
                raise Exception("Plat nomor sudah terdaftar")

            student = Student(
                nim=nim,
                name=name,
                major=major,
                phone=phone,
                license_plate=license_plate,
            )
            db.session.add(student)
            db.session.commit()
            return student.to_dict()
        except Exception as e:
            db.session.rollback()
            raise Exception(f"Error adding student: {str(e)}")

    @staticmethod
    def update_student(nim, data):
        """Update student data"""
        try:
            student = Student.query.filter_by(nim=nim).first()
            if not student:
                raise Exception("Mahasiswa tidak ditemukan")

            # Update fields jika ada
            if "name" in data:
                student.name = data["name"]
            if "major" in data:
                student.major = data["major"]
            if "phone" in data:
                student.phone = data["phone"]
            if "license_plate" in data:
                # Cek plat nomor unik
                existing = Student.query.filter_by(
                    license_plate=data["license_plate"]
                ).first()
                if existing and existing.nim != nim:
                    raise Exception("Plat nomor sudah terdaftar")
                student.license_plate = data["license_plate"]

            student.updated_at = datetime.utcnow()
            db.session.commit()
            return student.to_dict()
        except Exception as e:
            db.session.rollback()
            raise Exception(f"Error updating student: {str(e)}")

    @staticmethod
    def delete_student(nim):
        """Delete student"""
        try:
            student = Student.query.filter_by(nim=nim).first()
            if not student:
                raise Exception("Mahasiswa tidak ditemukan")

            # Cek apakah mahasiswa memiliki record parkir aktif
            active_parking = ParkingRecord.query.filter_by(
                student_nim=nim, status="MASUK"
            ).first()
            if active_parking:
                raise Exception(
                    "Tidak dapat menghapus mahasiswa yang masih memiliki kendaraan dalam area parkir"
                )

            db.session.delete(student)
            db.session.commit()
            return True
        except Exception as e:
            db.session.rollback()
            raise Exception(f"Error deleting student: {str(e)}")

    @staticmethod
    def search_students(query=None, major=None):
        """Search students with filters"""
        try:
            students = Student.query

            if query:
                query = f"%{query}%"
                students = students.filter(
                    (Student.name.ilike(query))
                    | (Student.nim.ilike(query))
                    | (Student.license_plate.ilike(query))
                )

            if major:
                students = students.filter_by(major=major)

            return [student.to_dict() for student in students.all()]
        except Exception as e:
            raise Exception(f"Error searching students: {str(e)}")

    def get_students_by_major(self, major):
        session = db.get_session()
        try:
            students = session.query(Student).filter_by(major=major).all()
            return [student.to_dict() for student in students]
        finally:
            session.close()


# Create a singleton instance
student_service = StudentService()
