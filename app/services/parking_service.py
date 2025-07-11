import logging
from datetime import datetime

from app.extensions import db
from app.models.models import ParkingRecord, Student


class ParkingService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_active_parking(self):
        """Get all active parking records"""
        try:
            records = (
                ParkingRecord.query.filter_by(status="MASUK")
                .join(Student)
                .order_by(ParkingRecord.timestamp.desc())
                .all()
            )
            self.logger.info(f"Found {len(records)} active parking records")
            return [record.to_dict() for record in records]
        except Exception as e:
            self.logger.error(f"Error getting active parking: {str(e)}")
            raise Exception(f"Error getting active parking: {str(e)}")

    def get_parking_history(self, limit=None, status=None):
        """Get parking history with optional limit and status filter"""
        try:
            query = ParkingRecord.query.join(Student)
            if status:
                query = query.filter_by(status=status)
            query = query.order_by(ParkingRecord.timestamp.desc())
            if limit:
                query = query.limit(limit)
            records = query.all()
            self.logger.info(f"Found {len(records)} parking history records")
            return [record.to_dict() for record in records]
        except Exception as e:
            self.logger.error(f"Error getting parking history: {str(e)}")
            raise Exception(f"Error getting parking history: {str(e)}")

    def add_parking_record(self, license_plate, status):
        """Add a new parking record"""
        try:
            # Find student by license plate
            student = Student.query.filter_by(license_plate=license_plate).first()
            if not student:
                raise Exception(f"Student with license plate {license_plate} not found")

            # Create new parking record
            record = ParkingRecord(
                student_nim=student.nim, status=status, timestamp=datetime.now()
            )

            db.session.add(record)
            db.session.commit()

            self.logger.info(
                f"Added new parking record for {license_plate} with status {status}"
            )
            return record.to_dict()
        except Exception as e:
            db.session.rollback()
            self.logger.error(f"Error adding parking record: {str(e)}")
            raise Exception(f"Error adding parking record: {str(e)}")

    def get_student_parking_history(self, student_nim, limit=None):
        """Get parking history for specific student"""
        try:
            query = ParkingRecord.query.filter_by(student_nim=student_nim).order_by(
                ParkingRecord.timestamp.desc()
            )
            if limit:
                query = query.limit(limit)
            records = query.all()
            return [record.to_dict() for record in records]
        except Exception as e:
            self.logger.error(f"Error getting student parking history: {str(e)}")
            raise Exception(f"Error getting student parking history: {str(e)}")

    def get_parking_stats(self):
        """Get parking statistics"""
        try:
            active_count = ParkingRecord.query.filter_by(status="MASUK").count()
            completed_count = ParkingRecord.query.filter_by(status="KELUAR").count()
            total_slots = 100  # Total parking slots
            available_slots = total_slots - active_count

            stats = {
                "active": active_count,
                "completed": completed_count,
                "total_slots": total_slots,
                "available_slots": available_slots,
            }

            self.logger.info(f"Retrieved parking stats: {stats}")
            return stats
        except Exception as e:
            self.logger.error(f"Error getting parking stats: {str(e)}")
            raise Exception(f"Error getting parking stats: {str(e)}")


parking_service = ParkingService()
