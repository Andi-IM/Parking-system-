from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.extensions import db


# ========== Model Student ==========
class Student(db.Model):
    __tablename__ = "students"

    nim = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    major = Column(String, nullable=False)
    phone = Column(String)
    license_plate = Column(String, nullable=False, unique=True)

    # Relationship
    parking_records = db.relationship(
        "ParkingRecord", backref="student_ref", lazy=True, cascade="all, delete-orphan"
    )
    parking_logs = relationship("ParkingLog", back_populates="student")

    def __init__(self, nim, name, major, phone, license_plate):
        self.nim = nim
        self.name = name
        self.major = major
        self.phone = phone
        self.license_plate = license_plate

    def to_dict(self):
        """Convert model instance to dictionary"""
        return {
            "nim": self.nim,
            "name": self.name,
            "major": self.major,
            "phone": self.phone,
            "license_plate": self.license_plate,
        }


# ========== Model ParkingLog ==========
class ParkingLog(db.Model):
    __tablename__ = "parking_logs"

    id = db.Column(db.Integer, primary_key=True)
    student_nim = db.Column(db.String, ForeignKey("students.nim"))
    entry_time = db.Column(db.DateTime, default=datetime.now)
    exit_time = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String, default="active")  # active or completed
    student = relationship("Student", back_populates="parking_logs")


# ========== Model ParkingSlot ==========
class ParkingSlot(db.Model):
    __tablename__ = "parking_slots"

    id = db.Column(db.Integer, primary_key=True)
    slot_number = db.Column(db.String(10), nullable=False)
    is_occupied = db.Column(db.Boolean, default=False)
    current_record_id = db.Column(
        db.Integer, db.ForeignKey("parking_records.id"), nullable=True
    )
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationship
    current_record = db.relationship(
        "ParkingRecord", foreign_keys=[current_record_id], post_update=True
    )

    def __init__(self, slot_number):
        self.slot_number = slot_number
        self.is_occupied = False

    def to_dict(self):
        """Convert model instance to dictionary"""
        return {
            "id": self.id,
            "slot_number": self.slot_number,
            "is_occupied": self.is_occupied,
            "current_record_id": self.current_record_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ========== Model ParkingRecord ==========
class ParkingRecord(db.Model):
    __tablename__ = "parking_records"

    id = db.Column(db.Integer, primary_key=True)
    student_nim = db.Column(
        db.String, db.ForeignKey("students.nim", ondelete="CASCADE"), nullable=False
    )
    slot_id = db.Column(db.Integer, db.ForeignKey("parking_slots.id"), nullable=True)
    status = db.Column(db.String(10), nullable=False)  # 'MASUK' atau 'KELUAR'
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    slot = db.relationship(
        "ParkingSlot", foreign_keys=[slot_id], overlaps="current_record"
    )

    def __init__(self, student_nim, status, slot_id=None):
        self.student_nim = student_nim
        self.status = status
        self.slot_id = slot_id

    def to_dict(self):
        """Convert model instance to dictionary"""
        return {
            "id": self.id,
            "student_nim": self.student_nim,
            "slot_id": self.slot_id,
            "status": self.status,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "student": self.student_ref.to_dict() if self.student_ref else None,
            "slot": self.slot.to_dict() if self.slot else None,
        }
