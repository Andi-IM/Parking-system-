import cv2
import numpy as np
import pandas as pd
from flask import Blueprint, Response, render_template
from ultralytics import YOLO

from tracker import Tracker  # tracker.py ada di root project

webcam_bp = Blueprint("webcam", __name__)

model = YOLO("models/best.pt")
area_masuk = [(720, 420), (690, 390), (410, 500), (440, 530)]
area_keluar = [(850, 420), (820, 390), (540, 500), (570, 530)]

cap = cv2.VideoCapture(0)  # webcam
tracker = Tracker()

motor_masuk = set()
motor_keluar = set()
masuk_dict = {}
keluar_dict = {}


def gen_frames():
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 2 != 0:
            continue
        frame = cv2.resize(frame, (1280, 720))
        results = model.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a.cpu().numpy()).astype("float")
        dets = []
        for _, row in px.iterrows():
            x1, y1, x2, y2, score, cls_id = row
            if int(cls_id) == 0 and score > 0.5:
                dets.append([int(x1), int(y1), int(x2), int(y2)])
        bbox_id = tracker.update(dets)
        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            cx, cy = int((x3 + x4) / 2), int((y3 + y4) / 2)
            in_masuk = cv2.pointPolygonTest(
                np.array(area_masuk, np.int32), (cx, cy), False
            )
            if in_masuk >= 0:
                masuk_dict[id] = (cx, cy)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
            if id in masuk_dict:
                in_keluar = cv2.pointPolygonTest(
                    np.array(area_keluar, np.int32), (cx, cy), False
                )
                if in_keluar >= 0:
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)
                    cv2.putText(
                        frame,
                        str(id),
                        (x3, y3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    motor_masuk.add(id)
            in_keluar2 = cv2.pointPolygonTest(
                np.array(area_keluar, np.int32), (cx, cy), False
            )
            if in_keluar2 >= 0:
                keluar_dict[id] = (cx, cy)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
            if id in keluar_dict:
                in_masuk2 = cv2.pointPolygonTest(
                    np.array(area_masuk, np.int32), (cx, cy), False
                )
                if in_masuk2 >= 0:
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
                    cv2.putText(
                        frame,
                        str(id),
                        (x3, y3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    motor_keluar.add(id)
        # Gambar polygon area
        cv2.polylines(frame, [np.array(area_masuk, np.int32)], True, (0, 255, 0), 2)
        cv2.putText(
            frame, "Masuk", area_masuk[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        cv2.polylines(frame, [np.array(area_keluar, np.int32)], True, (0, 0, 255), 2)
        cv2.putText(
            frame,
            "Keluar",
            area_keluar[0],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        # Tampilkan count masuk/keluar
        cv2.putText(
            frame,
            f"Masuk: {len(motor_masuk)}",
            (60, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Keluar: {len(motor_keluar)}",
            (60, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        # Encode ke JPEG dan yield
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@webcam_bp.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@webcam_bp.route("/webcam")
def webcam():
    return render_template("webcam.html")
