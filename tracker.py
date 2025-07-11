import math  # Untuk menghitung jarak Euclidean antara titik (hipotenusa)


# Class untuk tracking objek berdasarkan posisi pusat (center point)
class Tracker:
    def __init__(self):
        # Dictionary untuk menyimpan posisi tengah (center point) dari setiap objek yang sedang dilacak
        self.center_points = {}
        # Counter untuk membuat ID unik untuk setiap objek yang baru terdeteksi
        self.id_count = 0

    def update(self, objects_rect):
        # List untuk menyimpan hasil deteksi: [x1, y1, x2, y2, id]
        objects_bbs_ids = []

        # Loop untuk setiap objek yang baru terdeteksi pada frame sekarang
        for rect in objects_rect:
            x1, y1, x2, y2 = rect
            # Hitung titik tengah dari bounding box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Flag untuk mengecek apakah objek yang sama sudah pernah terdeteksi sebelumnya
            same_object_detected = False

            # Bandingkan titik tengah objek sekarang dengan semua titik tengah yang tersimpan
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])  # Hitung jarak Euclidean
                # Jika jarak < 35 piksel, anggap objek yang sama (toleransi pergerakan objek)
                if dist < 35:
                    self.center_points[id] = (cx, cy)  # Update titik tengah objek
                    objects_bbs_ids.append(
                        [x1, y1, x2, y2, id]
                    )  # Tambahkan ID lama ke list hasil
                    same_object_detected = True
                    break  # Stop pencarian, karena objek sudah dikenali

            # Jika tidak ditemukan objek yang sama, berarti objek baru
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)  # Simpan titik tengah baru
                objects_bbs_ids.append(
                    [x1, y1, x2, y2, self.id_count]
                )  # Tambahkan dengan ID baru
                self.id_count += 1  # Naikkan counter ID

        # Membersihkan objek lama yang tidak muncul lagi
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]  # Ambil titik tengah dari ID
            new_center_points[object_id] = center  # Simpan ke dictionary baru

        # Update center_points dengan objek yang masih aktif
        self.center_points = new_center_points.copy()

        # Kembalikan hasil akhir: bounding box + ID
        return objects_bbs_ids
