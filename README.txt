

1. INSTALL DEPENDENCIES
   pip install -r requirements.txt

2. JALANKAN APLIKASI
   python app.py
   Aplikasi akan berjalan di http://127.0.0.1:5000

3. LOGIN ADMIN
   Username: admin
   Password: admin123

4. CARA MENUTUP
   Tekan Ctrl + C di terminal

5. TROUBLESHOOTING
   - Jika port 5000 sudah digunakan:
     taskkill /F /IM python.exe
   - Jika database error:
     Hapus instance/parking.db dan jalankan ulang

6. FOLDER PENTING
   - app/         : Kode aplikasi
   - instance/    : Database
   - templates/   : File HTML
   - static/      : CSS, JS, gambar
   - flask_session: Session login

7. KEAMANAN
   - Ganti password admin default
   - Backup database secara berkala
   - Jangan share file parking.db

8. TIPS
   - Selalu tutup dengan Ctrl+C
   - Jangan hapus folder instance dan flask_session
   - Gunakan browser Chrome/Firefox 