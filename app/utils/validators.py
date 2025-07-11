# Import modul regex untuk validasi format
import re


def validate_nim(nim):
    """
    Memvalidasi format NIM mahasiswa
    - Harus terdiri dari 10 digit angka
    - Contoh valid: 2111520001
    - Contoh tidak valid: 123, ABC123, 12345678901
    """
    pattern = r"^\d{10}$"
    return bool(re.match(pattern, str(nim)))


def validate_phone(phone):
    """
    Memvalidasi format nomor telepon Indonesia
    - Harus diawali dengan: +62, 62, atau 0
    - Diikuti angka 8
    - Diikuti 1-9 (operator seluler)
    - Total panjang 10-13 digit
    - Contoh valid: 081234567890, +6281234567890
    - Contoh tidak valid: 12345, 0812345, +65812345678
    """
    pattern = r"^(\+62|62|0)[8][1-9][0-9]{6,10}$"
    return bool(re.match(pattern, str(phone)))


def validate_license_plate(plate):
    """
    Memvalidasi format plat nomor kendaraan Indonesia
    - Format: [1-2 huruf] [1-4 angka] [1-3 huruf]
    - Huruf harus kapital
    - Dipisahkan dengan spasi
    - Contoh valid: BA 1234 XX, B 123 ABC
    - Contoh tidak valid: BAA 123 X, B1234XX, ba 123 xx
    """
    pattern = r"^[A-Z]{1,2}\s\d{1,4}\s[A-Z]{1,3}$"
    return bool(re.match(pattern, str(plate)))
