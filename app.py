# File utama aplikasi Flask
from app import create_app

# Membuat instance aplikasi dengan konfigurasi development
app = create_app()

from app.api.webcam_routes import webcam_bp

app.register_blueprint(webcam_bp)

if __name__ == "__main__":
    app.run(debug=True)

# entry point
