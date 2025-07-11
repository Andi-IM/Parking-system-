import logging
from datetime import datetime, timedelta

from flask import session

from app.models.admin import Admin

logger = logging.getLogger(__name__)


class AuthService:
    @staticmethod
    def login(username, password):
        logger.debug(f"Attempting login for username: {username}")
        admin = Admin.query.filter_by(username=username).first()

        if not admin or not admin.check_password(password):
            logger.warning(f"Login failed for username: {username}")
            return False, "Username atau password salah"

        if not admin.is_active:
            logger.warning(f"Inactive account attempt: {username}")
            return False, "Akun admin tidak aktif"

        # Set session
        session["admin_id"] = admin.id
        session["admin_name"] = admin.name
        session["admin_username"] = admin.username
        session["last_activity"] = datetime.now().isoformat()

        logger.info(f"Successful login for admin: {admin.username}")
        return True, admin.to_dict()

    @staticmethod
    def logout():
        logger.info(f"Logout for admin: {session.get('admin_username')}")
        session.clear()
        return True, "Logout berhasil"

    @staticmethod
    def get_current_admin():
        if "admin_id" not in session:
            logger.debug("No admin_id in session")
            return None

        # Check session timeout (30 menit)
        last_activity = datetime.fromisoformat(
            session.get("last_activity", "2000-01-01T00:00:00")
        )
        if datetime.now() - last_activity > timedelta(minutes=30):
            logger.info(f"Session timeout for admin: {session.get('admin_username')}")
            session.clear()
            return None

        # Update last activity
        session["last_activity"] = datetime.now().isoformat()

        admin = Admin.query.get(session["admin_id"])
        if not admin:
            logger.warning(f"Admin not found for id: {session['admin_id']}")
            session.clear()
            return None

        return admin

    @staticmethod
    def is_authenticated():
        is_auth = AuthService.get_current_admin() is not None
        logger.debug(f"Authentication check: {is_auth}")
        return is_auth

    @staticmethod
    def require_auth():
        if not AuthService.is_authenticated():
            logger.warning("Authentication required but not authenticated")
            return False, "Silakan login terlebih dahulu"
        return True, None
