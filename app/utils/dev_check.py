import importlib.util
import os
import warnings
from typing import List


def check_development_dependencies() -> List[str]:
    """Check if development dependencies are installed.

    Returns:
        List of development packages found in production
    """
    dev_packages = [
        "black",
        "flake8",
        "pytest",
        "mypy",
        "bandit",
        "jupyter",
        "ipython",
        "sphinx",
        "pre_commit",
    ]

    found_dev_packages = []

    for package in dev_packages:
        if importlib.util.find_spec(package) is not None:
            found_dev_packages.append(package)

    return found_dev_packages


def warn_if_dev_dependencies_in_production():
    """Warn if development dependencies are found in production."""
    if os.getenv("ENVIRONMENT") == "production":
        dev_deps = check_development_dependencies()

        if dev_deps:
            warnings.warn(
                f"Development dependencies found in production: {', '.join(dev_deps)}. "
                "This may indicate a misconfigured environment.",
                UserWarning,
                stacklevel=2,
            )

            # Log to application logger if available
            try:
                from flask import current_app

                current_app.logger.warning(
                    f"Development dependencies detected: {dev_deps}"
                )
            except (ImportError, RuntimeError):
                pass


# Auto-check when module is imported
warn_if_dev_dependencies_in_production()
