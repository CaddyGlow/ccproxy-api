"""Entry point for python -m ccproxy.cli"""

from .main import app


if __name__ == "__main__":
    import sys

    sys.exit(app())
