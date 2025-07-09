"""Entry point for python -m ccproxy.cli"""

from .app import app, main


if __name__ == "__main__":
    import sys

    sys.exit(app())
