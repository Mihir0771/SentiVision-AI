"""
Entry point for the Sentiment Analyzer application.

Loads environment variables from ``.env``, creates the Flask app via the
application factory, and starts the development server through
Flask-SocketIO.

Usage::

    # Standard run (open http://localhost:5000)
    python run.py

    # With HTTPS / SSL support for remote / LAN camera access
    python run.py --ssl

    # Custom port / host
    python run.py --port 8080 --host 127.0.0.1 --debug
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

# Load .env before anything reads os.environ
load_dotenv()

from app import create_app, socketio  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse optional command-line arguments for host, port, debug, and SSL."""
    parser = argparse.ArgumentParser(
        description="Run the SentiVision AI Sentiment Analyzer server."
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("HOST", "127.0.0.1"),
        help="Hostname to bind to (default: 127.0.0.1 for instant camera security)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", 5000)),
        help="Port to listen on (default: 5000)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=os.environ.get("FLASK_DEBUG", "0") == "1",
        help="Enable debug / auto-reload mode",
    )
    parser.add_argument(
        "--ssl",
        action="store_true",
        default=os.environ.get("FLASK_SSL", "0") == "1",
        help="Enable ad-hoc HTTPS/SSL for secure camera access across network",
    )
    return parser.parse_args()


def print_banner(host: str, port: int, use_ssl: bool) -> None:
    """Display an informative, polished terminal startup banner (ASCII-safe)."""
    scheme = "https" if use_ssl else "http"
    local_url = f"{scheme}://localhost:{port}"
    loopback_url = f"{scheme}://127.0.0.1:{port}"

    print("\n" + "=" * 68)
    print(" [SentiVision AI] - Open-Source Sentiment & Facial Emotion Platform")
    print("=" * 68)
    print(f" -> Web UI:      {local_url}")
    print(f" -> Loopback:    {loopback_url}")
    if host == "0.0.0.0":
        print(f" -> Network:     {scheme}://0.0.0.0:{port}")
    print("-" * 68)
    print(" WEBCAM & SECURITY NOTICE:")
    print("   Modern browsers (Chrome, Edge, Safari) require a Secure Context")
    print("   for webcam access. 'localhost' is automatically trusted!")
    print(f"   PLEASE OPEN: {local_url} in your browser.")
    if not use_ssl:
        print("   If testing over local IP on another device, restart with:")
        print("   python run.py --ssl --host 0.0.0.0")
    print("=" * 68 + "\n")
    sys.stdout.flush()


def main() -> None:
    """Create the application and start the SocketIO server."""
    args = parse_args()
    config_name: str = os.environ.get("FLASK_ENV", "development")
    app = create_app(config_name)

    ssl_context = None
    if args.ssl:
        try:
            ssl_context = "adhoc"
            app.logger.info("SSL enabled using adhoc certificate.")
        except Exception as exc:
            app.logger.warning("Could not enable adhoc SSL (requires cryptography): %s", exc)

    print_banner(args.host, args.port, bool(ssl_context))

    socketio_kwargs = {
        "host": args.host,
        "port": args.port,
        "debug": args.debug,
        "allow_unsafe_werkzeug": True,
    }
    if ssl_context:
        socketio_kwargs["ssl_context"] = ssl_context

    socketio.run(app, **socketio_kwargs)


if __name__ == "__main__":
    main()
