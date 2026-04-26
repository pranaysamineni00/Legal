"""Kill any existing server on port 5001 and launch the LexScan dashboard."""
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


PORT = 5001
URL = f"http://127.0.0.1:{PORT}"


def kill_port(port: int) -> None:
    result = subprocess.run(
        ["lsof", "-ti", f"tcp:{port}"],
        capture_output=True, text=True
    )
    pids = result.stdout.strip().split()
    for pid in filter(None, pids):
        subprocess.run(["kill", "-9", pid], check=False)
        print(f"Killed process {pid} on port {port}.")


def wait_for_server(port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.25)
    return False


def main() -> None:
    kill_port(PORT)
    time.sleep(0.5)

    server = subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=str(Path(__file__).parent),
    )

    print(f"Server starting (PID {server.pid})…")
    if not wait_for_server(PORT):
        print("Server did not become ready in time.")
        server.terminate()
        sys.exit(1)

    webbrowser.open(URL)
    print(f"Opened {URL}")

    try:
        server.wait()
    except KeyboardInterrupt:
        server.terminate()
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
