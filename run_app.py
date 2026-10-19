"""
Unified launcher for the Multimodal RGB-Thermal Human Detection Studio.
Starts the FastAPI server serving both API endpoints and the React frontend.
"""

import os
import sys
import webbrowser
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure venv Python is used if available
venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
if venv_python.exists() and sys.executable.lower() != str(venv_python).lower():
    import subprocess
    print(f"Switching to virtual environment: {venv_python}")
    sys.exit(subprocess.call([str(venv_python), __file__] + sys.argv[1:]))

import uvicorn

def main():
    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}"

    print("\n" + "=" * 70)
    print("  Multimodal RGB-Thermal Human Detection Studio (React + YOLOv8n)")
    print("=" * 70)
    print(f"  * Web Dashboard:  {url}")
    print(f"  * API Docs:       {url}/docs")
    print(f"  * Project Root:   {PROJECT_ROOT}")
    print("=" * 70)
    print("  Press Ctrl+C to stop the server.\n")

    # Open browser automatically if not in headless environment
    try:
        webbrowser.open(url)
    except Exception:
        pass

    uvicorn.run("backend.app:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
