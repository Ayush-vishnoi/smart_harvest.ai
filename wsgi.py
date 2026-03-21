import sys
import os

# ── Set your PythonAnywhere username here ──────────────────────────────────────
USERNAME = 'your_pythonanywhere_username'  # ← CHANGE THIS
PROJECT_DIR = f'/home/{USERNAME}/smart_harvest_ai'
# ──────────────────────────────────────────────────────────────────────────────

# Add project root and backend to path
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'backend'))

# Change working directory so relative paths inside app work
os.chdir(PROJECT_DIR)

# Import the Flask app
from backend.app_v2 import app as application
