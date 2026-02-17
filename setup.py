#!/usr/bin/env python3
"""
Smart Harvest AI - Quick Setup Script
Run this to automatically setup your project
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{text:^60}{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")

def check_python_version():
    print_header("Checking Python Version")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print(f"Python version: {version_str}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error("Python 3.8 or higher is required!")
        return False
    
    print_success(f"Python {version_str} is compatible")
    return True

def check_dataset():
    print_header("Checking Dataset")
    dataset_path = Path("FINAL_DATASET_cleaned.csv")
    
    if not dataset_path.exists():
        print_error("FINAL_DATASET_cleaned.csv not found!")
        print_warning("Please place your dataset in the project root folder")
        return False
    
    size_mb = dataset_path.stat().st_size / (1024 * 1024)
    print_success(f"Dataset found ({size_mb:.1f} MB)")
    return True

def create_directories():
    print_header("Creating Directory Structure")
    
    directories = [
        "backend/models/yield_model",
        "backend/models/irrigation_model",
        "backend/ml",
        "frontend",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print_success(f"Created: {directory}")
    
    return True

def create_virtual_environment():
    print_header("Setting Up Virtual Environment")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print_warning("Virtual environment already exists")
        response = input("Recreate it? (y/n): ").lower()
        if response != 'y':
            print_info("Using existing virtual environment")
            return True
        
        import shutil
        shutil.rmtree(venv_path)
    
    print_info("Creating virtual environment...")
    result = subprocess.run([sys.executable, "-m", "venv", "venv"], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print_error("Failed to create virtual environment")
        print(result.stderr)
        return False
    
    print_success("Virtual environment created")
    return True

def install_dependencies():
    print_header("Installing Dependencies")
    
    # Detect OS and get pip path
    if platform.system() == "Windows":
        pip_path = "venv\\Scripts\\pip.exe"
    else:
        pip_path = "venv/bin/pip"
    
    if not Path(pip_path).exists():
        print_error(f"Pip not found at {pip_path}")
        return False
    
    # Upgrade pip
    print_info("Upgrading pip...")
    subprocess.run([pip_path, "install", "--upgrade", "pip"], 
                  capture_output=True)
    
    # Install requirements
    requirements_file = Path("backend/requirements.txt")
    
    if not requirements_file.exists():
        print_error("requirements.txt not found in backend folder!")
        print_warning("Please create backend/requirements.txt with required packages")
        return False
    
    print_info("Installing packages (this may take 5-10 minutes)...")
    result = subprocess.run([pip_path, "install", "-r", str(requirements_file)],
                          capture_output=False, text=True)
    
    if result.returncode != 0:
        print_error("Failed to install dependencies")
        return False
    
    print_success("All dependencies installed")
    return True

def train_models():
    print_header("Training Machine Learning Models")
    
    # Get python path
    if platform.system() == "Windows":
        python_path = "venv\\Scripts\\python.exe"
    else:
        python_path = "venv/bin/python"
    
    # Check if training scripts exist
    yield_script = Path("backend/ml/train_yield_model_v2.py")
    irrigation_script = Path("backend/ml/train_irrigation_model.py")
    
    if not yield_script.exists():
        print_error(f"Training script not found: {yield_script}")
        print_warning("Please place train_yield_model_v2.py in backend/ml/")
        return False
    
    if not irrigation_script.exists():
        print_error(f"Training script not found: {irrigation_script}")
        print_warning("Please place train_irrigation_model.py in backend/ml/")
        return False
    
    # Train yield model
    print_info("Training Yield Prediction Model (10-20 minutes)...")
    print_info("You'll see training progress below:\n")
    
    result = subprocess.run([python_path, str(yield_script)], cwd=".")
    
    if result.returncode != 0:
        print_error("Yield model training failed")
        return False
    
    print_success("Yield model trained")
    
    # Train irrigation model
    print_info("\nTraining Irrigation Model (5-10 minutes)...")
    
    result = subprocess.run([python_path, str(irrigation_script)], cwd=".")
    
    if result.returncode != 0:
        print_error("Irrigation model training failed")
        return False
    
    print_success("Irrigation model trained")
    return True

def verify_models():
    print_header("Verifying Model Files")
    
    required_files = [
        "backend/models/yield_model/yield_rf_model.pkl",
        "backend/models/yield_model/yield_nn_model.keras",
        "backend/models/yield_model/yield_scaler.pkl",
        "backend/models/yield_model/yield_hybrid_weights.txt",
        "backend/models/yield_model/feature_columns.pkl",
        "backend/models/yield_model/unique_values.pkl",
        "backend/models/irrigation_model/irrigation_lstm_model.h5",
        "backend/models/irrigation_model/irrigation_scaler.pkl",
    ]
    
    all_present = True
    for file_path in required_files:
        if Path(file_path).exists():
            print_success(f"{Path(file_path).name}")
        else:
            print_error(f"{Path(file_path).name} - MISSING")
            all_present = False
    
    if all_present:
        print_success("\nAll model files present!")
    else:
        print_error("\nSome model files are missing")
    
    return all_present

def create_startup_scripts():
    print_header("Creating Startup Scripts")
    
    # Windows batch file
    windows_script = """@echo off
echo Starting Smart Harvest AI...
echo.

echo Starting Backend Server...
start "Backend" cmd /k "venv\\Scripts\\activate && cd backend && python app_v2.py"

timeout /t 3 /nobreak >nul

echo Starting Frontend Server...
start "Frontend" cmd /k "cd frontend && python -m http.server 8000"

timeout /t 2 /nobreak >nul

echo.
echo ========================================
echo Smart Harvest AI is running!
echo ========================================
echo Backend:  http://localhost:5000
echo Frontend: http://localhost:8000
echo.
echo Open http://localhost:8000 in your browser
echo Close the terminal windows to stop.
echo ========================================

timeout /t 3 /nobreak >nul
start http://localhost:8000
"""
    
    # Unix shell script
    unix_script = """#!/bin/bash

echo "Starting Smart Harvest AI..."
echo ""

# Start backend
echo "Starting Backend Server..."
source venv/bin/activate
cd backend
python app_v2.py &
BACKEND_PID=$!
cd ..

sleep 3

# Start frontend
echo "Starting Frontend Server..."
cd frontend
python -m http.server 8000 &
FRONTEND_PID=$!
cd ..

sleep 2

echo ""
echo "========================================"
echo "Smart Harvest AI is running!"
echo "========================================"
echo "Backend:  http://localhost:5000"
echo "Frontend: http://localhost:8000"
echo ""
echo "Open http://localhost:8000 in your browser"
echo "Press Ctrl+C to stop all servers"
echo "========================================"

# Open browser
sleep 2
if command -v xdg-open > /dev/null; then
    xdg-open http://localhost:8000
elif command -v open > /dev/null; then
    open http://localhost:8000
fi

# Wait for interrupt
trap "echo ''; echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
"""
    
    # Write files
    with open("start.bat", "w") as f:
        f.write(windows_script)
    print_success("Created start.bat (Windows)")
    
    with open("start.sh", "w") as f:
        f.write(unix_script)
    
    if platform.system() != "Windows":
        os.chmod("start.sh", 0o755)
    
    print_success("Created start.sh (Linux/Mac)")
    return True

def print_final_instructions():
    print_header("Setup Complete!")
    
    print(f"{Colors.GREEN}{Colors.BOLD}🎉 Smart Harvest AI is ready!{Colors.END}\n")
    
    print(f"{Colors.BOLD}To start the application:{Colors.END}\n")
    
    if platform.system() == "Windows":
        print(f"  {Colors.BLUE}Double-click:{Colors.END} start.bat")
        print(f"  {Colors.BLUE}Or run:{Colors.END} start.bat\n")
    else:
        print(f"  {Colors.BLUE}Run:{Colors.END} ./start.sh\n")
    
    print(f"{Colors.BOLD}Manual start:{Colors.END}\n")
    print("  Terminal 1 (Backend):")
    if platform.system() == "Windows":
        print("    venv\\Scripts\\activate")
    else:
        print("    source venv/bin/activate")
    print("    cd backend")
    print("    python app_v2.py\n")
    
    print("  Terminal 2 (Frontend):")
    print("    cd frontend")
    print("    python -m http.server 8000\n")
    
    print(f"{Colors.BOLD}Then open in browser:{Colors.END}")
    print(f"  {Colors.GREEN}http://localhost:8000{Colors.END}\n")

def main():
    print_header("Smart Harvest AI - Setup Script")
    
    print("This script will:")
    print("  1. Check system requirements")
    print("  2. Create directory structure")
    print("  3. Setup virtual environment")
    print("  4. Install dependencies")
    print("  5. Train ML models")
    print("  6. Create startup scripts\n")
    
    response = input(f"{Colors.YELLOW}Continue? (y/n): {Colors.END}").lower()
    if response != 'y':
        print("Setup cancelled.")
        return
    
    # Run setup steps
    steps = [
        (check_python_version, "Check Python version"),
        (check_dataset, "Verify dataset"),
        (create_directories, "Create directories"),
        (create_virtual_environment, "Setup virtual environment"),
        (install_dependencies, "Install dependencies"),
        (train_models, "Train ML models"),
        (verify_models, "Verify models"),
        (create_startup_scripts, "Create startup scripts"),
    ]
    
    for step_func, step_name in steps:
        if not step_func():
            print_error(f"\nSetup failed at: {step_name}")
            print_warning("Please fix the error and run setup again.")
            return False
    
    print_final_instructions()
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Setup interrupted by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {e}{Colors.END}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
