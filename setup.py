#!/usr/bin/env python3
"""
Setup script for Advanced Hand Gesture Game Controller
========================================================
This script helps set up the environment and dependencies.

Usage:
    python setup.py [--install] [--test] [--check]
"""

import subprocess
import sys
import os
import argparse


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Error: Python 3.8 or higher required (found {version.major}.{version.minor})")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def install_requirements():
    """Install required packages."""
    print("\n📦 Installing required packages...")
    print("-" * 50)
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("-" * 50)
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False


def test_imports():
    """Test if all required packages can be imported."""
    print("\n🧪 Testing package imports...")
    print("-" * 50)
    
    packages = [
        ("cv2", "OpenCV"),
        ("mediapipe", "MediaPipe"),
        ("pyautogui", "PyAutoGUI"),
        ("numpy", "NumPy")
    ]
    
    all_ok = True
    
    for module, name in packages:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name}: {e}")
            all_ok = False
    
    print("-" * 50)
    
    if all_ok:
        print("✅ All packages imported successfully!")
    else:
        print("❌ Some packages failed to import")
    
    return all_ok


def check_camera():
    """Check if camera is available."""
    print("\n📷 Checking camera availability...")
    print("-" * 50)
    
    try:
        import cv2
        
        # Try default camera
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                print(f"  ✅ Camera 0 is available")
                print(f"     Resolution: {width}x{height}")
            else:
                print(f"  ⚠️  Camera 0 opened but cannot read frames")
        else:
            print(f"  ❌ Camera 0 is not available")
        
        cap.release()
        
        # Check for additional cameras
        for i in range(1, 3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"  ✅ Camera {i} is also available")
            cap.release()
        
        print("-" * 50)
        return True
        
    except Exception as e:
        print(f"  ❌ Error checking camera: {e}")
        print("-" * 50)
        return False


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    print("-" * 50)
    
    dirs = ["profiles", "calibration"]
    
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"  ✅ Created '{dir_name}/'")
        else:
            print(f"  ℹ️  '{dir_name}/' already exists")
    
    print("-" * 50)


def print_usage():
    """Print usage instructions."""
    print("\n" + "="*60)
    print("🎮 SETUP COMPLETE!")
    print("="*60)
    print("\nQuick Start:")
    print("  1. Start with default profile (Temple Run):")
    print("     python hand_controller.py")
    print()
    print("  2. Use a specific profile:")
    print("     python hand_controller.py --profile subway_surfers")
    print()
    print("  3. List all available profiles:")
    print("     python hand_controller.py --list")
    print()
    print("  4. Use secondary camera:")
    print("     python hand_controller.py --camera 1")
    print()
    print("Controls:")
    print("  P - Pause/Resume the controller")
    print("  Q - Quit the application")
    print("  C - Calibrate hand size")
    print()
    print("Tips:")
    print("  • Ensure good lighting for better hand detection")
    print("  • Keep your hand within the camera frame")
    print("  • Position yourself 2-3 feet from the camera")
    print("  • Run the calibration for better accuracy")
    print()
    print("="*60)


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Setup script for Advanced Hand Gesture Game Controller"
    )
    
    parser.add_argument(
        '--install', '-i',
        action='store_true',
        help='Install required packages'
    )
    
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Test package imports'
    )
    
    parser.add_argument(
        '--check', '-c',
        action='store_true',
        help='Check camera availability'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run all setup steps'
    )
    
    args = parser.parse_args()
    
    # If no arguments, run all
    if not any([args.install, args.test, args.check, args.all]):
        args.all = True
    
    print("\n" + "="*60)
    print("🚀 Advanced Hand Gesture Game Controller - Setup")
    print("="*60 + "\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if args.all or args.install:
        if not install_requirements():
            print("\n❌ Setup failed. Please check the error messages above.")
            sys.exit(1)
    
    # Test imports
    if args.all or args.test:
        if not test_imports():
            print("\n❌ Some packages are missing. Run with --install to fix.")
            sys.exit(1)
    
    # Check camera
    if args.all or args.check:
        check_camera()
    
    # Create directories
    create_directories()
    
    # Print usage
    if args.all:
        print_usage()


if __name__ == "__main__":
    main()
