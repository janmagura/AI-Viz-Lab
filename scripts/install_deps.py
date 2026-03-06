#!/usr/bin/env python3
"""
AI-Viz-Lab: Automatic Dependency Installer
Installs all required packages and checks system compatibility.
"""

import subprocess
import sys
import os
from pathlib import Path

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")

def check_python_version():
    """Check if Python version is 3.10+."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ ERROR: Python 3.10+ is required!")
        print(f"   Current version: {version.major}.{version.minor}")
        return False
    
    print("✅ Python version OK")
    return True

def install_requirements():
    """Install requirements from requirements.txt."""
    req_file = Path(__file__).parent.parent / "requirements.txt"
    
    if not req_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    print(f"Installing dependencies from {req_file}...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", str(req_file),
            "--upgrade"
        ])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def verify_installation():
    """Verify that key packages are installed."""
    print("\nVerifying installation...")
    
    required_packages = [
        'gradio', 'numpy', 'matplotlib', 'torch', 
        'transformers', 'yaml', 'PIL'
    ]
    
    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg}")
            missing.append(pkg)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("   Try running: pip install -r requirements.txt")
        return False
    
    print("\n✅ All required packages verified!")
    return True

def main():
    """Main installation routine."""
    print_header("🎓 AI-Viz-Lab: Dependency Installer")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("\n⚠️  Installation completed with warnings.")
        print("   You can still try to run the application.")
    
    # Verify installation
    verify_installation()
    
    print_header("Installation Complete!")
    print("Next steps:")
    print("  1. Run: python src/utils/hardware_detect.py --generate-config")
    print("  2. Run: python scripts/download_model.py --model Qwen/Qwen2.5-1.5B-Instruct")
    print("  3. Run: python src/main.py")
    print("\nHappy learning! 🧠✨\n")

if __name__ == "__main__":
    main()
