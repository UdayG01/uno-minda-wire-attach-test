import os
import subprocess
import sys
from pathlib import Path

def create_executable():
    """
    Create executable for push_detector10.py using PyInstaller
    Uses the existing workspace structure and dependencies
    """
    print("🎯 PUSH DETECTOR EXECUTABLE BUILDER")
    print("="*50)
    
    # Get current directory (src2)
    current_dir = Path(__file__).parent
    root_dir = current_dir.parent
    
    print(f"📁 Current directory: {current_dir}")
    print(f"📁 Root directory: {root_dir}")
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Virtual environment not detected!")
        print("   Please activate your venv first:")
        print("   > cd ..\\..\\venv\\Scripts")
        print("   > activate")
        print("   > cd ..\\..\\src2")
        print("   > python create_executable.py")
        return False
    
    print(f"✅ Virtual environment active: {sys.prefix}")
    
    # Verify required files exist
    required_files = {
        'push_detector10.py': current_dir / 'push_detector10.py',
        'video_handler.py': current_dir / 'video_handler.py',
        'terminal_boundary_drawer.py': current_dir / 'terminal_boundary_drawer.py',
        'connector_detector.py': current_dir / 'connector_detector.py',
        'test_1_arduino.py': current_dir / 'test_1_arduino.py',
        'postgres_database_manager.py': current_dir / 'postgres_database_manager.py',
        'model/seg_con2.pt': root_dir / 'model' / 'seg-con2.pt',
        'terminal_config.json': root_dir / 'terminal_config.json'
    }
    
    print("\n🔍 Checking required files...")
    missing_files = []
    for name, path in required_files.items():
        if path.exists():
            print(f"   ✅ {name}")
        else:
            print(f"   ❌ {name} - NOT FOUND: {path}")
            missing_files.append(name)
    
    if missing_files:
        print(f"\n❌ Missing required files: {missing_files}")
        return False
    
    # Check if PyInstaller is available
    try:
        result = subprocess.run(['pyinstaller', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"\n✅ PyInstaller version: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n❌ PyInstaller not found! Installing...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyinstaller'], 
                          check=True)
            print("✅ PyInstaller installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install PyInstaller")
            return False
    
    # Clean previous builds
    print("\n🧹 Cleaning previous builds...")
    build_dir = current_dir / 'build'
    dist_dir = current_dir / 'dist'
    
    if build_dir.exists():
        import shutil
        shutil.rmtree(build_dir)
        print("   ✅ Removed build directory")
    
    if dist_dir.exists():
        import shutil
        shutil.rmtree(dist_dir)
        print("   ✅ Removed dist directory")
    
    # Prepare PyInstaller command
    print("\n🚀 Building executable with PyInstaller...")
    
    pyinstaller_cmd = [
        'pyinstaller',
        '--onefile',  # Create single executable file
        '--name', 'PushDetector10',
        '--console',  # Keep console window
        '--add-data', f'{root_dir / "model"};model',  # Include model directory
        '--add-data', f'{root_dir / "terminal_config.json"};.',  # Include config file
        '--add-data', f'{root_dir / "requirements.txt"};.',  # Include requirements
        # Hidden imports for all dependencies
        '--hidden-import', 'cv2',
        '--hidden-import', 'numpy',
        '--hidden-import', 'torch',
        '--hidden-import', 'torchvision',
        '--hidden-import', 'ultralytics',
        '--hidden-import', 'pyserial',
        '--hidden-import', 'psycopg2',
        '--hidden-import', 'ultralytics.models',
        '--hidden-import', 'ultralytics.engine',
        '--hidden-import', 'ultralytics.utils',
        '--hidden-import', 'sqlite3',
        '--hidden-import', 'dotenv',
        # Include the main script
        'push_detector10.py'
    ]
    
    print("   Command:", ' '.join(pyinstaller_cmd))
    
    try:
        # Run PyInstaller
        result = subprocess.run(pyinstaller_cmd, cwd=current_dir, 
                              capture_output=True, text=True, check=True)
        
        print("\n✅ PyInstaller completed successfully!")
        
        # Check if executable was created
        exe_path = dist_dir / 'PushDetector10.exe'
        if exe_path.exists():
            exe_size = exe_path.stat().st_size / (1024 * 1024)  # Size in MB
            print(f"🎉 EXECUTABLE CREATED SUCCESSFULLY!")
            print(f"   📁 Location: {exe_path}")
            print(f"   📊 Size: {exe_size:.1f} MB")
            
            # Create a simple batch file to run the executable
            batch_content = f"""@echo off
echo Starting Push Detector 10...
echo.
cd /d "{exe_path.parent}"
PushDetector10.exe
echo.
echo Press any key to exit...
pause >nul
"""
            batch_path = exe_path.parent / 'run_push_detector.bat'
            with open(batch_path, 'w') as f:
                f.write(batch_content)
            
            print(f"   🎯 Batch file created: {batch_path}")
            print(f"\n📋 Usage Instructions:")
            print(f"   1. Double-click: {exe_path.name}")
            print(f"   2. Or use batch file: {batch_path.name}")
            print(f"   3. Executable includes all dependencies and model files")
            print(f"   4. Make sure your Arduino is connected to the configured COM port")
            
            return True
        else:
            print("❌ Executable file not found after build")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ PyInstaller failed with error:")
        print(f"   Return code: {e.returncode}")
        print(f"   STDOUT: {e.stdout}")
        print(f"   STDERR: {e.stderr}")
        return False

def main():
    """Main function to create executable"""
    print("🔧 Push Detector 10 - Executable Builder")
    print("   This script creates a standalone executable from push_detector10.py")
    print("   All dependencies, model files, and configuration will be included")
    print()
    
    # Ask for confirmation
    response = input("Do you want to build the executable? (y/n): ").strip().lower()
    if response != 'y':
        print("❌ Build cancelled")
        return
    
    success = create_executable()
    
    if success:
        print("\n🎉 BUILD COMPLETED SUCCESSFULLY!")
        print("   Your executable is ready to use on any Windows machine")
        print("   No need to install Python or dependencies on target machine")
    else:
        print("\n❌ BUILD FAILED!")
        print("   Please check the error messages above")
        print("   Make sure all dependencies are installed in your virtual environment")

if __name__ == "__main__":
    main()
