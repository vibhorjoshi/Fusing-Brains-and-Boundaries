"""
Deployment script for GeoAI Research Project
Starts both backend and frontend servers
"""

import os
import sys
import time
import subprocess
import threading
from pathlib import Path

class ServerManager:
    """Manages both backend and frontend servers"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.backend_dir = self.project_root / "backend"
        self.frontend_dir = self.project_root / "frontend"
        self.processes = []
    
    def check_dependencies(self):
        """Check if required dependencies are installed"""
        print("🔍 Checking dependencies...")
        
        # Check Python
        try:
            result = subprocess.run([sys.executable, "--version"], 
                                  capture_output=True, text=True)
            print(f"✅ Python: {result.stdout.strip()}")
        except Exception as e:
            print(f"❌ Python check failed: {e}")
            return False
        
        # Check Node.js
        try:
            result = subprocess.run(["node", "--version"], 
                                  capture_output=True, text=True)
            print(f"✅ Node.js: {result.stdout.strip()}")
        except Exception as e:
            print(f"❌ Node.js not found: {e}")
            print("Please install Node.js from https://nodejs.org/")
            return False
        
        # Check npm
        try:
            result = subprocess.run(["npm", "--version"], 
                                  capture_output=True, text=True)
            print(f"✅ npm: {result.stdout.strip()}")
        except Exception as e:
            print(f"❌ npm not found: {e}")
            return False
        
        return True
    
    def setup_backend(self):
        """Setup backend environment"""
        print("\n🐍 Setting up backend...")
        
        if not self.backend_dir.exists():
            print(f"❌ Backend directory not found: {self.backend_dir}")
            return False
        
        os.chdir(self.backend_dir)
        
        # Create virtual environment if it doesn't exist
        venv_dir = self.project_root / ".venv"
        if not venv_dir.exists():
            print("📦 Creating Python virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        
        # Install backend dependencies
        if (self.backend_dir / "requirements.txt").exists():
            print("📦 Installing Python dependencies...")
            if os.name == 'nt':  # Windows
                pip_cmd = str(venv_dir / "Scripts" / "pip.exe")
            else:  # Unix-like
                pip_cmd = str(venv_dir / "bin" / "pip")
            
            subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
        
        print("✅ Backend setup complete")
        return True
    
    def setup_frontend(self):
        """Setup frontend environment"""
        print("\n⚛️  Setting up frontend...")
        
        if not self.frontend_dir.exists():
            print(f"❌ Frontend directory not found: {self.frontend_dir}")
            return False
        
        os.chdir(self.frontend_dir)
        
        # Install frontend dependencies
        if (self.frontend_dir / "package.json").exists():
            print("📦 Installing Node.js dependencies...")
            subprocess.run(["npm", "install"], check=True)
        
        print("✅ Frontend setup complete")
        return True
    
    def start_backend(self):
        """Start backend server"""
        print("\n🚀 Starting backend server...")
        
        os.chdir(self.backend_dir)
        
        # Determine Python executable
        venv_dir = self.project_root / ".venv"
        if os.name == 'nt':  # Windows
            python_cmd = str(venv_dir / "Scripts" / "python.exe")
        else:  # Unix-like
            python_cmd = str(venv_dir / "bin" / "python")
        
        # Start backend
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.backend_dir)
        
        backend_process = subprocess.Popen(
            [python_cmd, "start.py"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        self.processes.append(("backend", backend_process))
        
        # Monitor backend output in separate thread
        def monitor_backend():
            for line in iter(backend_process.stdout.readline, ''):
                print(f"[Backend] {line.strip()}")
        
        threading.Thread(target=monitor_backend, daemon=True).start()
        
        return backend_process
    
    def start_frontend(self):
        """Start frontend server"""
        print("\n🌐 Starting frontend server...")
        
        os.chdir(self.frontend_dir)
        
        # Wait a bit for backend to start
        time.sleep(5)
        
        # Start frontend
        frontend_process = subprocess.Popen(
            ["npm", "run", "dev"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        self.processes.append(("frontend", frontend_process))
        
        # Monitor frontend output in separate thread
        def monitor_frontend():
            for line in iter(frontend_process.stdout.readline, ''):
                print(f"[Frontend] {line.strip()}")
        
        threading.Thread(target=monitor_frontend, daemon=True).start()
        
        return frontend_process
    
    def wait_for_servers(self):
        """Wait for servers to be ready"""
        print("\n⏳ Waiting for servers to start...")
        
        # Wait for backend (port 8002)
        backend_ready = False
        for _ in range(30):  # Wait up to 30 seconds
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', 8002))
                sock.close()
                
                if result == 0:
                    backend_ready = True
                    break
                    
            except Exception:
                pass
            
            time.sleep(1)
        
        if backend_ready:
            print("✅ Backend server is ready!")
            print("🔧 Backend API: http://localhost:8002")
            print("📖 API Documentation: http://localhost:8002/docs")
        else:
            print("❌ Backend server failed to start")
        
        # Wait for frontend (port 3000)
        frontend_ready = False
        for _ in range(60):  # Wait up to 60 seconds
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', 3000))
                sock.close()
                
                if result == 0:
                    frontend_ready = True
                    break
                    
            except Exception:
                pass
            
            time.sleep(1)
        
        if frontend_ready:
            print("✅ Frontend server is ready!")
            print("🌐 Frontend App: http://localhost:3000")
        else:
            print("❌ Frontend server failed to start")
        
        return backend_ready and frontend_ready
    
    def shutdown(self):
        """Shutdown all processes"""
        print("\n🛑 Shutting down servers...")
        
        for name, process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=10)
                print(f"✅ {name.title()} server stopped")
            except Exception as e:
                print(f"❌ Error stopping {name}: {e}")
                try:
                    process.kill()
                except:
                    pass
    
    def run(self):
        """Main run method"""
        try:
            print("🛰️  GeoAI Research Project Deployment")
            print("=" * 60)
            
            # Check dependencies
            if not self.check_dependencies():
                return False
            
            # Setup backend
            if not self.setup_backend():
                return False
            
            # Setup frontend
            if not self.setup_frontend():
                return False
            
            # Start servers
            backend_proc = self.start_backend()
            frontend_proc = self.start_frontend()
            
            # Wait for servers
            if self.wait_for_servers():
                print("\n🎉 All servers are running!")
                print("=" * 60)
                print("🚀 GeoAI Research System is ready!")
                print("🌐 Frontend: http://localhost:3000")
                print("🔧 Backend API: http://localhost:8002")
                print("📖 API Docs: http://localhost:8002/docs")
                print("=" * 60)
                print("Press Ctrl+C to stop all servers")
                
                # Wait for interrupt
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
            
            return True
            
        except KeyboardInterrupt:
            print("\n🛑 Deployment interrupted by user")
            return True
        except Exception as e:
            print(f"\n❌ Deployment error: {e}")
            return False
        finally:
            self.shutdown()

def main():
    """Main function"""
    manager = ServerManager()
    success = manager.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()