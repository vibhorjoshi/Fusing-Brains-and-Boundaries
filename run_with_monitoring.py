#!/usr/bin/env python
"""
GeoAI Project Runner with Monitoring
Runs both frontend and backend services with quality checks and log monitoring.
"""

import os
import sys
import time
import subprocess
import threading
import signal
import logging
import json
import psutil
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("geoai_runner.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Configuration
class Config:
    # Ports
    STREAMLIT_PORT = 8502
    FRONTEND_PORT = 8080
    API_PORT = 8000
    MONITORING_PORT = 9090
    
    # Directories
    PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    VENV_DIR = PROJECT_DIR / ".venv_new"
    FRONTEND_DIR = PROJECT_DIR / "frontend"
    OUTPUT_DIR = PROJECT_DIR / "outputs" / "runner_logs"
    
    # Environment
    ACTIVATE_VENV = str(VENV_DIR / "Scripts" / "Activate.ps1")
    
    # Files to monitor
    LOG_FILES = [
        PROJECT_DIR / "geoai_runner.log",
        PROJECT_DIR / "streamlit_backend.log"
    ]
    
    # Health check endpoints
    HEALTH_ENDPOINTS = {
        "streamlit": f"http://localhost:{STREAMLIT_PORT}",
        "frontend": f"http://localhost:{FRONTEND_PORT}",
        "api": f"http://localhost:{API_PORT}/health"
    }

class ServiceRunner:
    def __init__(self):
        self.processes = {}
        self.stop_event = threading.Event()
        self.config = Config()
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories."""
        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output directory created: {self.config.OUTPUT_DIR}")

    def run_command(self, name, command, cwd=None, shell=True):
        """Run a command in a subprocess and capture output."""
        logging.info(f"Starting {name}: {command}")
        
        log_file = self.config.OUTPUT_DIR / f"{name}.log"
        with open(log_file, "w") as log:
            process = subprocess.Popen(
                command,
                cwd=cwd or self.config.PROJECT_DIR,
                shell=shell,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True
            )
            self.processes[name] = process
            logging.info(f"{name} started with PID {process.pid}")
            return process

    def start_streamlit(self):
        """Start Streamlit backend."""
        # Fix for PowerShell execution issues
        command = f"cd {self.config.PROJECT_DIR} && python -m streamlit run streamlit_backend.py --server.port {self.config.STREAMLIT_PORT}"
        return self.run_command("streamlit", command)
    
    def start_frontend(self):
        """Start frontend server."""
        if (self.config.FRONTEND_DIR / "package.json").exists():
            command = f"cd '{self.config.FRONTEND_DIR}' && npm run dev -- --port {self.config.FRONTEND_PORT}"
            return self.run_command("frontend", command, cwd=self.config.FRONTEND_DIR)
        else:
            # Use simple HTTP server as fallback
            command = f"& '{self.config.ACTIVATE_VENV}'; python -m http.server {self.config.FRONTEND_PORT}"
            return self.run_command("frontend", command)
    
    def start_monitoring(self):
        """Start the monitoring service."""
        metrics_script = str(self.config.PROJECT_DIR / "generate_usa_metrics.py")
        # Fix for PowerShell execution issues
        command = f"cd {self.config.PROJECT_DIR} && python {metrics_script}"
        return self.run_command("monitoring", command)
        
    def start_monitoring_dashboard(self):
        """Start the monitoring dashboard."""
        dashboard_script = str(self.config.PROJECT_DIR / "monitoring_dashboard.py")
        # Fix for PowerShell execution issues
        command = f"cd {self.config.PROJECT_DIR} && python -m streamlit run {dashboard_script} --server.port {self.config.MONITORING_PORT}"
        return self.run_command("monitoring_dashboard", command)
    
    def start_all_services(self):
        """Start all services."""
        logging.info("Starting all GeoAI services...")
        
        # Start streamlit backend
        self.start_streamlit()
        time.sleep(2)  # Give streamlit time to start
        
        # Start frontend
        self.start_frontend()
        time.sleep(2)  # Give frontend time to start
        
        # Run initial monitoring
        self.start_monitoring()
        time.sleep(2)  # Give monitoring time to complete
        
        # Start monitoring dashboard
        self.start_monitoring_dashboard()
        
        logging.info("All services started!")
        self.print_access_info()

    def print_access_info(self):
        """Print access information for services."""
        access_info = f"""
{'='*50}
GeoAI SERVICES ACCESS INFORMATION
{'='*50}
Streamlit Dashboard: http://localhost:{self.config.STREAMLIT_PORT}
Frontend: http://localhost:{self.config.FRONTEND_PORT}
Monitoring Dashboard: http://localhost:{self.config.MONITORING_PORT}
{'='*50}
        """
        logging.info(access_info)
        print(access_info)

    def monitor_services(self):
        """Monitor service health and logs."""
        logging.info("Starting service monitoring...")
        
        try:
            while not self.stop_event.is_set():
                self.check_service_health()
                self.capture_resource_usage()
                self.check_for_errors()
                
                # Run metrics generation periodically
                if datetime.now().minute % 10 == 0:  # Every 10 minutes
                    self.start_monitoring()
                    
                time.sleep(30)  # Check every 30 seconds
        except KeyboardInterrupt:
            logging.info("Monitoring stopped by user.")
        except Exception as e:
            logging.error(f"Error in monitoring: {str(e)}")
        finally:
            self.cleanup()
    
    def check_service_health(self):
        """Check if services are still running."""
        for name, process in list(self.processes.items()):
            if process.poll() is not None:
                logging.warning(f"Service {name} exited with code {process.returncode}")
                if process.returncode != 0:
                    # Try to restart the service
                    logging.info(f"Attempting to restart {name}...")
                    if name == "streamlit":
                        self.start_streamlit()
                    elif name == "frontend":
                        self.start_frontend()
                    elif name == "monitoring":
                        self.start_monitoring()
                    elif name == "monitoring_dashboard":
                        self.start_monitoring_dashboard()
                    
    def capture_resource_usage(self):
        """Capture and log resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Check individual process usage
            process_stats = {}
            for name, process in self.processes.items():
                try:
                    if process.poll() is None:  # Process is still running
                        p = psutil.Process(process.pid)
                        process_stats[name] = {
                            "cpu_percent": p.cpu_percent(interval=0.1),
                            "memory_percent": p.memory_percent(),
                            "status": "running"
                        }
                    else:
                        process_stats[name] = {
                            "status": "stopped",
                            "exit_code": process.returncode
                        }
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    process_stats[name] = {"status": "unknown"}
            
            # Log the resource usage
            usage_data = {
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                },
                "processes": process_stats
            }
            
            # Save to file
            self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            usage_file = self.config.OUTPUT_DIR / "resource_usage.json"
            with open(usage_file, "w") as f:
                json.dump(usage_data, f, indent=2)
                
            logging.info(f"System: CPU {cpu_percent}%, Memory {memory_percent}%")
        except Exception as e:
            logging.error(f"Error capturing resource usage: {str(e)}")
    
    def check_for_errors(self):
        """Check log files for errors."""
        error_keywords = ["error", "exception", "failed", "traceback", "critical"]
        
        for log_file in self.config.LOG_FILES:
            if log_file.exists():
                try:
                    with open(log_file, "r") as f:
                        # Read the last 50 lines
                        all_lines = f.readlines()
                        lines = all_lines[-50:] if len(all_lines) > 50 else all_lines
                        
                    for line in lines:
                        line_lower = line.lower()
                        if any(keyword in line_lower for keyword in error_keywords):
                            logging.warning(f"Possible error in {log_file.name}: {line.strip()}")
                except Exception as e:
                    logging.error(f"Error checking log file {log_file}: {str(e)}")
    
    def cleanup(self):
        """Clean up and stop all processes."""
        logging.info("Stopping all services...")
        
        for name, process in self.processes.items():
            if process.poll() is None:  # Process is still running
                try:
                    logging.info(f"Terminating {name} (PID {process.pid})...")
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logging.warning(f"Process {name} did not terminate, killing...")
                    process.kill()
                except Exception as e:
                    logging.error(f"Error stopping {name}: {str(e)}")
        
        logging.info("All services stopped.")

def main():
    """Main function."""
    runner = ServiceRunner()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        logging.info("Shutdown signal received. Stopping services...")
        runner.stop_event.set()
        runner.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        runner.start_all_services()
        runner.monitor_services()
    except KeyboardInterrupt:
        logging.info("Interrupted by user. Shutting down...")
    finally:
        runner.cleanup()

if __name__ == "__main__":
    main()