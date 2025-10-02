#!/usr/bin/env python3
"""
Performance Monitoring Service
Collects system metrics and docker container stats
"""

import os
import time
import json
import psutil
import docker
import redis
import threading
from datetime import datetime
from flask import Flask, jsonify, Response
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Initialize Flask app
app = Flask(__name__)

# Initialize Redis client
redis_host = os.environ.get('REDIS_HOST', 'localhost')
redis_port = int(os.environ.get('REDIS_PORT', 6379))
redis_client = redis.Redis(host=redis_host, port=redis_port)
print(f"Connected to Redis at {redis_host}:{redis_port}")

# Initialize Docker client
try:
    docker_client = docker.from_env()
    print("Connected to Docker")
except Exception as e:
    docker_client = None
    print(f"Failed to connect to Docker: {e}")

# Prometheus metrics
system_cpu_usage = Gauge('system_cpu_usage', 'System CPU usage percentage')
system_memory_usage = Gauge('system_memory_usage', 'System memory usage percentage')
system_disk_usage = Gauge('system_disk_usage', 'System disk usage percentage')

container_cpu_usage = Gauge('container_cpu_usage', 'Container CPU usage percentage', ['container_name'])
container_memory_usage = Gauge('container_memory_usage', 'Container memory usage in MB', ['container_name'])

api_requests_total = Counter('api_requests_total', 'Total API requests')
processing_time_seconds = Gauge('processing_time_seconds', 'Time to process a request', ['component'])

class PerformanceMonitor:
    """Monitors system performance metrics"""
    
    def __init__(self):
        self.metrics = {
            "system": {
                "cpu_usage": [],
                "memory_usage": [],
                "disk_usage": []
            },
            "containers": {},
            "timestamps": []
        }
        self.monitoring = False
        
    def start_monitoring(self):
        """Start performance monitoring in a separate thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("Performance monitoring started")
        
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            timestamp = datetime.now().isoformat()
            
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_info = psutil.virtual_memory()
                memory_percent = memory_info.percent
                disk_info = psutil.disk_usage('/')
                disk_percent = disk_info.percent
                
                # Update Prometheus metrics
                system_cpu_usage.set(cpu_percent)
                system_memory_usage.set(memory_percent)
                system_disk_usage.set(disk_percent)
                
                # Record metrics
                self.metrics["system"]["cpu_usage"].append(cpu_percent)
                self.metrics["system"]["memory_usage"].append(memory_percent)
                self.metrics["system"]["disk_usage"].append(disk_percent)
                self.metrics["timestamps"].append(timestamp)
                
                # Keep only the last 300 data points (5 minutes at 1s intervals)
                if len(self.metrics["timestamps"]) > 300:
                    self.metrics["system"]["cpu_usage"] = self.metrics["system"]["cpu_usage"][-300:]
                    self.metrics["system"]["memory_usage"] = self.metrics["system"]["memory_usage"][-300:]
                    self.metrics["system"]["disk_usage"] = self.metrics["system"]["disk_usage"][-300:]
                    self.metrics["timestamps"] = self.metrics["timestamps"][-300:]
                
                # Collect Docker container metrics
                if docker_client:
                    containers = docker_client.containers.list()
                    for container in containers:
                        try:
                            stats = container.stats(stream=False)
                            name = container.name
                            
                            # Calculate CPU percentage
                            cpu_delta = float(stats["cpu_stats"]["cpu_usage"]["total_usage"]) - \
                                      float(stats["precpu_stats"]["cpu_usage"]["total_usage"])
                            system_cpu_delta = float(stats["cpu_stats"]["system_cpu_usage"]) - \
                                             float(stats["precpu_stats"]["system_cpu_usage"])
                            num_cpus = len(stats["cpu_stats"]["cpu_usage"]["percpu_usage"])
                            cpu_percent = (cpu_delta / system_cpu_delta) * num_cpus * 100.0
                            
                            # Calculate memory usage
                            memory_usage = float(stats["memory_stats"]["usage"]) / (1024 * 1024)  # MB
                            
                            # Update Prometheus metrics
                            container_cpu_usage.labels(container_name=name).set(cpu_percent)
                            container_memory_usage.labels(container_name=name).set(memory_usage)
                            
                            # Record container metrics
                            if name not in self.metrics["containers"]:
                                self.metrics["containers"][name] = {
                                    "cpu_usage": [],
                                    "memory_usage": []
                                }
                                
                            self.metrics["containers"][name]["cpu_usage"].append(cpu_percent)
                            self.metrics["containers"][name]["memory_usage"].append(memory_usage)
                            
                            # Keep only the last 300 data points
                            if len(self.metrics["containers"][name]["cpu_usage"]) > 300:
                                self.metrics["containers"][name]["cpu_usage"] = self.metrics["containers"][name]["cpu_usage"][-300:]
                                self.metrics["containers"][name]["memory_usage"] = self.metrics["containers"][name]["memory_usage"][-300:]
                                
                        except Exception as e:
                            print(f"Error collecting stats for container {container.name}: {e}")
                
                # Store latest metrics in Redis
                latest_metrics = {
                    "system": {
                        "cpu": self.metrics["system"]["cpu_usage"][-1],
                        "memory": self.metrics["system"]["memory_usage"][-1],
                        "disk": self.metrics["system"]["disk_usage"][-1]
                    },
                    "containers": {},
                    "timestamp": timestamp
                }
                
                for name in self.metrics["containers"]:
                    if self.metrics["containers"][name]["cpu_usage"]:
                        latest_metrics["containers"][name] = {
                            "cpu": self.metrics["containers"][name]["cpu_usage"][-1],
                            "memory": self.metrics["containers"][name]["memory_usage"][-1]
                        }
                
                redis_client.setex(
                    "monitor:latest",
                    3600,  # 1 hour expiration
                    json.dumps(latest_metrics)
                )
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
            
            # Log metrics every 30 seconds
            if len(self.metrics["timestamps"]) % 30 == 0:
                avg_cpu = sum(self.metrics["system"]["cpu_usage"][-30:]) / 30
                avg_mem = sum(self.metrics["system"]["memory_usage"][-30:]) / 30
                print(f"System: CPU {avg_cpu:.1f}%, Memory {avg_mem:.1f}%")
                
                for name in self.metrics["containers"]:
                    if len(self.metrics["containers"][name]["cpu_usage"]) >= 30:
                        avg_cpu = sum(self.metrics["containers"][name]["cpu_usage"][-30:]) / 30
                        avg_mem = sum(self.metrics["containers"][name]["memory_usage"][-30:]) / 30
                        print(f"Container {name}: CPU {avg_cpu:.1f}%, Memory {avg_mem:.1f} MB")
            
            # Sleep before next recording
            time.sleep(1.0)
    
    def get_metrics(self):
        """Get current performance metrics"""
        return self.metrics

# Initialize the performance monitor
monitor = PerformanceMonitor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "monitor"}), 200

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route('/stats', methods=['GET'])
def stats():
    """Get current performance metrics"""
    return jsonify(monitor.get_metrics())

@app.route('/latest', methods=['GET'])
def latest():
    """Get latest metrics from Redis"""
    latest_data = redis_client.get("monitor:latest")
    if latest_data:
        return jsonify(json.loads(latest_data))
    return jsonify({"error": "No metrics available"})

@app.route('/processing-times', methods=['GET'])
def processing_times():
    """Get processing times from Redis"""
    # Get all keys matching the pattern
    keys = redis_client.keys("timing:*")
    times = {}
    
    for key in keys:
        component = key.decode('utf-8').split(':')[1]
        data = redis_client.get(key)
        if data:
            times[component] = json.loads(data)
    
    return jsonify(times)

if __name__ == "__main__":
    # Start performance monitoring
    monitor.start_monitoring()
    
    # Start Flask app
    port = int(os.environ.get('PORT', 9090))
    app.run(host='0.0.0.0', port=port)