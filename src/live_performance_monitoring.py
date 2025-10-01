import os
import time
import psutil
import GPUtil
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Summary

# Set up Prometheus metrics
REQUEST_COUNT = Counter('geoai_api_requests_total', 'Total number of API requests', ['endpoint'])
REQUEST_TIME = Histogram('geoai_api_request_duration_seconds', 'API request duration in seconds', ['endpoint'])
PROCESSING_TIME = Summary('geoai_processing_time_seconds', 'Time spent processing images')
DETECTION_ACCURACY = Gauge('geoai_detection_accuracy', 'Current detection accuracy score')
MODEL_CONFIDENCE = Gauge('geoai_model_confidence', 'Current model confidence')
CPU_USAGE = Gauge('geoai_cpu_usage_percent', 'CPU usage percent')
RAM_USAGE = Gauge('geoai_ram_usage_percent', 'RAM usage percent')
GPU_USAGE = Gauge('geoai_gpu_usage_percent', 'GPU usage percent', ['gpu'])
GPU_MEMORY = Gauge('geoai_gpu_memory_percent', 'GPU memory usage percent', ['gpu'])
PROCESSING_STAGES = Gauge('geoai_processing_stages', 'Processing stage time', ['stage'])
ACTIVE_MODELS = Gauge('geoai_active_models', 'Number of active models in use')
IMAGE_PROCESSED = Counter('geoai_images_processed_total', 'Total number of images processed')

# Initialize some example metrics
DETECTION_ACCURACY.set(95.7)  # 95.7% accuracy example
MODEL_CONFIDENCE.set(0.87)    # 87% confidence example
ACTIVE_MODELS.set(3)          # 3 models in use

class LivePerformanceMonitor:
    """
    Live performance monitoring class for GeoAI system.
    Collects and exposes metrics for CPU, RAM, GPU usage, and application-specific metrics.
    """
    
    def __init__(self, port=8004):
        """Initialize the performance monitor and start the metrics server"""
        self.port = port
        self.interval = int(os.environ.get('MONITORING_INTERVAL', 60))  # seconds
        self.running = False
        
        # Start Prometheus metrics server
        start_http_server(self.port)
        print(f"Performance metrics server started on port {self.port}")
        
    def start(self):
        """Start continuous monitoring"""
        self.running = True
        print(f"Starting continuous performance monitoring (interval: {self.interval}s)")
        
        try:
            while self.running:
                self.collect_metrics()
                time.sleep(self.interval)
        except KeyboardInterrupt:
            print("Performance monitoring stopped by user")
            self.running = False
        except Exception as e:
            print(f"Error in performance monitoring: {e}")
            self.running = False
    
    def stop(self):
        """Stop continuous monitoring"""
        self.running = False
        
    def collect_metrics(self):
        """Collect and update all metrics"""
        self._collect_system_metrics()
        self._collect_gpu_metrics()
        
        # Example of tracking processing stages
        self._update_processing_stages()
        
    def _collect_system_metrics(self):
        """Collect system metrics (CPU, RAM)"""
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        CPU_USAGE.set(cpu_percent)
        
        # RAM usage
        mem = psutil.virtual_memory()
        RAM_USAGE.set(mem.percent)
    
    def _collect_gpu_metrics(self):
        """Collect GPU metrics if available"""
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                GPU_USAGE.labels(gpu=f"gpu{i}").set(gpu.load * 100)
                GPU_MEMORY.labels(gpu=f"gpu{i}").set(gpu.memoryUtil * 100)
        except Exception as e:
            print(f"Error collecting GPU metrics: {e}")
    
    def _update_processing_stages(self):
        """Update example processing stage metrics"""
        # These would be updated by actual processing stages in the real system
        stages = {
            "preprocessing": 120,  # ms
            "detection": 450,      # ms
            "segmentation": 380,   # ms
            "postprocessing": 150  # ms
        }
        
        for stage, time_ms in stages.items():
            PROCESSING_STAGES.labels(stage=stage).set(time_ms)
    
    def record_request(self, endpoint, duration):
        """Record an API request"""
        REQUEST_COUNT.labels(endpoint=endpoint).inc()
        REQUEST_TIME.labels(endpoint=endpoint).observe(duration)
    
    def record_processing(self, duration):
        """Record image processing time"""
        PROCESSING_TIME.observe(duration)
        IMAGE_PROCESSED.inc()
    
    def update_detection_metrics(self, accuracy, confidence):
        """Update detection quality metrics"""
        DETECTION_ACCURACY.set(accuracy)
        MODEL_CONFIDENCE.set(confidence)
    
    def update_active_models(self, count):
        """Update the number of active models"""
        ACTIVE_MODELS.set(count)

# Create a singleton instance
monitor = LivePerformanceMonitor(port=int(os.environ.get('METRICS_PORT', 8004)))

if __name__ == "__main__":
    # Run standalone monitoring server
    monitor.start()