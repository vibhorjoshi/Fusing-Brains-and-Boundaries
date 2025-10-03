FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1-mesa-glx \
        ffmpeg \
        git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements files first for better caching
COPY requirements_ci.txt .
COPY additional_requirements.txt .
COPY ci_install_detectron2.sh .

# Install Python dependencies with special handling for detectron2
RUN pip install --no-cache-dir -r requirements_ci.txt && \
    bash ci_install_detectron2.sh

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/usa \
    /app/outputs/runner_logs \
    /app/outputs/usa_metrics \
    /app/frontend

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 8502
EXPOSE 8080
EXPOSE 9090
EXPOSE 8000

# Set entrypoint to run the monitoring system
CMD ["python", "run_with_monitoring_fixed.py"]