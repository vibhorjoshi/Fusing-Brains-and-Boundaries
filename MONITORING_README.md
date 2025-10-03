# GeoAI Monitoring System

This monitoring system provides comprehensive oversight and management of the GeoAI research project services. It includes automatic service management, resource monitoring, and performance metrics.

## Overview

The monitoring system consists of the following components:

1. **Service Runner**: Manages all GeoAI services, including the Streamlit backend, frontend, and monitoring tools
2. **Resource Monitor**: Tracks CPU and memory usage of all running services
3. **USA Agricultural Metrics Generator**: Produces detailed agricultural performance metrics
4. **Monitoring Dashboard**: Visualizes system health and performance metrics

## Services

The monitoring system manages the following services:

- **Streamlit Backend** (Port 8502): Main GeoAI application interface
- **Frontend** (Port 8080): Simple web interface for users
- **Monitoring Dashboard** (Port 9090): System monitoring and performance visualization

## Features

- **Automatic Service Recovery**: Detects and restarts failed services
- **Resource Tracking**: Monitors CPU and memory usage
- **Performance Metrics**: Generates agricultural detection performance metrics
- **Error Detection**: Scans logs for errors and issues
- **Clean Shutdown**: Gracefully terminates all services on exit

## Usage

### Standard Launch

To start the monitoring system with the fixed PowerShell execution:

```powershell
python run_with_monitoring_fixed.py
```

The script will:
1. Start all services (Streamlit, frontend, monitoring)
2. Generate initial metrics
3. Begin monitoring services and resources

### Accessing Services

Once running, you can access the services at:
- Streamlit Dashboard: http://localhost:8502
- Frontend: http://localhost:8080
- Monitoring Dashboard: http://localhost:9090

### Stopping Services

Press `Ctrl+C` in the terminal running the monitoring script to gracefully shut down all services.

## Logs and Outputs

- **Main Log**: `geoai_runner_fixed.log` contains all monitoring events and service status changes
- **Resource Data**: `outputs/runner_logs/resource_usage.json` contains current resource usage metrics
- **USA Metrics**: `outputs/usa_metrics/` contains agricultural performance metrics and visualizations

## Troubleshooting

If services fail to start:

1. Check the respective service log in `outputs/runner_logs/`
2. Ensure Python environment is properly set up (with Streamlit and other dependencies)
3. Check that ports 8080, 8502, and 9090 are not in use by other applications
4. Verify that the required Python modules are installed:
   ```
   pip install streamlit psutil matplotlib seaborn pandas numpy
   ```

## Dependencies

- Python 3.6+
- psutil
- streamlit
- pandas
- matplotlib
- seaborn
- numpy

Install dependencies:
```
pip install -r requirements.txt
```