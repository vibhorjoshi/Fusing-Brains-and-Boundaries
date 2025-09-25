FROM condaforge/mambaforge:latest

LABEL maintainer="Building Footprint API"

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CONDA_AUTO_UPDATE_CONDA=false

# Create conda environment with geospatial dependencies
COPY environment.yml /app/
RUN mamba env create -f environment.yml && \
    conda clean -afy

# Activate environment
ENV PATH /opt/conda/envs/buildingapi/bin:$PATH
SHELL ["conda", "run", "-n", "buildingapi", "/bin/bash", "-c"]

# Copy requirements and install additional pip packages
COPY requirements.txt /app/
RUN conda run -n buildingapi pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/temp

# Set permissions
RUN chmod +x /app/scripts/*.sh

# Expose port
EXPOSE 8000

# Set command
CMD ["conda", "run", "--no-capture-output", "-n", "buildingapi", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]