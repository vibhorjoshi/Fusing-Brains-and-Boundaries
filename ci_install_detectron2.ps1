# Install detectron2 for Windows CI/CD pipeline
# This script should be run after installing the main requirements

Write-Host "Installing detectron2 from source..." -ForegroundColor Cyan
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

Write-Host "Installing additional dependencies..." -ForegroundColor Cyan
python -m pip install -r additional_requirements.txt

Write-Host "Installation complete." -ForegroundColor Green