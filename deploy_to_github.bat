@echo off
REM GitHub Deployment Script for Real USA Agricultural Detection System
REM Run this batch file to initialize Git, commit code, and push to GitHub

echo 🚀 Starting GitHub deployment for Real USA Agricultural Detection System...

REM Initialize Git repository
git init

REM Configure Git user (replace with your details)
git config user.name "Your Name"
git config user.email "your.email@example.com"

REM Add all files
echo ➕ Adding all files to Git...
git add .

REM Initial commit with comprehensive message
echo 📝 Creating initial commit...
git commit -m "🚀 Initial commit: Real USA Agricultural Detection System - Enhanced Adaptive Fusion API with original architecture pipeline - Beautiful animated dashboard with live performance metrics - Complete Docker containerization for all services - GitHub Actions CI/CD pipeline ready for deployment - Redis live storage system integrated - Multi-satellite provider support (NASA, Google, Sentinel, MODIS) - Real-time WebSocket updates and live tracking - 5 major USA agricultural regions supported"

REM Add remote origin (replace with your GitHub repository URL)
echo 🔗 Adding GitHub remote repository...
git remote add origin https://github.com/vibhorjoshi/Fusing-Brains-and-Boundaries.git

REM Push to GitHub
echo 🚀 Pushing to GitHub...
git branch -M main
git push -u origin main

echo.
echo 🎉 Successfully pushed to GitHub!
echo 🚀 GitHub Actions will now trigger the CI/CD pipeline
echo 🌐 Check your repository for the automated deployment process
echo.
echo Next steps:
echo 1. Go to https://github.com/vibhorjoshi/Fusing-Brains-and-Boundaries
echo 2. Check the Actions tab for CI/CD pipeline status  
echo 3. Monitor Docker container builds and deployments
echo 4. Access the deployed system once pipeline completes
echo.
pause