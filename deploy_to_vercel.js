node deploy_to_vercel.js --production#!/usr/bin/env node

/**
 * Vercel Deployment Script for GeoAI USA Agricultural Detection System
 * This script prepares and deploys the frontend to Vercel
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Configuration
const FRONTEND_DIR = path.join(__dirname, 'frontend');
const PROD_ENV_FILE = path.join(FRONTEND_DIR, '.env.production');

// Colors for console output
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  red: '\x1b[31m'
};

console.log(`${colors.blue}
========================================
   GeoAI Frontend Vercel Deployment
========================================
${colors.reset}`);

// Check if Vercel CLI is installed
function checkVercelCLI() {
  try {
    console.log(`${colors.yellow}Checking for Vercel CLI...${colors.reset}`);
    execSync('vercel --version', { stdio: 'ignore' });
    console.log(`${colors.green}✓ Vercel CLI is installed${colors.reset}`);
    return true;
  } catch (error) {
    console.log(`${colors.red}✗ Vercel CLI is not installed${colors.reset}`);
    console.log(`${colors.yellow}Installing Vercel CLI...${colors.reset}`);
    try {
      execSync('npm install -g vercel', { stdio: 'inherit' });
      console.log(`${colors.green}✓ Vercel CLI installed successfully${colors.reset}`);
      return true;
    } catch (installError) {
      console.error(`${colors.red}Error installing Vercel CLI: ${installError.message}${colors.reset}`);
      return false;
    }
  }
}

// Check and update environment variables
function updateEnvironmentVariables() {
  console.log(`${colors.yellow}Checking environment configuration...${colors.reset}`);
  
  if (!fs.existsSync(PROD_ENV_FILE)) {
    console.log(`${colors.red}✗ Production environment file not found at: ${PROD_ENV_FILE}${colors.reset}`);
    return false;
  }

  console.log(`${colors.green}✓ Environment configuration is ready${colors.reset}`);
  return true;
}

// Build the frontend
function buildFrontend() {
  console.log(`${colors.yellow}Building frontend for production...${colors.reset}`);
  try {
    process.chdir(FRONTEND_DIR);
    execSync('npm ci', { stdio: 'inherit' });
    execSync('npm run build', { stdio: 'inherit' });
    console.log(`${colors.green}✓ Frontend built successfully${colors.reset}`);
    return true;
  } catch (error) {
    console.error(`${colors.red}✗ Frontend build failed: ${error.message}${colors.reset}`);
    return false;
  }
}

// Deploy to Vercel
function deployToVercel(environment = 'production') {
  console.log(`${colors.yellow}Deploying to Vercel (${environment})...${colors.reset}`);
  try {
    process.chdir(FRONTEND_DIR);
    const command = environment === 'production' 
      ? 'vercel --prod'
      : 'vercel';
    
    execSync(command, { stdio: 'inherit' });
    console.log(`${colors.green}✓ Deployment successful${colors.reset}`);
    return true;
  } catch (error) {
    console.error(`${colors.red}✗ Deployment failed: ${error.message}${colors.reset}`);
    return false;
  }
}

// Main function
async function main() {
  const args = process.argv.slice(2);
  const isProd = args.includes('--prod') || args.includes('--production');
  const environment = isProd ? 'production' : 'preview';
  
  console.log(`${colors.blue}Deploying to ${environment} environment${colors.reset}`);
  
  // Run deployment steps
  const hasVercel = checkVercelCLI();
  if (!hasVercel) {
    process.exit(1);
  }
  
  const envReady = updateEnvironmentVariables();
  if (!envReady) {
    process.exit(1);
  }
  
  const buildSuccess = buildFrontend();
  if (!buildSuccess) {
    process.exit(1);
  }
  
  const deploySuccess = deployToVercel(environment);
  if (!deploySuccess) {
    process.exit(1);
  }
  
  console.log(`${colors.blue}
========================================
   Deployment Complete! 
   
   Your GeoAI frontend is now live on Vercel.
   Check your Vercel dashboard for the URL.
========================================
${colors.reset}`);
}

// Run the script
main().catch(error => {
  console.error(`${colors.red}Unexpected error: ${error.message}${colors.reset}`);
  process.exit(1);
});