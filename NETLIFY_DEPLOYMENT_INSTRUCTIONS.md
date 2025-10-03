# Netlify Deployment Instructions for GeoAI Project

This guide will help you deploy the GeoAI project on Netlify, addressing dependency issues and configuration requirements.

## Preparation

The repository has been configured with Netlify-specific files:

1. `netlify.toml` - Configuration for the Netlify build
2. `netlify.requirements.txt` - A simplified Python requirements file that excludes problematic dependencies
3. `netlify-build.sh` - A custom build script to handle dependency installation gracefully

## Deployment Steps

### 1. Connect to Netlify

1. Go to [Netlify's website](https://netlify.com) and sign in
2. Click on "Add new site" and select "Import an existing project"
3. Connect to your GitHub account and select the `Fusing-Brains-and-Boundaries` repository

### 2. Configure Build Settings

The repository already includes a `netlify.toml` file with the correct build settings:
- Base directory: Root directory
- Build command: `bash netlify-build.sh`
- Publish directory: `frontend/.next`

### 3. Environment Variables

Ensure the following environment variables are set in your Netlify project settings:
```
NEXT_PUBLIC_API_URL=https://your-netlify-site.netlify.app/.netlify/functions
NEXT_PUBLIC_GEOAI_ENGINE_URL=https://your-netlify-site.netlify.app/.netlify/functions
```

### 4. Deploy the Site

Click "Deploy site" and Netlify will begin building your project. The custom build script will handle dependency installation issues gracefully.

## Handling Dependency Issues

The main challenge with deploying this project on Netlify is that some Python packages like `detectron2` cannot be installed in Netlify's environment. We've addressed this by:

1. Creating a simplified `netlify.requirements.txt` file that excludes problematic packages
2. Using a custom build script that continues even if some dependencies fail to install
3. Configuring the Netlify build environment to use Python 3.9

## API Functions Limitations

Note that Netlify Functions have limitations that may affect the full functionality of the API:
- Maximum execution time: 10 seconds
- Maximum bundle size: 50MB
- Limited memory: 1024MB

For full functionality, consider deploying the backend separately on a platform like Heroku, AWS, or GCP.

## Troubleshooting

If you encounter issues during deployment:

1. Check the Netlify build logs for specific errors
2. Ensure all required environment variables are set
3. Consider further simplifying the `netlify.requirements.txt` file if dependency issues persist
4. If you need full functionality with all dependencies, consider using Docker with a different hosting provider