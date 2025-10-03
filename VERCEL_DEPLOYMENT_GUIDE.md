# Vercel Deployment Guide for GeoAI Project

This guide explains how to deploy the GeoAI project to Vercel for web hosting.

## Prerequisites

1. [Node.js](https://nodejs.org/) installed (v18.0.0 or newer)
2. Vercel CLI installed: `npm install -g vercel`
3. A Vercel account (signup at [vercel.com](https://vercel.com))
4. Vercel project created through the dashboard

## Setup Steps

### 1. Environment Variables

Set up the following environment variables:

```bash
export VERCEL_TOKEN=your_vercel_token
export VERCEL_ORG_ID=your_org_id
export VERCEL_PROJECT_ID=your_project_id
```

You can find these values in your Vercel dashboard settings.

### 2. Deployment

Run the provided deployment script:

```bash
python deploy_to_vercel.py
```

The script will:
- Check prerequisites
- Prepare API files
- Deploy to Vercel

### 3. Manual Deployment (alternative)

If you prefer manual deployment:

```bash
# Login to Vercel
vercel login

# Deploy to Vercel
vercel --prod
```

## Project Structure for Vercel

The project is set up with the following structure for Vercel deployment:

```
├── api/                   # Python FastAPI backend
│   ├── index.py           # Main API file (entry point)
│   └── requirements.txt   # API dependencies
├── frontend/              # Next.js frontend
│   ├── pages/             # React pages
│   ├── public/            # Static files
│   ├── styles/            # CSS styles
│   └── package.json       # Frontend dependencies
├── vercel.json            # Vercel configuration
└── deploy_to_vercel.py    # Deployment script
```

## Frontend-Backend Connection

The frontend and backend are configured to connect properly in production:

1. Frontend requests to `/api/*` are routed to the Python API
2. CORS is configured to allow requests between frontend and API
3. Environment variables are used to store API URLs and other configuration

## Troubleshooting

### API Dependencies

If the API fails to deploy due to dependency issues:

1. Check `api/requirements.txt` to ensure all dependencies are compatible
2. Try removing any problematic dependencies
3. For packages like `detectron2`, consider using alternative packages or pre-trained models

### Frontend Build Errors

If the frontend build fails:

1. Test locally with `npm run build` inside the frontend directory
2. Check for any TypeScript errors
3. Ensure all dependencies are compatible with Vercel

### Deployment Timeouts

If deployment times out:

1. Consider splitting the deployment into multiple projects
2. Optimize build steps in `package.json`
3. Remove unnecessary large files or dependencies

## Environment-Specific Configuration

To use different settings for development and production:

1. Add environment variables in the Vercel dashboard
2. Reference these variables in your code
3. Use `.env.local` for local development

## Regular Updates

Keep your deployment up to date:

1. Commit changes to your repository
2. Run `vercel --prod` to deploy updates
3. Use GitHub integration for automatic deployments