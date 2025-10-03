# Vercel Deployment Instructions for GeoAI Project

Since we're encountering issues with the Vercel CLI deployment, we'll deploy using Vercel's GitHub integration which provides better reliability and automatic deployments on pushes to your repository.

## Step 1: Prepare Your Repository

Make sure your project is pushed to GitHub. Your configuration files (`vercel.json`) have already been updated with the correct settings.

## Step 2: Connect to Vercel

1. Go to [Vercel's website](https://vercel.com) and sign in with your GitHub account
2. Click on "Add New..." and select "Project"
3. Select the GitHub repository `Fusing-Brains-and-Boundaries`
4. Configure the project settings:
   - Framework Preset: Next.js
   - Root Directory: `frontend`
   - Build Command: `npm run build`
   - Output Directory: `.next`
   - Install Command: `npm install`

## Step 3: Configure Environment Variables

Add the following environment variables to your Vercel project:
```
NEXT_PUBLIC_API_URL=https://geoai-api.vercel.app
NEXT_PUBLIC_GEOAI_ENGINE_URL=https://geoai-engine.vercel.app
NEXT_PUBLIC_DASHBOARD_URL=https://geoai-dashboard.vercel.app
```

## Step 4: Deploy

Click "Deploy" and Vercel will build and deploy your project. The deployment process should take a few minutes.

## Step 5: Verify Deployment

Once deployed, Vercel will provide you with a URL to access your site (e.g., `https://fusing-brains-and-boundaries.vercel.app`).

## Set Up Automatic Deployments

With GitHub integration enabled, Vercel will automatically deploy your project whenever you push changes to the main branch of your repository.

## Custom Domain Setup (Optional)

To use a custom domain with your Vercel deployment:

1. Go to your project on Vercel
2. Navigate to "Settings" > "Domains"
3. Add your domain and follow the DNS configuration instructions

## Troubleshooting

If you encounter any issues with the deployment:

1. Check the build logs for errors
2. Verify that all environment variables are correctly set
3. Make sure the `vercel.json` configuration is valid
4. Ensure that the Next.js project is properly configured for production builds