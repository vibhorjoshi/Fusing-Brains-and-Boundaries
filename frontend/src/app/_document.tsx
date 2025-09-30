import { Html, Head, Main, NextScript } from 'next/document'

export default function Document() {
  return (
    <Html lang="en">
      <Head>
        {/* Meta Tags */}
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="theme-color" content="#0B3D91" />
        
        {/* SEO Meta Tags */}
        <meta name="description" content="NASA-Level Professional GeoAI Building Footprint Detection Platform with Advanced Satellite Image Analysis" />
        <meta name="keywords" content="GeoAI, Building Footprint Detection, Satellite Imagery, Machine Learning, Computer Vision, NASA, Space Technology" />
        <meta name="author" content="GeoAI Research Team" />
        
        {/* Open Graph / Facebook */}
        <meta property="og:type" content="website" />
        <meta property="og:title" content="GeoAI - NASA-Level Building Footprint Platform" />
        <meta property="og:description" content="Advanced AI-powered building footprint extraction using cutting-edge satellite image analysis" />
        <meta property="og:image" content="/images/geoai-preview.png" />
        
        {/* Twitter */}
        <meta property="twitter:card" content="summary_large_image" />
        <meta property="twitter:title" content="GeoAI - NASA-Level Building Footprint Platform" />
        <meta property="twitter:description" content="Advanced AI-powered building footprint extraction using cutting-edge satellite image analysis" />
        <meta property="twitter:image" content="/images/geoai-preview.png" />
        
        {/* Favicon */}
        <link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png" />
        <link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png" />
        <link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png" />
        <link rel="manifest" href="/site.webmanifest" />
        
        {/* Preload Critical Fonts */}
        <link 
          rel="preload" 
          href="https://fonts.googleapis.com/css2?family=Inter:wght@100;200;300;400;500;600;700;800;900&display=swap" 
          as="style"
        />
        <link 
          rel="preload" 
          href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800&display=swap" 
          as="style"
        />
        <link 
          rel="preload" 
          href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&display=swap" 
          as="style"
        />
        
        {/* NASA-inspired CSS Variables for Theme Consistency */}
        <style jsx>{`
          :root {
            --nasa-mission-start: 1958;
            --current-year: ${new Date().getFullYear()};
            --mission-duration: calc(var(--current-year) - var(--nasa-mission-start));
          }
        `}</style>
      </Head>
      <body className="antialiased">
        {/* Loading Animation */}
        <div id="loading-screen" style={{ 
          position: 'fixed', 
          top: 0, 
          left: 0, 
          width: '100%', 
          height: '100%', 
          background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 25%, #16213e 50%, #0f3460 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 9999,
          color: 'white'
        }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{
              width: '60px',
              height: '60px',
              border: '4px solid #00d4ff',
              borderTop: '4px solid transparent',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite',
              margin: '0 auto 20px'
            }}></div>
            <h2>🚀 INITIALIZING GEOAI SYSTEMS</h2>
            <p style={{ opacity: 0.7 }}>NASA-Level Mission Control Loading...</p>
          </div>
        </div>

        <Main />
        <NextScript />
        
        {/* Hide loading screen after page loads */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              window.addEventListener('load', function() {
                const loadingScreen = document.getElementById('loading-screen');
                if (loadingScreen) {
                  loadingScreen.style.transition = 'opacity 0.5s ease-out';
                  loadingScreen.style.opacity = '0';
                  setTimeout(() => {
                    loadingScreen.remove();
                  }, 500);
                }
              });
              
              // Add keyframe animation for spinner
              const style = document.createElement('style');
              style.textContent = \`
                @keyframes spin {
                  0% { transform: rotate(0deg); }
                  100% { transform: rotate(360deg); }
                }
              \`;
              document.head.appendChild(style);
            `
          }}
        />
      </body>
    </Html>
  )
}