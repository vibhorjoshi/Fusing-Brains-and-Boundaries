import '../styles/globals.css'
import type { Metadata } from 'next'
import { Inter, JetBrains_Mono, Orbitron } from 'next/font/google'
import { Providers } from './providers'

const inter = Inter({ 
  subsets: ['latin'],
  variable: '--font-primary'
})

const jetbrainsMono = JetBrains_Mono({ 
  subsets: ['latin'],
  variable: '--font-mono'
})

const orbitron = Orbitron({ 
  subsets: ['latin'],
  variable: '--font-display'
})

export const metadata: Metadata = {
  title: 'GeoAI - NASA-Level Building Footprint Platform',
  description: 'NASA-Level Professional GeoAI Building Footprint Detection Platform with Advanced Satellite Image Analysis and Machine Learning',
  keywords: 'GeoAI, NASA, Building Footprint Detection, Satellite Imagery, Machine Learning, Computer Vision, Space Technology, AI Research',
  authors: [{ name: 'GeoAI Research Team' }],
  openGraph: {
    title: 'GeoAI - NASA-Level Building Footprint Platform',
    description: 'Advanced AI-powered building footprint extraction using cutting-edge satellite image analysis',
    type: 'website',
  },
  icons: {
    icon: '/favicon.ico',
    apple: '/apple-touch-icon.png',
  },
  viewport: 'width=device-width, initial-scale=1',
  themeColor: '#0B3D91',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="scroll-smooth">
      <body className={`${inter.variable} ${jetbrainsMono.variable} ${orbitron.variable} antialiased`}>
        <Providers>
          <main>
            {children}
          </main>
        </Providers>
      </body>
    </html>
  )
}