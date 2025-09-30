"use client"

import { useRef, useState, useEffect, Suspense } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Sphere, Text, Points, PointMaterial } from '@react-three/drei'
import * as THREE from 'three'

// Earth Globe Component
function Earth() {
  const earthRef = useRef<THREE.Mesh>(null)
  const [earthTexture, setEarthTexture] = useState<THREE.Texture | null>(null)

  useEffect(() => {
    // Create a simple earth texture using canvas
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')!
    canvas.width = 512
    canvas.height = 256
    
    // Create gradient for earth-like appearance
    const gradient = ctx.createLinearGradient(0, 0, 512, 256)
    gradient.addColorStop(0, '#1a4f8b')
    gradient.addColorStop(0.3, '#2d5aa0')
    gradient.addColorStop(0.7, '#4a90e2')
    gradient.addColorStop(1, '#87ceeb')
    
    ctx.fillStyle = gradient
    ctx.fillRect(0, 0, 512, 256)
    
    // Add some landmass-like patterns
    ctx.fillStyle = '#228b22'
    for (let i = 0; i < 20; i++) {
      const x = Math.random() * 512
      const y = Math.random() * 256
      const w = Math.random() * 80 + 20
      const h = Math.random() * 40 + 10
      ctx.fillRect(x, y, w, h)
    }
    
    const texture = new THREE.CanvasTexture(canvas)
    setEarthTexture(texture)
  }, [])

  useFrame((state) => {
    if (earthRef.current) {
      earthRef.current.rotation.y += 0.002
    }
  })

  return (
    <Sphere ref={earthRef} args={[2, 64, 64]} position={[0, 0, 0]}>
      <meshStandardMaterial 
        map={earthTexture} 
        transparent 
        opacity={0.9}
      />
    </Sphere>
  )
}

// Floating building markers
function BuildingMarkers() {
  const markersRef = useRef<THREE.Group>(null)
  
  const markers = [
    { position: [2.2, 0.5, 1.5], count: 1247, city: "New York" },
    { position: [-1.8, 0.8, 1.2], count: 892, city: "London" },
    { position: [1.5, -1.2, 1.8], count: 2156, city: "Tokyo" },
    { position: [-0.5, 1.8, 1.1], count: 734, city: "Paris" },
    { position: [0.8, -0.9, 2.0], count: 1543, city: "Sydney" },
  ]

  useFrame((state) => {
    if (markersRef.current) {
      markersRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.1
    }
  })

  return (
    <group ref={markersRef}>
      {markers.map((marker, index) => (
        <group key={index} position={marker.position as [number, number, number]}>
          <Sphere args={[0.05, 16, 16]}>
            <meshBasicMaterial color="#ff6b6b" />
          </Sphere>
          <Text
            position={[0, 0.2, 0]}
            fontSize={0.1}
            color="#ffffff"
            anchorX="center"
            anchorY="middle"
          >
            {marker.city}\n{marker.count}
          </Text>
        </group>
      ))}
    </group>
  )
}

// Orbital rings and particles
function OrbitalSystem() {
  const particlesRef = useRef<THREE.Points>(null)
  
  const particlesCount = 1000
  const positions = new Float32Array(particlesCount * 3)
  
  for (let i = 0; i < particlesCount; i++) {
    const radius = 3 + Math.random() * 2
    const theta = Math.random() * Math.PI * 2
    const phi = Math.acos(2 * Math.random() - 1)
    
    positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta)
    positions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta)
    positions[i * 3 + 2] = radius * Math.cos(phi)
  }

  useFrame((state) => {
    if (particlesRef.current) {
      particlesRef.current.rotation.y += 0.001
      particlesRef.current.rotation.z += 0.0005
    }
  })

  return (
    <Points ref={particlesRef} positions={positions} stride={3} frustumCulled={false}>
      <PointMaterial
        transparent
        color="#4fc3f7"
        size={0.02}
        sizeAttenuation={true}
        depthWrite={false}
      />
    </Points>
  )
}

// Main GeoGlobe Component
export default function GeoGlobe() {
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const timer = setTimeout(() => setIsLoading(false), 2000)
    return () => clearTimeout(timer)
  }, [])

  if (isLoading) {
    return (
      <div className="h-96 bg-gray-900 rounded-lg flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-400">Loading 3D Globe...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="h-96 bg-black rounded-lg overflow-hidden">
      <Canvas camera={{ position: [0, 0, 8], fov: 50 }}>
        <Suspense fallback={null}>
          {/* Lighting */}
          <ambientLight intensity={0.3} />
          <pointLight position={[10, 10, 10]} intensity={1} />
          <pointLight position={[-10, -10, -10]} intensity={0.5} color="#4fc3f7" />
          
          {/* 3D Elements */}
          <Earth />
          <BuildingMarkers />
          <OrbitalSystem />
          
          {/* Controls */}
          <OrbitControls 
            enablePan={false}
            enableZoom={true}
            enableRotate={true}
            minDistance={5}
            maxDistance={15}
            autoRotate={true}
            autoRotateSpeed={0.5}
          />
        </Suspense>
      </Canvas>
      
      {/* Overlay Info */}
      <div className="absolute top-4 left-4 bg-black/50 rounded-lg p-3 text-white">
        <h3 className="font-bold mb-1">Global Analysis</h3>
        <p className="text-sm text-gray-300">
          Real-time building detection across major cities
        </p>
      </div>
    </div>
  )
}