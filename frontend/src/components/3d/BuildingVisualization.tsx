"use client"

import { useRef, useState, useMemo, Suspense } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Box, Cylinder, Plane, Text } from '@react-three/drei'
import * as THREE from 'three'

// Individual Building Component
function Building({ 
  position, 
  height, 
  width, 
  depth, 
  color, 
  confidence,
  onClick 
}: {
  position: [number, number, number]
  height: number
  width: number
  depth: number
  color: string
  confidence: number
  onClick?: () => void
}) {
  const buildingRef = useRef<THREE.Mesh>(null)
  const [hovered, setHovered] = useState(false)

  useFrame((state) => {
    if (buildingRef.current) {
      const targetY = hovered ? position[1] + 0.1 : position[1]
      buildingRef.current.position.y = THREE.MathUtils.lerp(
        buildingRef.current.position.y,
        targetY,
        0.1
      )
    }
  })

  return (
    <group>
      {/* Building Base */}
      <Box
        ref={buildingRef}
        position={position}
        args={[width, height, depth]}
        onClick={onClick}
        onPointerEnter={() => setHovered(true)}
        onPointerLeave={() => setHovered(false)}
      >
        <meshStandardMaterial 
          color={hovered ? '#ff6b6b' : color}
          transparent
          opacity={0.8 + confidence * 0.2}
        />
      </Box>
      
      {/* Confidence Indicator */}
      <Cylinder
        position={[position[0], position[1] + height/2 + 0.2, position[2]]}
        args={[0.05, 0.05, confidence * 0.5, 8]}
      >
        <meshBasicMaterial 
          color={confidence > 0.8 ? '#4ade80' : confidence > 0.6 ? '#fbbf24' : '#f87171'}
        />
      </Cylinder>
      
      {/* Label */}
      {hovered && (
        <Text
          position={[position[0], position[1] + height/2 + 0.8, position[2]]}
          fontSize={0.2}
          color="#ffffff"
          anchorX="center"
          anchorY="middle"
        >
          {`${Math.round(confidence * 100)}% confident`}
        </Text>
      )}
    </group>
  )
}

// Ground Grid Component
function GroundGrid() {
  const gridSize = 20
  const divisions = 40

  return (
    <group>
      <Plane position={[0, -0.1, 0]} rotation={[-Math.PI / 2, 0, 0]} args={[gridSize, gridSize]}>
        <meshBasicMaterial color="#1a1a1a" transparent opacity={0.3} />
      </Plane>
      <gridHelper args={[gridSize, divisions, '#333333', '#222222']} />
    </group>
  )
}

// Neural Network Connections
function NeuralConnections({ buildings }: { buildings: any[] }) {
  const linesRef = useRef<THREE.Group>(null)

  const connections = useMemo(() => {
    const lines = []
    for (let i = 0; i < buildings.length - 1; i++) {
      const start = buildings[i].position
      const end = buildings[i + 1].position
      
      const points = []
      points.push(new THREE.Vector3(start[0], start[1] + buildings[i].height/2, start[2]))
      
      // Add curve point
      const midX = (start[0] + end[0]) / 2
      const midY = Math.max(start[1] + buildings[i].height/2, end[1] + buildings[i+1].height/2) + 1
      const midZ = (start[2] + end[2]) / 2
      points.push(new THREE.Vector3(midX, midY, midZ))
      
      points.push(new THREE.Vector3(end[0], end[1] + buildings[i+1].height/2, end[2]))
      
      lines.push(points)
    }
    return lines
  }, [buildings])

  useFrame((state) => {
    if (linesRef.current) {
      linesRef.current.children.forEach((child, index) => {
        const material = (child as any).material
        if (material) {
          material.opacity = 0.3 + Math.sin(state.clock.elapsedTime * 2 + index) * 0.2
        }
      })
    }
  })

  return (
    <group ref={linesRef}>
      {connections.map((points, index) => {
        const curve = new THREE.CatmullRomCurve3(points)
        const geometry = new THREE.TubeGeometry(curve, 20, 0.02, 8, false)
        
        return (
          <mesh key={index} geometry={geometry}>
            <meshBasicMaterial 
              color="#4fc3f7" 
              transparent 
              opacity={0.4}
            />
          </mesh>
        )
      })}
    </group>
  )
}

// Processing Effect
function ProcessingEffect() {
  const particlesRef = useRef<THREE.Points>(null)
  
  const particleCount = 50
  const positions = new Float32Array(particleCount * 3)
  const colors = new Float32Array(particleCount * 3)
  
  for (let i = 0; i < particleCount; i++) {
    positions[i * 3] = (Math.random() - 0.5) * 10
    positions[i * 3 + 1] = Math.random() * 5
    positions[i * 3 + 2] = (Math.random() - 0.5) * 10
    
    colors[i * 3] = Math.random()
    colors[i * 3 + 1] = Math.random() * 0.5 + 0.5
    colors[i * 3 + 2] = 1
  }

  useFrame((state) => {
    if (particlesRef.current) {
      const positions = particlesRef.current.geometry.attributes.position.array as Float32Array
      
      for (let i = 0; i < particleCount; i++) {
        positions[i * 3 + 1] += Math.sin(state.clock.elapsedTime * 2 + i) * 0.01
      }
      
      particlesRef.current.geometry.attributes.position.needsUpdate = true
      particlesRef.current.rotation.y += 0.005
    }
  })

  return (
    <points ref={particlesRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={particleCount}
          array={positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={particleCount}
          array={colors}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial size={0.1} vertexColors transparent opacity={0.6} />
    </points>
  )
}

// Main Building Visualization Component
export default function BuildingVisualization() {
  const [selectedBuilding, setSelectedBuilding] = useState<number | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)

  // Mock building data
  const buildings = useMemo(() => [
    {
      id: 1,
      position: [-2, 0.5, -2] as [number, number, number],
      height: 1,
      width: 0.8,
      depth: 0.6,
      color: '#4ade80',
      confidence: 0.95,
      type: 'Residential'
    },
    {
      id: 2,
      position: [0, 0.75, -1] as [number, number, number],
      height: 1.5,
      width: 1,
      depth: 0.8,
      color: '#3b82f6',
      confidence: 0.87,
      type: 'Commercial'
    },
    {
      id: 3,
      position: [2, 1, 0] as [number, number, number],
      height: 2,
      width: 1.2,
      depth: 1,
      color: '#8b5cf6',
      confidence: 0.92,
      type: 'Industrial'
    },
    {
      id: 4,
      position: [-1, 0.4, 1] as [number, number, number],
      height: 0.8,
      width: 0.6,
      depth: 0.4,
      color: '#f59e0b',
      confidence: 0.78,
      type: 'Residential'
    },
    {
      id: 5,
      position: [1, 0.6, 2] as [number, number, number],
      height: 1.2,
      width: 0.9,
      depth: 0.7,
      color: '#ef4444',
      confidence: 0.83,
      type: 'Mixed Use'
    }
  ], [])

  const handleBuildingClick = (buildingId: number) => {
    setSelectedBuilding(buildingId)
  }

  const startProcessing = () => {
    setIsProcessing(true)
    setTimeout(() => setIsProcessing(false), 3000)
  }

  return (
    <div className="h-64 bg-gray-900 rounded-lg overflow-hidden relative">
      <Canvas camera={{ position: [5, 5, 5], fov: 60 }}>
        <Suspense fallback={null}>
          {/* Lighting */}
          <ambientLight intensity={0.4} />
          <directionalLight position={[5, 10, 5]} intensity={1} />
          <pointLight position={[-5, 5, -5]} intensity={0.5} color="#4fc3f7" />
          
          {/* Ground and Grid */}
          <GroundGrid />
          
          {/* Buildings */}
          {buildings.map((building) => (
            <Building
              key={building.id}
              position={building.position}
              height={building.height}
              width={building.width}
              depth={building.depth}
              color={building.color}
              confidence={building.confidence}
              onClick={() => handleBuildingClick(building.id)}
            />
          ))}
          
          {/* Neural Network Connections */}
          <NeuralConnections buildings={buildings} />
          
          {/* Processing Effect */}
          {isProcessing && <ProcessingEffect />}
          
          {/* Controls */}
          <OrbitControls 
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            minDistance={3}
            maxDistance={15}
            maxPolarAngle={Math.PI / 2}
          />
        </Suspense>
      </Canvas>
      
      {/* Control Panel */}
      <div className="absolute top-4 right-4 bg-black/70 rounded-lg p-3 text-white">
        <div className="text-sm space-y-2">
          <div className="flex items-center justify-between">
            <span>Buildings:</span>
            <span className="text-blue-400">{buildings.length}</span>
          </div>
          <div className="flex items-center justify-between">
            <span>Avg Confidence:</span>
            <span className="text-green-400">
              {Math.round(buildings.reduce((acc, b) => acc + b.confidence, 0) / buildings.length * 100)}%
            </span>
          </div>
          <button
            onClick={startProcessing}
            disabled={isProcessing}
            className="w-full px-3 py-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded text-xs transition-colors"
          >
            {isProcessing ? 'Processing...' : 'Run Analysis'}
          </button>
        </div>
      </div>
      
      {/* Selected Building Info */}
      {selectedBuilding && (
        <div className="absolute bottom-4 left-4 bg-black/70 rounded-lg p-3 text-white">
          {(() => {
            const building = buildings.find(b => b.id === selectedBuilding)
            return building ? (
              <div className="text-sm">
                <h4 className="font-bold text-blue-400 mb-1">Building #{building.id}</h4>
                <div>Type: {building.type}</div>
                <div>Confidence: {Math.round(building.confidence * 100)}%</div>
                <div>Dimensions: {building.width}×{building.depth}×{building.height}</div>
              </div>
            ) : null
          })()}
        </div>
      )}
    </div>
  )
}