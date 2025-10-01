"use client"

import { useRef, useState, useMemo, Suspense, useEffect } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Box, Cylinder, Plane, Text } from '@react-three/drei'
import * as THREE from 'three'

// Types for API integration
interface Building3D {
  id: string
  coordinates: number[][]
  height: number
  building_type: string
  confidence: number
  area_m2: number
}

interface CityData {
  name: string
  lat: number
  lng: number
  buildings: number
  accuracy: number
  processing_time: number
  performance_score: number
  building_density: number
  improvement_rate: number
  population?: number
  area_km2?: number
}

interface ProcessedBuilding {
  id: string
  position: [number, number, number]
  height: number
  width: number
  depth: number
  color: string
  confidence: number
  type: string
}

// API Client Hook
function useGeoAIAPI() {
  const [cities, setCities] = useState<CityData[]>([])
  const [buildings, setBuildings] = useState<Building3D[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const API_BASE = 'http://localhost:8000'

  const fetchCities = async () => {
    try {
      setLoading(true)
      const response = await fetch(`${API_BASE}/api/cities`)
      if (!response.ok) throw new Error('Failed to fetch cities')
      const data = await response.json()
      setCities(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  const fetch3DBuildings = async (cityName: string, limit: number = 100) => {
    try {
      setLoading(true)
      const response = await fetch(`${API_BASE}/api/3d/buildings/${cityName}?limit=${limit}`)
      if (!response.ok) throw new Error(`Failed to fetch 3D buildings for ${cityName}`)
      const data = await response.json()
      setBuildings(data.buildings)
      return data
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
      return null
    } finally {
      setLoading(false)
    }
  }

  const startProcessing = async (cityName: string) => {
    try {
      const response = await fetch(`${API_BASE}/api/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          city_name: cityName,
          processing_type: 'full_pipeline',
          options: {}
        })
      })
      if (!response.ok) throw new Error(`Failed to start processing for ${cityName}`)
      return await response.json()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
      return null
    }
  }

  return {
    cities,
    buildings,
    loading,
    error,
    fetchCities,
    fetch3DBuildings,
    startProcessing
  }
}

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
function NeuralConnections({ buildings }: { buildings: ProcessedBuilding[] }) {
  const linesRef = useRef<THREE.Group>(null)

  const connections = useMemo(() => {
    const lines = []
    for (let i = 0; i < buildings.length - 1; i++) {
      const start = buildings[i].position
      const end = buildings[i + 1].position
      
      const points = [
        new THREE.Vector3(start[0], start[1], start[2]),
        new THREE.Vector3(end[0], end[1], end[2])
      ]
      
      lines.push(
        <line key={`connection-${i}`}>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={points.length}
              array={new Float32Array(points.flatMap(p => [p.x, p.y, p.z]))}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color="#4ade80" transparent opacity={0.3} />
        </line>
      )
    }
    return lines
  }, [buildings])

  return <group ref={linesRef}>{connections}</group>
}

// Main Building Visualization Component with FastAPI Integration
export default function BuildingVisualization() {
  const [selectedBuilding, setSelectedBuilding] = useState<string | null>(null)
  const [selectedCity, setSelectedCity] = useState<string>('Birmingham')
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingProgress, setProcessingProgress] = useState(0)
  const [websocket, setWebsocket] = useState<WebSocket | null>(null)
  
  const { 
    cities, 
    buildings, 
    loading, 
    error, 
    fetchCities, 
    fetch3DBuildings, 
    startProcessing: apiStartProcessing
  } = useGeoAIAPI()

  // Initialize data and WebSocket
  useEffect(() => {
    fetchCities()
    
    // Setup WebSocket for real-time updates
    const ws = new WebSocket('ws://localhost:8000/ws')
    
    ws.onopen = () => {
      console.log('✅ WebSocket connected')
      setWebsocket(ws)
    }
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      if (data.type === 'job_progress') {
        setProcessingProgress(data.progress)
      } else if (data.type === 'job_completed') {
        setIsProcessing(false)
        setProcessingProgress(100)
        // Refresh buildings data
        fetch3DBuildings(selectedCity)
      }
    }
    
    ws.onclose = () => {
      console.log('❌ WebSocket disconnected')
      setWebsocket(null)
    }
    
    return () => {
      ws.close()
    }
  }, [])

  // Load 3D buildings when city changes
  useEffect(() => {
    if (selectedCity) {
      fetch3DBuildings(selectedCity, 50)
    }
  }, [selectedCity])

  // Convert API buildings to 3D format
  const processedBuildings: ProcessedBuilding[] = useMemo(() => {
    return buildings.map((building, index) => {
      const coords = building.coordinates[0] || [0, 0, 0]
      // Convert to relative 3D positions for demo
      const x = Math.random() * 10 - 5 // Spread buildings randomly for demo
      const z = Math.random() * 10 - 5
      const y = Math.max(0.2, building.height / 10) // Scale height

      return {
        id: building.id,
        position: [x, y / 2, z] as [number, number, number],
        height: y,
        width: Math.sqrt(building.area_m2) / 50 || 0.8,
        depth: Math.sqrt(building.area_m2) / 50 || 0.6,
        color: building.building_type === 'residential' ? '#4ade80' :
               building.building_type === 'commercial' ? '#3b82f6' :
               building.building_type === 'industrial' ? '#8b5cf6' : '#f59e0b',
        confidence: building.confidence,
        type: building.building_type
      }
    })
  }, [buildings])

  const handleBuildingClick = (buildingId: string) => {
    setSelectedBuilding(buildingId)
  }

  const handleStartProcessing = async () => {
    if (selectedCity) {
      setIsProcessing(true)
      setProcessingProgress(0)
      await apiStartProcessing(selectedCity)
    }
  }

  const avgConfidence = processedBuildings.length > 0 
    ? processedBuildings.reduce((acc, b) => acc + b.confidence, 0) / processedBuildings.length
    : 0

  return (
    <div className="h-64 bg-gray-900 rounded-lg overflow-hidden relative">
      <Canvas camera={{ position: [5, 5, 5] }}>
        <Suspense fallback={null}>
          {/* Lighting */}
          <ambientLight intensity={0.4} />
          <directionalLight 
            position={[10, 10, 5]} 
            intensity={1}
            castShadow
          />
          <pointLight position={[0, 10, 0]} intensity={0.5} />

          {/* Environment */}
          <GroundGrid />
          
          {/* Buildings from API */}
          {processedBuildings.map((building) => (
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

          {/* Neural Network Style Connections */}
          <NeuralConnections buildings={processedBuildings} />
          
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
          {/* City Selector */}
          <div>
            <label className="block text-xs text-gray-300 mb-1">City:</label>
            <select 
              value={selectedCity} 
              onChange={(e) => setSelectedCity(e.target.value)}
              className="w-full px-2 py-1 bg-gray-800 border border-gray-600 rounded text-xs"
            >
              {cities.map(city => (
                <option key={city.name} value={city.name}>{city.name}</option>
              ))}
            </select>
          </div>
          
          <div className="flex items-center justify-between">
            <span>Buildings:</span>
            <span className="text-blue-400">{processedBuildings.length}</span>
          </div>
          
          <div className="flex items-center justify-between">
            <span>Avg Confidence:</span>
            <span className="text-green-400">
              {Math.round(avgConfidence * 100)}%
            </span>
          </div>
          
          {isProcessing && (
            <div className="w-full">
              <div className="flex items-center justify-between text-xs mb-1">
                <span>Processing:</span>
                <span>{Math.round(processingProgress)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-1">
                <div 
                  className="bg-blue-500 h-1 rounded-full transition-all duration-300"
                  style={{ width: `${processingProgress}%` }}
                />
              </div>
            </div>
          )}
          
          <button
            onClick={handleStartProcessing}
            disabled={isProcessing || loading}
            className="w-full px-3 py-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded text-xs transition-colors"
          >
            {isProcessing ? 'Processing...' : 
             loading ? 'Loading...' : 
             'Run 3D Analysis'}
          </button>
          
          {error && (
            <div className="text-red-400 text-xs">{error}</div>
          )}
        </div>
      </div>
      
      {/* Selected Building Info */}
      {selectedBuilding && (
        <div className="absolute bottom-4 left-4 bg-black/70 rounded-lg p-3 text-white">
          {(() => {
            const building = processedBuildings.find(b => b.id === selectedBuilding)
            return building ? (
              <div className="text-sm">
                <h4 className="font-bold text-blue-400 mb-1">Building #{building.id.slice(-4)}</h4>
                <div>Type: {building.type}</div>
                <div>Confidence: {Math.round(building.confidence * 100)}%</div>
                <div>Dimensions: {building.width.toFixed(1)}×{building.depth.toFixed(1)}×{building.height.toFixed(1)}</div>
              </div>
            ) : null
          })()}
        </div>
      )}
      
      {/* Loading State */}
      {loading && (
        <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
          <div className="text-white text-sm">Loading 3D data...</div>
        </div>
      )}
    </div>
  )
}