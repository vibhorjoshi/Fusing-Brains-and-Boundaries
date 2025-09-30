"use client"

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'

export default function UnifiedDashboard() {
  const [activeView, setActiveView] = useState('overview')
  const [isConnected, setIsConnected] = useState(false)
  const [stats, setStats] = useState({
    activeJobs: 0,
    totalBuildings: 0,
    processingSpeed: 0,
    accuracy: 0
  })

  useEffect(() => {
    // Connect to WebSocket for real-time updates
    const ws = new WebSocket('ws://127.0.0.1:8002/ws')
    
    ws.onopen = () => {
      setIsConnected(true)
      console.log('Connected to GeoAI backend')
    }
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.type === 'stats_update') {
        setStats(data.stats)
      }
    }
    
    ws.onclose = () => {
      setIsConnected(false)
    }
    
    return () => ws.close()
  }, [])

  const navigationItems = [
    { id: 'overview', label: '🏠 Overview', description: 'System status and metrics' },
    { id: 'live', label: '🌟 Live Processing', description: '3D building visualization' },
    { id: 'globe', label: '🌍 Global View', description: 'Geographic visualization' },
    { id: 'ml', label: '🧠 ML Pipeline', description: 'Neural network processing' },
    { id: 'analytics', label: '📊 Analytics', description: 'Performance dashboard' }
  ]

  const renderVisualization = () => {
    const baseUrl = 'http://127.0.0.1:8002'
    
    switch (activeView) {
      case 'live':
        return (
          <iframe
            src={`${baseUrl}/live`}
            className="w-full h-full border-0 rounded-lg"
            title="Live 3D Visualization"
          />
        )
      case 'globe':
        return (
          <iframe
            src={`${baseUrl}/globe`}
            className="w-full h-full border-0 rounded-lg"
            title="Globe Visualization"
          />
        )
      case 'ml':
        return (
          <iframe
            src={`${baseUrl}/ml-processing`}
            className="w-full h-full border-0 rounded-lg"
            title="ML Processing Pipeline"
          />
        )
      case 'analytics':
        return (
          <iframe
            src={`${baseUrl}/analytics`}
            className="w-full h-full border-0 rounded-lg"
            title="Analytics Dashboard"
          />
        )
      default:
        return <OverviewDashboard stats={stats} isConnected={isConnected} />
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
      <div className="flex h-screen">
        {/* Sidebar Navigation */}
        <motion.div
          initial={{ x: -300 }}
          animate={{ x: 0 }}
          className="w-80 bg-black/20 backdrop-blur-xl border-r border-white/10 p-6"
        >
          <div className="mb-8">
            <div className="flex items-center space-x-3 mb-2">
              <div className="h-12 w-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
                <span className="text-2xl">🌍</span>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">GeoAI Platform</h1>
                <p className="text-sm text-gray-400">Building Footprint AI</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-2 mt-4">
              <div className={`h-3 w-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm text-gray-300">
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>

          <nav className="space-y-3">
            {navigationItems.map((item) => (
              <motion.button
                key={item.id}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setActiveView(item.id)}
                className={`w-full text-left p-4 rounded-xl transition-all ${
                  activeView === item.id
                    ? 'bg-gradient-to-r from-blue-500/20 to-purple-500/20 border border-blue-500/30'
                    : 'bg-white/5 hover:bg-white/10 border border-transparent'
                }`}
              >
                <div className="text-lg font-semibold text-white mb-1">
                  {item.label}
                </div>
                <div className="text-sm text-gray-400">
                  {item.description}
                </div>
              </motion.button>
            ))}
          </nav>

          {/* Quick Stats */}
          <div className="mt-8 space-y-4">
            <h3 className="text-lg font-semibold text-white mb-4">Quick Stats</h3>
            
            <div className="bg-white/5 rounded-lg p-4">
              <div className="flex justify-between items-center mb-2">
                <span className="text-gray-400">Active Jobs</span>
                <span className="text-blue-400 font-semibold">{stats.activeJobs}</span>
              </div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-gray-400">Buildings Detected</span>
                <span className="text-green-400 font-semibold">{stats.totalBuildings}</span>
              </div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-gray-400">Processing Speed</span>
                <span className="text-purple-400 font-semibold">{stats.processingSpeed.toFixed(1)}ms</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Accuracy</span>
                <span className="text-cyan-400 font-semibold">{stats.accuracy.toFixed(1)}%</span>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Main Content */}
        <div className="flex-1 p-6">
          <motion.div
            key={activeView}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="h-full bg-black/10 backdrop-blur-xl rounded-xl border border-white/10 overflow-hidden"
          >
            {renderVisualization()}
          </motion.div>
        </div>
      </div>
    </div>
  )
}

function OverviewDashboard({ stats, isConnected }: { stats: any, isConnected: boolean }) {
  return (
    <div className="p-8 h-full overflow-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-6xl mx-auto"
      >
        <div className="mb-8">
          <h2 className="text-4xl font-bold text-white mb-2">
            Welcome to GeoAI Platform
          </h2>
          <p className="text-xl text-gray-300">
            Advanced Building Footprint Detection using Deep Learning
          </p>
        </div>

        {/* Feature Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <FeatureCard
            icon="🌟"
            title="Live Processing"
            description="Real-time 3D building detection and visualization with interactive controls"
            color="from-blue-500 to-cyan-500"
          />
          <FeatureCard
            icon="🌍"
            title="Global View"
            description="Geographic visualization with interactive 3D globe and city markers"
            color="from-green-500 to-emerald-500"
          />
          <FeatureCard
            icon="🧠"
            title="ML Pipeline"
            description="Neural network visualization with real-time processing stages"
            color="from-purple-500 to-pink-500"
          />
          <FeatureCard
            icon="📊"
            title="Analytics"
            description="Comprehensive performance monitoring and system metrics"
            color="from-orange-500 to-red-500"
          />
        </div>

        {/* System Status */}
        <div className="bg-white/5 rounded-xl p-6 border border-white/10">
          <h3 className="text-2xl font-bold text-white mb-6">System Status</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <StatusCard
              title="Backend Service"
              status={isConnected ? 'Online' : 'Offline'}
              color={isConnected ? 'green' : 'red'}
              details="FastAPI + WebSocket"
            />
            <StatusCard
              title="ML Models"
              status="Ready"
              color="blue"
              details="Mask R-CNN, Adaptive Fusion"
            />
            <StatusCard
              title="Database"
              status="Connected"
              color="purple"
              details="Redis Cache"
            />
          </div>
        </div>

        {/* Getting Started */}
        <div className="mt-8 bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-xl p-6 border border-blue-500/20">
          <h3 className="text-xl font-bold text-white mb-4">🚀 Getting Started</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-gray-300">
            <div>
              <p className="mb-2">• Navigate using the sidebar to explore different visualizations</p>
              <p className="mb-2">• Upload satellite images for building detection</p>
              <p>• Monitor real-time processing and results</p>
            </div>
            <div>
              <p className="mb-2">• View global building data on the interactive globe</p>
              <p className="mb-2">• Analyze ML model performance and accuracy</p>
              <p>• Export results and generate reports</p>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  )
}

function FeatureCard({ icon, title, description, color }: {
  icon: string
  title: string
  description: string
  color: string
}) {
  return (
    <motion.div
      whileHover={{ scale: 1.05 }}
      className="bg-white/5 rounded-xl p-6 border border-white/10 hover:border-white/20 transition-all"
    >
      <div className={`h-12 w-12 bg-gradient-to-r ${color} rounded-lg flex items-center justify-center mb-4`}>
        <span className="text-2xl">{icon}</span>
      </div>
      <h4 className="text-lg font-semibold text-white mb-2">{title}</h4>
      <p className="text-gray-400 text-sm">{description}</p>
    </motion.div>
  )
}

function StatusCard({ title, status, color, details }: {
  title: string
  status: string
  color: string
  details: string
}) {
  const colorClasses = {
    green: 'text-green-400 bg-green-500/20',
    red: 'text-red-400 bg-red-500/20',
    blue: 'text-blue-400 bg-blue-500/20',
    purple: 'text-purple-400 bg-purple-500/20'
  }

  return (
    <div className="bg-white/5 rounded-lg p-4 border border-white/10">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-white font-semibold">{title}</h4>
        <span className={`px-3 py-1 rounded-full text-xs font-semibold ${colorClasses[color as keyof typeof colorClasses]}`}>
          {status}
        </span>
      </div>
      <p className="text-gray-400 text-sm">{details}</p>
    </div>
  )
}