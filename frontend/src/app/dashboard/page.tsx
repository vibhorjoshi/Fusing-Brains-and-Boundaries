"use client"

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import dynamic from 'next/dynamic'

// Dynamic imports for 3D components to avoid SSR issues
const GeoGlobe = dynamic(() => import('@/components/3d/GeoGlobe'), { 
  ssr: false,
  loading: () => <div className="h-96 bg-gray-900 rounded-lg animate-pulse" />
})

const BuildingVisualization = dynamic(() => import('@/components/3d/BuildingVisualization'), {
  ssr: false,
  loading: () => <div className="h-64 bg-gray-900 rounded-lg animate-pulse" />
})

export default function Dashboard() {
  const [user, setUser] = useState<any>(null)
  const [stats, setStats] = useState({
    totalBuildings: 0,
    activeJobs: 0,
    totalProcessed: 0,
    apiCalls: 0
  })
  const [loading, setLoading] = useState(true)
  const router = useRouter()

  useEffect(() => {
    // Check authentication
    const token = localStorage.getItem('geoai_token')
    const userData = localStorage.getItem('geoai_user')
    
    if (!token || !userData) {
      router.push('/login')
      return
    }
    
    setUser(JSON.parse(userData))
    loadDashboardData()
  }, [router])

  const loadDashboardData = async () => {
    try {
      // Simulate API calls - replace with actual API when dependencies are installed
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      setStats({
        totalBuildings: 1247,
        activeJobs: 3,
        totalProcessed: 8936,
        apiCalls: 15420
      })
    } catch (error) {
      console.error('Failed to load dashboard data:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
        <div className="container mx-auto px-4 py-8">
          <div className="animate-pulse">
            <div className="h-8 bg-gray-700 rounded mb-6 w-64"></div>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
              {[1,2,3,4].map(i => (
                <div key={i} className="h-24 bg-gray-700 rounded-lg"></div>
              ))}
            </div>
            <div className="h-96 bg-gray-700 rounded-lg mb-8"></div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Animated Background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="neural-network-bg opacity-10"></div>
      </div>
      
      <div className="relative z-10 container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-white mb-2">
                Welcome back, {user?.username}
              </h1>
              <p className="text-gray-300">
                GeoAI Building Footprint Analysis Dashboard
              </p>
            </div>
            <button
              onClick={() => router.push('/dashboard/unified')}
              className="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white font-semibold rounded-lg hover:from-blue-600 hover:to-purple-700 transition-all"
            >
              🚀 Unified Platform
            </button>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="glass-morphism p-6 rounded-xl border border-white/10">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-300 text-sm">Total Buildings</p>
                <p className="text-3xl font-bold text-white">{stats.totalBuildings.toLocaleString()}</p>
              </div>
              <div className="h-12 w-12 bg-blue-500/20 rounded-lg flex items-center justify-center">
                🏢
              </div>
            </div>
          </div>

          <div className="glass-morphism p-6 rounded-xl border border-white/10">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-300 text-sm">Active Jobs</p>
                <p className="text-3xl font-bold text-white">{stats.activeJobs}</p>
              </div>
              <div className="h-12 w-12 bg-green-500/20 rounded-lg flex items-center justify-center">
                ⚡
              </div>
            </div>
          </div>

          <div className="glass-morphism p-6 rounded-xl border border-white/10">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-300 text-sm">Total Processed</p>
                <p className="text-3xl font-bold text-white">{stats.totalProcessed.toLocaleString()}</p>
              </div>
              <div className="h-12 w-12 bg-purple-500/20 rounded-lg flex items-center justify-center">
                🔄
              </div>
            </div>
          </div>

          <div className="glass-morphism p-6 rounded-xl border border-white/10">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-300 text-sm">API Calls</p>
                <p className="text-3xl font-bold text-white">{stats.apiCalls.toLocaleString()}</p>
              </div>
              <div className="h-12 w-12 bg-orange-500/20 rounded-lg flex items-center justify-center">
                🚀
              </div>
            </div>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* 3D Globe */}
          <div className="glass-morphism p-6 rounded-xl border border-white/10">
            <h2 className="text-2xl font-bold text-white mb-4">Global Building Analysis</h2>
            <GeoGlobe />
          </div>

          {/* Building Visualization */}
          <div className="glass-morphism p-6 rounded-xl border border-white/10">
            <h2 className="text-2xl font-bold text-white mb-4">3D Building Models</h2>
            <BuildingVisualization />
          </div>
        </div>

        {/* Action Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
          <div 
            className="glass-morphism p-6 rounded-xl border border-white/10 cursor-pointer hover:bg-white/5 transition-all duration-300 group"
            onClick={() => router.push('/upload')}
          >
            <div className="text-center">
              <div className="h-16 w-16 bg-blue-500/20 rounded-lg flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform">
                📤
              </div>
              <h3 className="text-xl font-bold text-white mb-2">Upload Images</h3>
              <p className="text-gray-300">Process satellite imagery for building detection</p>
            </div>
          </div>

          <div 
            className="glass-morphism p-6 rounded-xl border border-white/10 cursor-pointer hover:bg-white/5 transition-all duration-300 group"
            onClick={() => router.push('/buildings')}
          >
            <div className="text-center">
              <div className="h-16 w-16 bg-green-500/20 rounded-lg flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform">
                🏗️
              </div>
              <h3 className="text-xl font-bold text-white mb-2">View Buildings</h3>
              <p className="text-gray-300">Browse and manage detected building footprints</p>
            </div>
          </div>

          <div 
            className="glass-morphism p-6 rounded-xl border border-white/10 cursor-pointer hover:bg-white/5 transition-all duration-300 group"
            onClick={() => router.push('/jobs')}
          >
            <div className="text-center">
              <div className="h-16 w-16 bg-purple-500/20 rounded-lg flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform">
                📊
              </div>
              <h3 className="text-xl font-bold text-white mb-2">Job Status</h3>
              <p className="text-gray-300">Monitor processing jobs and results</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}