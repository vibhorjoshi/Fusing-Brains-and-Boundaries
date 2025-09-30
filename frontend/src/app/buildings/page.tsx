"use client"

import { useState, useEffect } from 'react'
import { useAuth } from '@/contexts/AuthContext'
import toast from 'react-hot-toast'

interface Building {
  id: string
  geometry: any
  confidence: number
  area: number
  perimeter: number
  created_at: string
  source_image?: string
}

export default function Buildings() {
  const [buildings, setBuildings] = useState<Building[]>([])
  const [loading, setLoading] = useState(true)
  const [stats, setStats] = useState({
    total: 0,
    avgConfidence: 0,
    totalArea: 0
  })
  const [filters, setFilters] = useState({
    minConfidence: 0.5,
    minArea: 0,
    limit: 50,
    offset: 0
  })
  const { token } = useAuth()

  useEffect(() => {
    if (token) {
      loadBuildings()
      loadStatistics()
    }
  }, [token, filters])

  const loadBuildings = async () => {
    try {
      const params = new URLSearchParams({
        limit: filters.limit.toString(),
        offset: filters.offset.toString(),
        min_confidence: filters.minConfidence.toString(),
        min_area: filters.minArea.toString()
      })

      const response = await fetch(`http://127.0.0.1:8002/api/v1/buildings?${params}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      if (!response.ok) {
        throw new Error('Failed to load buildings')
      }

      const data = await response.json()
      setBuildings(data.buildings || [])
    } catch (error: any) {
      toast.error(error.message || 'Failed to load buildings')
      // Mock data for demo
      setBuildings([
        {
          id: '1',
          geometry: { type: 'Polygon', coordinates: [] },
          confidence: 0.95,
          area: 1250.5,
          perimeter: 145.2,
          created_at: '2025-09-26T10:30:00Z',
          source_image: 'satellite_01.jpg'
        },
        {
          id: '2',
          geometry: { type: 'Polygon', coordinates: [] },
          confidence: 0.87,
          area: 890.3,
          perimeter: 120.8,
          created_at: '2025-09-26T11:15:00Z',
          source_image: 'satellite_02.jpg'
        },
        {
          id: '3',
          geometry: { type: 'Polygon', coordinates: [] },
          confidence: 0.92,
          area: 2150.7,
          perimeter: 185.6,
          created_at: '2025-09-26T12:00:00Z',
          source_image: 'satellite_03.jpg'
        }
      ])
    } finally {
      setLoading(false)
    }
  }

  const loadStatistics = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8002/api/v1/buildings/statistics/overview', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      if (!response.ok) {
        throw new Error('Failed to load statistics')
      }

      const data = await response.json()
      setStats(data)
    } catch (error) {
      // Mock stats
      setStats({
        total: 1247,
        avgConfidence: 0.89,
        totalArea: 125740.5
      })
    }
  }

  const deleteBuilding = async (id: string) => {
    if (!confirm('Are you sure you want to delete this building?')) return

    try {
      const response = await fetch(`http://127.0.0.1:8002/api/v1/buildings/${id}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      if (!response.ok) {
        throw new Error('Failed to delete building')
      }

      setBuildings(buildings.filter(b => b.id !== id))
      toast.success('Building deleted successfully')
    } catch (error: any) {
      toast.error(error.message || 'Failed to delete building')
    }
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
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
          <h1 className="text-4xl font-bold text-white mb-2">Building Footprints</h1>
          <p className="text-gray-300">Manage and analyze detected building footprints</p>
        </div>

        {/* Statistics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="glass-morphism p-6 rounded-xl border border-white/10">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-300 text-sm">Total Buildings</p>
                <p className="text-3xl font-bold text-white">{stats.total.toLocaleString()}</p>
              </div>
              <div className="h-12 w-12 bg-blue-500/20 rounded-lg flex items-center justify-center">
                🏢
              </div>
            </div>
          </div>

          <div className="glass-morphism p-6 rounded-xl border border-white/10">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-300 text-sm">Avg Confidence</p>
                <p className="text-3xl font-bold text-white">{Math.round(stats.avgConfidence * 100)}%</p>
              </div>
              <div className="h-12 w-12 bg-green-500/20 rounded-lg flex items-center justify-center">
                🎯
              </div>
            </div>
          </div>

          <div className="glass-morphism p-6 rounded-xl border border-white/10">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-300 text-sm">Total Area</p>
                <p className="text-3xl font-bold text-white">{(stats.totalArea / 1000).toFixed(1)}K</p>
                <p className="text-gray-400 text-xs">sq meters</p>
              </div>
              <div className="h-12 w-12 bg-purple-500/20 rounded-lg flex items-center justify-center">
                📐
              </div>
            </div>
          </div>
        </div>

        {/* Filters */}
        <div className="glass-morphism p-6 rounded-xl border border-white/10 mb-8">
          <h2 className="text-xl font-bold text-white mb-4">Filters</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <label className="block text-gray-300 text-sm mb-1">Min Confidence</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={filters.minConfidence}
                onChange={(e) => setFilters(prev => ({
                  ...prev,
                  minConfidence: parseFloat(e.target.value)
                }))}
                className="w-full"
              />
              <span className="text-gray-400 text-xs">{Math.round(filters.minConfidence * 100)}%</span>
            </div>

            <div>
              <label className="block text-gray-300 text-sm mb-1">Min Area (sq m)</label>
              <input
                type="number"
                value={filters.minArea}
                onChange={(e) => setFilters(prev => ({
                  ...prev,
                  minArea: parseInt(e.target.value) || 0
                }))}
                className="w-full p-2 bg-gray-800 text-white rounded border border-gray-600"
              />
            </div>

            <div>
              <label className="block text-gray-300 text-sm mb-1">Results per page</label>
              <select
                value={filters.limit}
                onChange={(e) => setFilters(prev => ({
                  ...prev,
                  limit: parseInt(e.target.value)
                }))}
                className="w-full p-2 bg-gray-800 text-white rounded border border-gray-600"
              >
                <option value={25}>25</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
              </select>
            </div>

            <div className="flex items-end">
              <button
                onClick={() => setFilters(prev => ({ ...prev, offset: 0 }))}
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
              >
                Apply Filters
              </button>
            </div>
          </div>
        </div>

        {/* Buildings Table */}
        <div className="glass-morphism rounded-xl border border-white/10 overflow-hidden">
          <div className="p-6 border-b border-white/10">
            <h2 className="text-xl font-bold text-white">Detected Buildings</h2>
          </div>

          {loading ? (
            <div className="p-8 text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
              <p className="text-gray-400">Loading buildings...</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead className="bg-gray-800/50">
                  <tr>
                    <th className="p-4 text-gray-300">ID</th>
                    <th className="p-4 text-gray-300">Confidence</th>
                    <th className="p-4 text-gray-300">Area (sq m)</th>
                    <th className="p-4 text-gray-300">Perimeter (m)</th>
                    <th className="p-4 text-gray-300">Source</th>
                    <th className="p-4 text-gray-300">Created</th>
                    <th className="p-4 text-gray-300">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {buildings.map((building) => (
                    <tr key={building.id} className="border-b border-gray-800 hover:bg-gray-800/30">
                      <td className="p-4 text-gray-300">#{building.id}</td>
                      <td className="p-4">
                        <div className="flex items-center">
                          <div className={`h-2 w-16 rounded-full mr-2 ${
                            building.confidence > 0.9 
                              ? 'bg-green-500' 
                              : building.confidence > 0.7 
                              ? 'bg-yellow-500' 
                              : 'bg-red-500'
                          }`}></div>
                          <span className="text-gray-300">{Math.round(building.confidence * 100)}%</span>
                        </div>
                      </td>
                      <td className="p-4 text-gray-300">{building.area.toFixed(1)}</td>
                      <td className="p-4 text-gray-300">{building.perimeter.toFixed(1)}</td>
                      <td className="p-4 text-gray-300">{building.source_image || 'N/A'}</td>
                      <td className="p-4 text-gray-300">{formatDate(building.created_at)}</td>
                      <td className="p-4">
                        <div className="flex space-x-2">
                          <button className="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 transition-colors">
                            View
                          </button>
                          <button 
                            onClick={() => deleteBuilding(building.id)}
                            className="px-3 py-1 bg-red-600 text-white text-sm rounded hover:bg-red-700 transition-colors"
                          >
                            Delete
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Pagination */}
          <div className="p-4 border-t border-white/10 flex items-center justify-between">
            <p className="text-gray-400">
              Showing {filters.offset + 1} to {Math.min(filters.offset + filters.limit, stats.total)} of {stats.total} buildings
            </p>
            <div className="flex space-x-2">
              <button
                onClick={() => setFilters(prev => ({
                  ...prev,
                  offset: Math.max(0, prev.offset - prev.limit)
                }))}
                disabled={filters.offset === 0}
                className="px-3 py-1 bg-gray-700 text-white rounded disabled:opacity-50 hover:bg-gray-600 transition-colors"
              >
                Previous
              </button>
              <button
                onClick={() => setFilters(prev => ({
                  ...prev,
                  offset: prev.offset + prev.limit
                }))}
                disabled={filters.offset + filters.limit >= stats.total}
                className="px-3 py-1 bg-gray-700 text-white rounded disabled:opacity-50 hover:bg-gray-600 transition-colors"
              >
                Next
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}