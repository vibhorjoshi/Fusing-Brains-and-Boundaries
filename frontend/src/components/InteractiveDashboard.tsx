'use client';

import React, { useState, useEffect } from 'react';
import AlabamaMap from './AlabamaMap';
import LivePipelineVisualizer from './LivePipelineVisualizer';

interface MetricData {
  value: number;
  change: number;
  isPositive: boolean;
}

interface SystemStatus {
  api: 'online' | 'offline' | 'processing';
  pipeline: 'online' | 'offline' | 'processing';
  database: 'online' | 'offline' | 'processing';
}

const InteractiveDashboard: React.FC = () => {
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    api: 'online',
    pipeline: 'processing',
    database: 'online'
  });

  const [metrics, setMetrics] = useState({
    totalBuildings: { value: 2847326, change: 12.5, isPositive: true },
    accuracy: { value: 94.2, change: 2.1, isPositive: true },
    processingTime: { value: 3.8, change: -8.3, isPositive: true },
    activeCities: { value: 15, change: 25.0, isPositive: true }
  });

  const [activeView, setActiveView] = useState('overview');
  const [isRealTimeEnabled, setIsRealTimeEnabled] = useState(true);

  useEffect(() => {
    // Simulate real-time updates
    if (isRealTimeEnabled) {
      const interval = setInterval(() => {
        setMetrics(prev => ({
          totalBuildings: {
            ...prev.totalBuildings,
            value: prev.totalBuildings.value + Math.floor(Math.random() * 100)
          },
          accuracy: {
            ...prev.accuracy,
            value: Math.min(100, prev.accuracy.value + (Math.random() - 0.5) * 0.1)
          },
          processingTime: {
            ...prev.processingTime,
            value: Math.max(0.1, prev.processingTime.value + (Math.random() - 0.5) * 0.2)
          },
          activeCities: prev.activeCities
        }));
      }, 5000);

      return () => clearInterval(interval);
    }
  }, [isRealTimeEnabled]);

  const MetricCard: React.FC<{ title: string; data: MetricData; unit?: string; icon: string }> = ({
    title, data, unit = '', icon
  }) => (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 hover:border-blue-500 transition-all duration-200">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-gray-400 text-sm font-medium">{title}</h3>
        <span className="text-2xl">{icon}</span>
      </div>
      <div className="flex items-end justify-between">
        <div>
          <div className="text-3xl font-bold text-white mb-1">
            {data.value.toLocaleString()}{unit}
          </div>
          <div className={`text-sm flex items-center ${
            data.isPositive ? 'text-green-400' : 'text-red-400'
          }`}>
            <span className="mr-1">
              {data.isPositive ? '↗' : '↘'}
            </span>
            {Math.abs(data.change)}% vs last week
          </div>
        </div>
      </div>
    </div>
  );

  const StatusIndicator: React.FC<{ status: string; label: string }> = ({ status, label }) => {
    const getStatusColor = () => {
      switch (status) {
        case 'online': return 'bg-green-500';
        case 'processing': return 'bg-yellow-500';
        case 'offline': return 'bg-red-500';
        default: return 'bg-gray-500';
      }
    };

    return (
      <div className="flex items-center gap-2">
        <div className={`w-3 h-3 rounded-full ${getStatusColor()} animate-pulse`} />
        <span className="text-sm text-gray-300">{label}: {status}</span>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gradient-to-r from-blue-600 to-blue-700 shadow-lg sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="text-2xl">🌍</div>
              <h1 className="text-2xl font-bold">GeoAI Live Dashboard</h1>
              <span className="text-blue-200 text-sm">Building Footprint Detection Platform</span>
            </div>
            
            <div className="flex items-center gap-6">
              <StatusIndicator status={systemStatus.api} label="API" />
              <StatusIndicator status={systemStatus.pipeline} label="Pipeline" />
              <StatusIndicator status={systemStatus.database} label="Database" />
              
              <button
                onClick={() => setIsRealTimeEnabled(!isRealTimeEnabled)}
                className={`px-3 py-1 rounded-full text-xs font-medium ${
                  isRealTimeEnabled 
                    ? 'bg-green-500 text-white' 
                    : 'bg-gray-600 text-gray-300'
                }`}
              >
                {isRealTimeEnabled ? 'LIVE' : 'PAUSED'}
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Navigation Tabs */}
        <div className="flex gap-1 mb-8 bg-gray-800 rounded-lg p-1">
          {['overview', 'pipeline', 'maps', 'analytics'].map((view) => (
            <button
              key={view}
              onClick={() => setActiveView(view)}
              className={`px-6 py-2 rounded-md text-sm font-medium capitalize transition-all ${
                activeView === view
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
            >
              {view}
            </button>
          ))}
        </div>

        {/* Overview Tab */}
        {activeView === 'overview' && (
          <div className="space-y-8">
            {/* Metrics Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <MetricCard 
                title="Total Buildings" 
                data={metrics.totalBuildings} 
                icon="🏢"
              />
              <MetricCard 
                title="Detection Accuracy" 
                data={metrics.accuracy} 
                unit="%" 
                icon="🎯"
              />
              <MetricCard 
                title="Avg Processing Time" 
                data={metrics.processingTime} 
                unit="s" 
                icon="⚡"
              />
              <MetricCard 
                title="Active Cities" 
                data={metrics.activeCities} 
                icon="🌆"
              />
            </div>

            {/* Quick Actions */}
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h2 className="text-xl font-bold mb-4">Quick Actions</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <button 
                  onClick={() => setActiveView('pipeline')}
                  className="p-4 bg-blue-600 hover:bg-blue-700 rounded-lg text-left transition-colors"
                >
                  <div className="text-2xl mb-2">🔄</div>
                  <div className="font-semibold">Start New Processing</div>
                  <div className="text-sm text-blue-200">Process satellite imagery</div>
                </button>
                
                <button 
                  onClick={() => setActiveView('maps')}
                  className="p-4 bg-green-600 hover:bg-green-700 rounded-lg text-left transition-colors"
                >
                  <div className="text-2xl mb-2">🗺️</div>
                  <div className="font-semibold">View City Analysis</div>
                  <div className="text-sm text-green-200">Explore Alabama cities</div>
                </button>
                
                <button 
                  onClick={() => setActiveView('analytics')}
                  className="p-4 bg-purple-600 hover:bg-purple-700 rounded-lg text-left transition-colors"
                >
                  <div className="text-2xl mb-2">📊</div>
                  <div className="font-semibold">View Analytics</div>
                  <div className="text-sm text-purple-200">Performance insights</div>
                </button>
              </div>
            </div>

            {/* Recent Activity */}
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h2 className="text-xl font-bold mb-4">Recent Activity</h2>
              <div className="space-y-3">
                {[
                  { time: '2 min ago', action: 'Birmingham processing completed', status: 'success' },
                  { time: '5 min ago', action: 'Montgomery analysis started', status: 'processing' },
                  { time: '12 min ago', action: 'Model accuracy improved to 94.2%', status: 'success' },
                  { time: '18 min ago', action: 'New satellite data received', status: 'info' },
                ].map((activity, index) => (
                  <div key={index} className="flex items-center justify-between py-2">
                    <div className="flex items-center gap-3">
                      <div className={`w-2 h-2 rounded-full ${
                        activity.status === 'success' ? 'bg-green-500' :
                        activity.status === 'processing' ? 'bg-yellow-500' :
                        'bg-blue-500'
                      }`} />
                      <span className="text-gray-300">{activity.action}</span>
                    </div>
                    <span className="text-sm text-gray-500">{activity.time}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Pipeline Tab */}
        {activeView === 'pipeline' && (
          <div>
            <LivePipelineVisualizer />
          </div>
        )}

        {/* Maps Tab */}
        {activeView === 'maps' && (
          <div>
            <AlabamaMap />
          </div>
        )}

        {/* Analytics Tab */}
        {activeView === 'analytics' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-4">Processing Performance</h3>
              <div className="h-64 bg-gray-700 rounded flex items-center justify-center">
                <span className="text-gray-400">Chart: Processing Time Trends</span>
              </div>
            </div>
            
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-4">Accuracy Distribution</h3>
              <div className="h-64 bg-gray-700 rounded flex items-center justify-center">
                <span className="text-gray-400">Chart: Accuracy by City</span>
              </div>
            </div>
            
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-4">Resource Utilization</h3>
              <div className="h-64 bg-gray-700 rounded flex items-center justify-center">
                <span className="text-gray-400">Chart: CPU/GPU Usage</span>
              </div>
            </div>
            
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-4">Detection Statistics</h3>
              <div className="space-y-4">
                <div className="flex justify-between">
                  <span className="text-gray-400">Buildings per km²</span>
                  <span className="text-white font-medium">1,247</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Avg Building Size</span>
                  <span className="text-white font-medium">184 m²</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Processing Speed</span>
                  <span className="text-white font-medium">2.3 km²/min</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Error Rate</span>
                  <span className="text-white font-medium">5.8%</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default InteractiveDashboard;