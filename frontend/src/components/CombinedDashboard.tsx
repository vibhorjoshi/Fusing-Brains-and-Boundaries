'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Line, Bar, Doughnut, Scatter } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';
import Enhanced3DVisualization from './3d/Enhanced3DVisualization';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

declare global {
  interface Window {
    google: any;
    initializeMap?: () => void;
  }
}

interface CityData {
  name: string;
  lat: number;
  lng: number;
  buildings: number;
  accuracy: number;
  processingTime: number;
  performanceScore: number;
  buildingDensity: number;
  improvementRate: number;
}

interface PipelineStep {
  id: string;
  name: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  progress: number;
  processingTime: number;
  memoryUsage: number;
  gpuUsage: number;
}

interface LiveMetrics {
  totalBuildings: number;
  processingSpeed: number;
  accuracy: number;
  systemLoad: number;
  activeProcesses: number;
  queueLength: number;
}

const CombinedDashboard: React.FC = () => {
  const mapRef = useRef<HTMLDivElement>(null);
  const [selectedCity, setSelectedCity] = useState('birmingham');
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(-1);
  const [activeTab, setActiveTab] = useState('overview');
  
  const [liveMetrics, setLiveMetrics] = useState<LiveMetrics>({
    totalBuildings: 2847326,
    processingSpeed: 245.8,
    accuracy: 94.2,
    systemLoad: 67.5,
    activeProcesses: 12,
    queueLength: 8
  });

  // API Base URL
  const API_BASE_URL = 'http://localhost:8000';

  const [steps, setSteps] = useState<PipelineStep[]>([
    { id: 'detection', name: 'Building Detection', status: 'pending', progress: 0, processingTime: 0, memoryUsage: 0, gpuUsage: 0 },
    { id: 'regularization_rt', name: 'RT Regularization', status: 'pending', progress: 0, processingTime: 0, memoryUsage: 0, gpuUsage: 0 },
    { id: 'regularization_rr', name: 'RR Regularization', status: 'pending', progress: 0, processingTime: 0, memoryUsage: 0, gpuUsage: 0 },
    { id: 'regularization_fer', name: 'FER Regularization', status: 'pending', progress: 0, processingTime: 0, memoryUsage: 0, gpuUsage: 0 },
    { id: 'rl_fusion', name: 'RL Adaptive Fusion', status: 'pending', progress: 0, processingTime: 0, memoryUsage: 0, gpuUsage: 0 },
    { id: 'lapnet_refinement', name: 'LapNet Refinement', status: 'pending', progress: 0, processingTime: 0, memoryUsage: 0, gpuUsage: 0 },
    { id: '3d_visualization', name: '3D Visualization', status: 'pending', progress: 0, processingTime: 0, memoryUsage: 0, gpuUsage: 0 }
  ]);

  const [alabamaCities, setAlabamaCities] = useState<CityData[]>([
    { 
      name: "Birmingham", lat: 33.5207, lng: -86.8025, buildings: 156421, accuracy: 91.2, 
      processingTime: 26.4, performanceScore: 8.7, buildingDensity: 1247, improvementRate: 12.5 
    },
    { 
      name: "Montgomery", lat: 32.3792, lng: -86.3077, buildings: 98742, accuracy: 89.7, 
      processingTime: 18.7, performanceScore: 8.3, buildingDensity: 987, improvementRate: 8.9 
    },
    { 
      name: "Mobile", lat: 30.6954, lng: -88.0399, buildings: 87634, accuracy: 88.4, 
      processingTime: 17.2, performanceScore: 8.1, buildingDensity: 876, improvementRate: 7.8 
    },
    { 
      name: "Huntsville", lat: 34.7304, lng: -86.5861, buildings: 124563, accuracy: 92.3, 
      processingTime: 22.9, performanceScore: 9.1, buildingDensity: 1156, improvementRate: 15.2 
    },
    { 
      name: "Tuscaloosa", lat: 33.2098, lng: -87.5692, buildings: 65432, accuracy: 90.1, 
      processingTime: 14.8, performanceScore: 8.5, buildingDensity: 654, improvementRate: 10.3 
    }
  ]);

  // Chart configurations
  const performanceChartData = {
    labels: alabamaCities.map(city => city.name),
    datasets: [
      {
        label: 'Detection Accuracy (%)',
        data: alabamaCities.map(city => city.accuracy),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.4,
      },
      {
        label: 'Performance Score',
        data: alabamaCities.map(city => city.performanceScore),
        borderColor: 'rgb(16, 185, 129)',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        tension: 0.4,
      }
    ]
  };

  const buildingCountData = {
    labels: alabamaCities.map(city => city.name),
    datasets: [
      {
        label: 'Buildings Detected',
        data: alabamaCities.map(city => city.buildings),
        backgroundColor: [
          'rgba(255, 99, 132, 0.8)',
          'rgba(54, 162, 235, 0.8)',
          'rgba(255, 205, 86, 0.8)',
          'rgba(75, 192, 192, 0.8)',
          'rgba(153, 102, 255, 0.8)',
        ],
        borderColor: [
          'rgba(255, 99, 132, 1)',
          'rgba(54, 162, 235, 1)',
          'rgba(255, 205, 86, 1)',
          'rgba(75, 192, 192, 1)',
          'rgba(153, 102, 255, 1)',
        ],
        borderWidth: 2,
      }
    ]
  };

  const processingTimeData = {
    labels: alabamaCities.map(city => city.name),
    datasets: [
      {
        label: 'Processing Time (seconds)',
        data: alabamaCities.map(city => city.processingTime),
        backgroundColor: 'rgba(168, 85, 247, 0.8)',
        borderColor: 'rgba(168, 85, 247, 1)',
        borderWidth: 2,
      }
    ]
  };

  const scatterData = {
    datasets: [
      {
        label: 'Accuracy vs Processing Time',
        data: alabamaCities.map(city => ({
          x: city.processingTime,
          y: city.accuracy,
          cityName: city.name
        })),
        backgroundColor: 'rgba(236, 72, 153, 0.8)',
        borderColor: 'rgba(236, 72, 153, 1)',
        pointRadius: 8,
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          color: '#ffffff',
          font: { size: 12 }
        }
      },
      title: {
        display: true,
        color: '#ffffff',
        font: { size: 14, weight: 'bold' }
      }
    },
    scales: {
      x: {
        ticks: { color: '#ffffff' },
        grid: { color: 'rgba(255, 255, 255, 0.1)' }
      },
      y: {
        ticks: { color: '#ffffff' },
        grid: { color: 'rgba(255, 255, 255, 0.1)' }
      }
    }
  };

  // API Functions
  const fetchCities = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/cities`);
      const data = await response.json();
      
      // Map backend data to frontend interface
      const cities = data.map((city: any) => ({
        name: city.name,
        lat: city.lat,
        lng: city.lng,
        buildings: city.buildings,
        accuracy: city.accuracy,
        processingTime: city.processing_time,
        performanceScore: city.performance_score,
        buildingDensity: city.building_density,
        improvementRate: city.improvement_rate
      }));
      
      setAlabamaCities(cities);
    } catch (error) {
      console.error('Failed to fetch cities:', error);
    }
  };

  const fetchMetrics = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/metrics`);
      const data = await response.json();
      
      setLiveMetrics({
        totalBuildings: data.total_buildings,
        processingSpeed: data.processing_speed,
        accuracy: data.accuracy,
        systemLoad: data.system_load,
        activeProcesses: data.active_processes,
        queueLength: data.queue_length
      });
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
    }
  };

  // Load initial data and set up real-time updates
  useEffect(() => {
    // Load initial data
    fetchCities();
    fetchMetrics();

    // Set up real-time updates
    const interval = setInterval(() => {
      fetchMetrics();
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  // Pipeline automation with API integration
  const startAutomatedPipeline = async () => {
    if (isProcessing) return;
    
    setIsProcessing(true);
    setCurrentStep(0);
    
    try {
      // Start processing via API
      const response = await fetch(`${API_BASE_URL}/api/process`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          city_name: alabamaCities.find(c => c.name.toLowerCase() === selectedCity)?.name || selectedCity,
          processing_options: {}
        })
      });
      
      const result = await response.json();
      console.log('Processing started:', result);
      
      // Simulate pipeline steps for UI
      for (let i = 0; i < steps.length; i++) {
        setCurrentStep(i);
        
        // Update step status to processing
        setSteps(prev => prev.map((step, index) => 
          index === i ? { ...step, status: 'processing', progress: 0 } : step
        ));

        // Simulate processing with progress updates
        for (let progress = 0; progress <= 100; progress += 10) {
          await new Promise(resolve => setTimeout(resolve, 200));
          setSteps(prev => prev.map((step, index) => 
            index === i ? { 
              ...step, 
              progress,
              memoryUsage: Math.random() * 8192,
              gpuUsage: Math.random() * 100
            } : step
          ));
        }

        // Complete the step
        const processingTime = Math.random() * 3 + 1;
        setSteps(prev => prev.map((step, index) => 
          index === i 
            ? { ...step, status: 'completed', progress: 100, processingTime }
            : step
        ));
      }
      
    } catch (error) {
      console.error('Processing failed:', error);
      // Fallback to local simulation
      for (let i = 0; i < steps.length; i++) {
        setCurrentStep(i);
        
        setSteps(prev => prev.map((step, index) => 
          index === i ? { ...step, status: 'processing', progress: 0 } : step
        ));

        for (let progress = 0; progress <= 100; progress += 20) {
          await new Promise(resolve => setTimeout(resolve, 300));
          setSteps(prev => prev.map((step, index) => 
            index === i ? { 
              ...step, 
              progress,
              memoryUsage: Math.random() * 8192,
              gpuUsage: Math.random() * 100
            } : step
          ));
        }

        const processingTime = Math.random() * 3 + 1;
        setSteps(prev => prev.map((step, index) => 
          index === i 
            ? { ...step, status: 'completed', progress: 100, processingTime }
            : step
        ));
      }
    }

    setIsProcessing(false);
    setCurrentStep(-1);
  };

  const resetPipeline = () => {
    setIsProcessing(false);
    setCurrentStep(-1);
    setSteps(prev => prev.map(step => ({ 
      ...step, 
      status: 'pending', 
      progress: 0, 
      processingTime: 0,
      memoryUsage: 0,
      gpuUsage: 0
    })));
  };

  // Google Maps integration
  useEffect(() => {
    const loadGoogleMaps = () => {
      if (window.google && window.google.maps) {
        initializeMap();
        return;
      }

      const script = document.createElement('script');
      script.src = `https://maps.googleapis.com/maps/api/js?key=YOUR_API_KEY&libraries=visualization&callback=initializeMap`;
      script.async = true;
      script.defer = true;
      
      window.initializeMap = initializeMap;
      document.head.appendChild(script);
    };

    loadGoogleMaps();
  }, []);

  const initializeMap = () => {
    if (!mapRef.current || !window.google) return;

    const alabamaCenter = { lat: 32.7794, lng: -86.8287 };
    
    const map = new window.google.maps.Map(mapRef.current, {
      center: alabamaCenter,
      zoom: 7,
      mapTypeId: 'hybrid',
      styles: [
        { elementType: 'geometry', stylers: [{ color: '#242f3e' }] },
        { elementType: 'labels.text.fill', stylers: [{ color: '#746855' }] },
        { featureType: 'water', elementType: 'geometry', stylers: [{ color: '#17263c' }] }
      ]
    });

    // Add city markers
    alabamaCities.forEach(city => {
      const marker = new window.google.maps.Marker({
        position: { lat: city.lat, lng: city.lng },
        map: map,
        title: city.name,
        icon: {
          path: window.google.maps.SymbolPath.CIRCLE,
          scale: 12,
          fillColor: city.accuracy > 90 ? '#10b981' : city.accuracy > 85 ? '#f59e0b' : '#ef4444',
          fillOpacity: 0.8,
          strokeWeight: 2,
          strokeColor: '#ffffff'
        }
      });

      const infoWindow = new window.google.maps.InfoWindow({
        content: `
          <div style="color: #000; padding: 10px;">
            <h3>${city.name}</h3>
            <p><strong>Buildings:</strong> ${city.buildings.toLocaleString()}</p>
            <p><strong>Accuracy:</strong> ${city.accuracy}%</p>
            <p><strong>Performance Score:</strong> ${city.performanceScore}/10</p>
            <button onclick="window.selectCity('${city.name.toLowerCase()}')" style="
              background: #3b82f6; 
              color: white; 
              border: none; 
              padding: 5px 10px; 
              border-radius: 4px; 
              cursor: pointer;
            ">
              Analyze City
            </button>
          </div>
        `
      });

      marker.addListener('click', () => {
        infoWindow.open(map, marker);
      });
    });

    // Global function for city selection
    (window as any).selectCity = (cityName: string) => {
      setSelectedCity(cityName);
      setActiveTab('pipeline');
    };
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 shadow-lg">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="text-3xl">🌍</div>
              <div>
                <h1 className="text-2xl font-bold">Alabama GeoAI Live Dashboard</h1>
                <p className="text-blue-100 text-sm">Real-time Building Detection & Analytics Platform</p>
              </div>
            </div>
            
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                  <span>Live: {liveMetrics.activeProcesses} processes</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-yellow-500 rounded-full animate-pulse"></div>
                  <span>Queue: {liveMetrics.queueLength}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Live Metrics Bar */}
      <div className="bg-gray-800 border-b border-gray-700 px-6 py-3">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-2 md:grid-cols-6 gap-4 text-sm">
            <div className="text-center">
              <div className="text-blue-400 font-semibold">{liveMetrics.totalBuildings.toLocaleString()}</div>
              <div className="text-gray-400">Total Buildings</div>
            </div>
            <div className="text-center">
              <div className="text-green-400 font-semibold">{liveMetrics.processingSpeed.toFixed(1)}/min</div>
              <div className="text-gray-400">Processing Speed</div>
            </div>
            <div className="text-center">
              <div className="text-purple-400 font-semibold">{liveMetrics.accuracy.toFixed(1)}%</div>
              <div className="text-gray-400">Avg Accuracy</div>
            </div>
            <div className="text-center">
              <div className="text-orange-400 font-semibold">{liveMetrics.systemLoad.toFixed(0)}%</div>
              <div className="text-gray-400">System Load</div>
            </div>
            <div className="text-center">
              <div className="text-red-400 font-semibold">{liveMetrics.activeProcesses}</div>
              <div className="text-gray-400">Active Jobs</div>
            </div>
            <div className="text-center">
              <div className="text-yellow-400 font-semibold">{liveMetrics.queueLength}</div>
              <div className="text-gray-400">Queue Length</div>
            </div>
          </div>
        </div>
      </div>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Navigation Tabs */}
        <div className="flex gap-1 mb-8 bg-gray-800 rounded-lg p-1">
          {['overview', 'analytics', 'pipeline', '3d', 'maps', 'live'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-6 py-2 rounded-md text-sm font-medium capitalize transition-all ${
                activeTab === tab
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>

        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 className="text-xl font-bold mb-6">City Performance Overview</h3>
              <div className="h-80">
                <Line data={performanceChartData} options={{...chartOptions, plugins: {...chartOptions.plugins, title: {display: false}}}} />
              </div>
            </div>
            
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 className="text-xl font-bold mb-6">Building Distribution</h3>
              <div className="h-80">
                <Doughnut data={buildingCountData} options={{...chartOptions, plugins: {...chartOptions.plugins, title: {display: false}}}} />
              </div>
            </div>
          </div>
        )}

        {/* Analytics Tab */}
        {activeTab === 'analytics' && (
          <div className="space-y-8">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <h3 className="text-xl font-bold mb-6">Processing Time Analysis</h3>
                <div className="h-80">
                  <Bar data={processingTimeData} options={{...chartOptions, plugins: {...chartOptions.plugins, title: {display: false}}}} />
                </div>
              </div>
              
              <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <h3 className="text-xl font-bold mb-6">Performance Correlation</h3>
                <div className="h-80">
                  <Scatter data={scatterData} options={{...chartOptions, plugins: {...chartOptions.plugins, title: {display: false}}}} />
                </div>
              </div>
            </div>

            {/* Performance Summary Table */}
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 className="text-xl font-bold mb-6">Detailed City Metrics</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left py-3 px-4">City</th>
                      <th className="text-left py-3 px-4">Buildings</th>
                      <th className="text-left py-3 px-4">Accuracy</th>
                      <th className="text-left py-3 px-4">Processing Time</th>
                      <th className="text-left py-3 px-4">Performance Score</th>
                      <th className="text-left py-3 px-4">Improvement Rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {alabamaCities.map((city, index) => (
                      <tr key={city.name} className="border-b border-gray-700/50 hover:bg-gray-700/20">
                        <td className="py-3 px-4 font-medium">{city.name}</td>
                        <td className="py-3 px-4">{city.buildings.toLocaleString()}</td>
                        <td className="py-3 px-4">
                          <span className={`px-2 py-1 rounded text-xs ${
                            city.accuracy > 90 ? 'bg-green-500/20 text-green-400' :
                            city.accuracy > 85 ? 'bg-yellow-500/20 text-yellow-400' :
                            'bg-red-500/20 text-red-400'
                          }`}>
                            {city.accuracy}%
                          </span>
                        </td>
                        <td className="py-3 px-4">{city.processingTime}s</td>
                        <td className="py-3 px-4">{city.performanceScore}/10</td>
                        <td className="py-3 px-4 text-green-400">+{city.improvementRate}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Pipeline Tab */}
        {activeTab === 'pipeline' && (
          <div className="space-y-8">
            {/* Pipeline Controls */}
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-bold">Live Processing Pipeline</h3>
                <div className="flex gap-3">
                  <select 
                    value={selectedCity}
                    onChange={(e) => setSelectedCity(e.target.value)}
                    className="bg-gray-700 text-white px-3 py-2 rounded border border-gray-600"
                    disabled={isProcessing}
                  >
                    {alabamaCities.map(city => (
                      <option key={city.name} value={city.name.toLowerCase()}>
                        {city.name}
                      </option>
                    ))}
                  </select>
                  <button 
                    onClick={startAutomatedPipeline}
                    disabled={isProcessing}
                    className="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded text-white font-medium"
                  >
                    {isProcessing ? 'Processing...' : 'Start Pipeline'}
                  </button>
                  <button 
                    onClick={resetPipeline}
                    disabled={isProcessing}
                    className="px-6 py-2 bg-gray-600 hover:bg-gray-700 disabled:bg-gray-700 rounded text-white font-medium"
                  >
                    Reset
                  </button>
                </div>
              </div>

              {/* Pipeline Steps */}
              <div className="space-y-4">
                {steps.map((step, index) => (
                  <div 
                    key={step.id}
                    className={`p-4 rounded-lg border ${
                      currentStep === index 
                        ? 'border-blue-500 bg-blue-900/20' 
                        : step.status === 'completed'
                        ? 'border-green-500 bg-green-900/20'
                        : 'border-gray-600 bg-gray-800/50'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-3">
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
                          step.status === 'completed' ? 'bg-green-500' :
                          step.status === 'processing' ? 'bg-blue-500' :
                          step.status === 'error' ? 'bg-red-500' : 'bg-gray-600'
                        }`}>
                          {step.status === 'completed' ? '✓' : 
                           step.status === 'processing' ? '⏳' :
                           step.status === 'error' ? '✗' : index + 1}
                        </div>
                        <h4 className="font-medium">{step.name}</h4>
                      </div>
                      
                      <div className="flex items-center gap-4 text-sm text-gray-400">
                        {step.status === 'processing' && (
                          <div>Progress: {step.progress}%</div>
                        )}
                        <div>Time: {step.processingTime.toFixed(1)}s</div>
                        {step.memoryUsage > 0 && <div>RAM: {step.memoryUsage.toFixed(0)}MB</div>}
                        {step.gpuUsage > 0 && <div>GPU: {step.gpuUsage.toFixed(0)}%</div>}
                      </div>
                    </div>
                    
                    {step.status === 'processing' && (
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div 
                          className="bg-blue-600 h-2 rounded-full transition-all duration-200"
                          style={{ width: `${step.progress}%` }}
                        />
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Real-time Visualization */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <h4 className="text-lg font-semibold mb-4">Processing Visualization</h4>
                <div className="space-y-4">
                  <div className="h-32 bg-gray-700 rounded flex items-center justify-center">
                    <span className="text-gray-400">Original: {alabamaCities.find(c => c.name.toLowerCase() === selectedCity)?.name || 'Select City'}</span>
                  </div>
                  <div className="h-32 bg-gray-700 rounded flex items-center justify-center">
                    <span className="text-gray-400">
                      {isProcessing && currentStep >= 0 
                        ? `Processing: ${steps[currentStep]?.name}`
                        : 'Ready for processing'
                      }
                    </span>
                  </div>
                </div>
              </div>

              <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <h4 className="text-lg font-semibold mb-4">Live System Metrics</h4>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">CPU Usage</span>
                    <div className="flex items-center gap-2">
                      <div className="w-24 bg-gray-700 rounded-full h-2">
                        <div 
                          className="bg-blue-500 h-2 rounded-full"
                          style={{ width: `${liveMetrics.systemLoad}%` }}
                        />
                      </div>
                      <span className="text-sm text-white w-12 text-right">{liveMetrics.systemLoad.toFixed(0)}%</span>
                    </div>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">Memory Usage</span>
                    <div className="flex items-center gap-2">
                      <div className="w-24 bg-gray-700 rounded-full h-2">
                        <div 
                          className="bg-green-500 h-2 rounded-full"
                          style={{ width: `${Math.min(100, liveMetrics.systemLoad + 10)}%` }}
                        />
                      </div>
                      <span className="text-sm text-white w-12 text-right">{Math.min(100, liveMetrics.systemLoad + 10).toFixed(0)}%</span>
                    </div>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">GPU Usage</span>
                    <div className="flex items-center gap-2">
                      <div className="w-24 bg-gray-700 rounded-full h-2">
                        <div 
                          className="bg-purple-500 h-2 rounded-full"
                          style={{ width: `${Math.min(100, liveMetrics.systemLoad - 5)}%` }}
                        />
                      </div>
                      <span className="text-sm text-white w-12 text-right">{Math.max(0, liveMetrics.systemLoad - 5).toFixed(0)}%</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 3D Visualization Tab */}
        {activeTab === '3d' && (
          <div className="space-y-8">
            {/* 3D Controls & Info */}
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 className="text-xl font-bold mb-6 flex items-center gap-3">
                <span className="text-blue-400">🏗️</span>
                3D Building Footprint Visualization
              </h3>
              
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
                <div className="bg-gray-700/50 rounded-lg p-4">
                  <h4 className="font-semibold text-green-400 mb-2">Real-time 3D Processing</h4>
                  <p className="text-gray-300 text-sm">
                    Interactive Three.js visualization connected to FastAPI backend for real-time building detection and 3D model generation.
                  </p>
                </div>
                
                <div className="bg-gray-700/50 rounded-lg p-4">
                  <h4 className="font-semibold text-purple-400 mb-2">Neural Network Analysis</h4>
                  <p className="text-gray-300 text-sm">
                    Advanced AI pipeline with confidence scoring, height estimation, and building type classification.
                  </p>
                </div>
                
                <div className="bg-gray-700/50 rounded-lg p-4">
                  <h4 className="font-semibold text-orange-400 mb-2">WebSocket Integration</h4>
                  <p className="text-gray-300 text-sm">
                    Live progress updates and real-time synchronization between frontend 3D visualization and backend processing.
                  </p>
                </div>
              </div>
            </div>

            {/* Main 3D Visualization */}
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-bold">Interactive 3D Building Detection</h3>
                <div className="flex items-center gap-3">
                  <span className="text-sm text-gray-400">Powered by Three.js + FastAPI</span>
                  <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
                </div>
              </div>
              
              {/* 3D Visualization Component */}
              <div className="h-96 rounded-lg overflow-hidden border border-gray-600">
                <Enhanced3DVisualization />
              </div>
              
              <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div className="bg-gray-700/50 rounded p-3">
                  <div className="text-green-400 font-semibold">WebGL Rendering</div>
                  <div className="text-gray-300">Hardware accelerated 3D graphics</div>
                </div>
                <div className="bg-gray-700/50 rounded p-3">
                  <div className="text-blue-400 font-semibold">Real-time Updates</div>
                  <div className="text-gray-300">Live WebSocket connection</div>
                </div>
                <div className="bg-gray-700/50 rounded p-3">
                  <div className="text-purple-400 font-semibold">AI Processing</div>
                  <div className="text-gray-300">Neural network analysis</div>
                </div>
                <div className="bg-gray-700/50 rounded p-3">
                  <div className="text-orange-400 font-semibold">Interactive Controls</div>
                  <div className="text-gray-300">Zoom, rotate, select buildings</div>
                </div>
              </div>
            </div>

            {/* 3D Analytics */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <h4 className="text-lg font-bold mb-4">3D Processing Pipeline</h4>
                <div className="space-y-3">
                  {[
                    { step: 'Satellite Image Acquisition', status: 'completed', color: 'green' },
                    { step: '3D Point Cloud Generation', status: 'completed', color: 'green' },
                    { step: 'Building Detection & Segmentation', status: 'processing', color: 'blue' },
                    { step: 'Height Estimation', status: 'queued', color: 'gray' },
                    { step: '3D Model Reconstruction', status: 'queued', color: 'gray' },
                    { step: 'Texture Mapping', status: 'queued', color: 'gray' },
                    { step: 'Quality Validation', status: 'queued', color: 'gray' }
                  ].map((item, index) => (
                    <div key={index} className="flex items-center gap-3">
                      <div className={`w-3 h-3 rounded-full bg-${item.color}-400`}></div>
                      <span className="flex-1 text-sm">{item.step}</span>
                      <span className={`text-xs text-${item.color}-400 capitalize`}>{item.status}</span>
                    </div>
                  ))}
                </div>
              </div>
              
              <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <h4 className="text-lg font-bold mb-4">3D Performance Metrics</h4>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-300">Rendering FPS</span>
                    <span className="text-green-400 font-mono">60</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-300">3D Objects</span>
                    <span className="text-blue-400 font-mono">{alabamaCities.reduce((sum, city) => sum + city.buildings, 0).toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-300">Memory Usage</span>
                    <span className="text-orange-400 font-mono">245 MB</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-300">GPU Utilization</span>
                    <span className="text-purple-400 font-mono">78%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Maps Tab */}
        {activeTab === 'maps' && (
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 className="text-xl font-bold mb-6">Interactive Alabama Cities Map</h3>
            <div 
              ref={mapRef}
              className="w-full h-96 rounded-lg"
              style={{ minHeight: '500px' }}
            />
            <div className="mt-4 text-sm text-gray-400">
              Click on city markers to view detailed metrics and start processing
            </div>
          </div>
        )}

        {/* Live Tab */}
        {activeTab === 'live' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Live Processing Queue */}
            <div className="lg:col-span-2 bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 className="text-xl font-bold mb-6">Live Processing Queue</h3>
              <div className="space-y-3">
                {Array.from({length: Math.max(1, liveMetrics.queueLength)}, (_, i) => (
                  <div key={i} className="flex items-center justify-between p-3 bg-gray-700 rounded">
                    <div className="flex items-center gap-3">
                      <div className="w-3 h-3 bg-yellow-500 rounded-full animate-pulse"></div>
                      <span>Job #{1000 + i} - {alabamaCities[i % alabamaCities.length].name}</span>
                    </div>
                    <div className="text-sm text-gray-400">
                      {i === 0 ? 'Processing...' : `Queue position: ${i}`}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* System Status */}
            <div className="space-y-6">
              <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <h4 className="text-lg font-semibold mb-4">System Health</h4>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-400">API Status</span>
                    <span className="text-green-400">Online</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Database</span>
                    <span className="text-green-400">Connected</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">ML Models</span>
                    <span className="text-green-400">Loaded</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Storage</span>
                    <span className="text-yellow-400">78% Full</span>
                  </div>
                </div>
              </div>

              <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <h4 className="text-lg font-semibold mb-4">Today's Stats</h4>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Jobs Completed</span>
                    <span className="text-white font-medium">127</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Buildings Processed</span>
                    <span className="text-white font-medium">48,392</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Avg Accuracy</span>
                    <span className="text-white font-medium">91.4%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Uptime</span>
                    <span className="text-green-400">99.8%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default CombinedDashboard;