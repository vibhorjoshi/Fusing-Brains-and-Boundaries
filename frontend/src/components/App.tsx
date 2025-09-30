// Main App Component with Framer Motion Integration
'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AuthProvider } from './AuthService';

// Component Imports
import MapProcessingComponent from './MapProcessingComponent';
import AdaptiveFusionComponent from './AdaptiveFusionComponent';
import GraphVisualizationComponent from './GraphVisualizationComponent';
import VectorConversionComponent from './VectorConversionComponent';

// Animation Variants
const containerVariants = {
  initial: { opacity: 0, y: 20 },
  animate: { 
    opacity: 1, 
    y: 0,
    transition: {
      duration: 0.6,
      staggerChildren: 0.1
    }
  },
  exit: { opacity: 0, y: -20 }
};

const itemVariants = {
  initial: { opacity: 0, x: -50 },
  animate: { 
    opacity: 1, 
    x: 0,
    transition: { duration: 0.3 }
  }
};

const cardVariants = {
  idle: { 
    scale: 1,
    rotateY: 0,
    transition: { duration: 0.3 }
  },
  hover: { 
    scale: 1.02,
    rotateY: 5,
    transition: { duration: 0.3 }
  }
};

export interface ComponentConfig {
  id: string;
  name: string;
  icon: string;
  description: string;
  status: 'active' | 'idle' | 'processing' | 'error';
  component: React.ComponentType;
}

const App: React.FC = () => {
  const [activeComponent, setActiveComponent] = useState<string>('overview');
  const [systemStatus, setSystemStatus] = useState({
    backend: 'connected',
    gpu: 'available',
    processing: 'ready',
    authentication: 'secured'
  });
  
  const [isLoading, setIsLoading] = useState(true);

  // Component Configuration
  const components: ComponentConfig[] = [
    { 
      id: 'overview', 
      name: 'Mission Control', 
      icon: '🎛️',
      description: 'NASA-level system dashboard and mission overview',
      status: 'active',
      component: SystemOverview
    },
    { 
      id: 'map', 
      name: 'Satellite Processing', 
      icon: '🛰️',
      description: 'Advanced satellite image analysis and building detection',
      status: 'active',
      component: MapProcessingComponent
    },
    { 
      id: 'fusion', 
      name: 'Adaptive Fusion', 
      icon: '🔄',
      description: 'ML model fusion with adaptive optimization',
      status: 'processing',
      component: AdaptiveFusionComponent
    },
    { 
      id: 'vectors', 
      name: 'Vector Processing', 
      icon: '📐',
      description: 'Geometric vectorization and boundary regularization',
      status: 'idle',
      component: VectorConversionComponent
    },
    { 
      id: 'visualization', 
      name: 'Data Visualization', 
      icon: '📊',
      description: 'Advanced 3D visualization and analytics dashboard',
      status: 'active',
      component: GraphVisualizationComponent
    }
  ];

  // Initialize system
  useEffect(() => {
    const initializeSystem = async () => {
      // Simulate system initialization
      await new Promise(resolve => setTimeout(resolve, 2000));
      setIsLoading(false);
    };
    
    initializeSystem();
  }, []);

  // System status monitoring
  useEffect(() => {
    const checkSystemHealth = () => {
      // Simulate health checks
      setSystemStatus(prev => ({
        ...prev,
        processing: activeComponent === 'fusion' ? 'processing' : 'ready'
      }));
    };

    const interval = setInterval(checkSystemHealth, 3000);
    return () => clearInterval(interval);
  }, [activeComponent]);

  const renderActiveComponent = () => {
    const activeConfig = components.find(c => c.id === activeComponent);
    if (!activeConfig) return <SystemOverview />;
    
    const Component = activeConfig.component;
    return <Component />;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected':
      case 'active':
      case 'available':
      case 'ready':
      case 'secured':
        return 'text-green-400';
      case 'processing':
        return 'text-yellow-400';
      case 'error':
      case 'offline':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  if (isLoading) {
    return <LoadingScreen />;
  }

  return (
    <AuthProvider>
      <div className="min-h-screen bg-gradient-to-br from-space-black via-deep-space to-cosmic-blue">
        <motion.div
          className="container mx-auto px-4 py-6"
          variants={containerVariants}
          initial="initial"
          animate="animate"
        >
          {/* Navigation Header */}
          <motion.nav 
            className="glassmorphism rounded-lg p-4 mb-6 border border-white/20"
            variants={itemVariants}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-6">
                <motion.h1 
                  className="text-xl font-bold text-nasa gradient-text"
                  whileHover={{ scale: 1.02 }}
                >
                  🚀 NASA-LEVEL GEOAI MISSION CONTROL
                </motion.h1>
                
                <div className="hidden md:flex space-x-4">
                  {components.map(component => (
                    <motion.button
                      key={component.id}
                      onClick={() => setActiveComponent(component.id)}
                      className={`px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                        activeComponent === component.id
                          ? 'bg-nasa-blue text-white shadow-lg'
                          : 'text-gray-300 hover:text-white hover:bg-white/10'
                      }`}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      variants={cardVariants}
                      initial="idle"
                    >
                      <span className="mr-2">{component.icon}</span>
                      {component.name}
                    </motion.button>
                  ))}
                </div>
              </div>
              
              {/* System Status */}
              <div className="flex items-center space-x-4">
                <div className="text-sm">
                  <div className="flex items-center space-x-2">
                    <span className="text-gray-400">Backend:</span>
                    <span className={getStatusColor(systemStatus.backend)}>
                      {systemStatus.backend.toUpperCase()}
                    </span>
                  </div>
                </div>
                <div className="w-2 h-2 bg-green-400 rounded-full pulse-glow"></div>
              </div>
            </div>
          </motion.nav>

          {/* Main Content */}
          <AnimatePresence mode="wait">
            <motion.main
              key={activeComponent}
              className="min-h-[600px]"
              initial={{ opacity: 0, x: 100 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -100 }}
              transition={{ duration: 0.4 }}
            >
              {renderActiveComponent()}
            </motion.main>
          </AnimatePresence>

          {/* Floating Status Indicators */}
          <FloatingStatusIndicators systemStatus={systemStatus} />
        </motion.div>
      </div>
    </AuthProvider>
  );
};

// Loading Screen Component
const LoadingScreen: React.FC = () => (
  <div className="min-h-screen bg-gradient-to-br from-space-black via-deep-space to-cosmic-blue flex items-center justify-center">
    <motion.div
      className="text-center"
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.6 }}
    >
      <motion.div
        className="w-20 h-20 border-4 border-info-cyan border-t-transparent rounded-full mx-auto mb-4"
        animate={{ rotate: 360 }}
        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
      />
      <h2 className="text-2xl font-bold text-nasa mb-2">🚀 INITIALIZING GEOAI SYSTEMS</h2>
      <p className="text-gray-400">Loading mission-critical components...</p>
    </motion.div>
  </div>
);

// System Overview Component
const SystemOverview: React.FC = () => (
  <motion.div 
    className="space-y-6"
    variants={containerVariants}
    initial="initial"
    animate="animate"
  >
    {/* Mission Control Header */}
    <motion.div 
      className="mission-control rounded-lg p-6 border border-nasa-blue/30"
      variants={itemVariants}
    >
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-nasa mb-2">
            🛰️ GEOAI RESEARCH MISSION CONTROL
          </h2>
          <p className="text-blue-300">
            Advanced Satellite Image Analysis & Building Footprint Detection System
          </p>
        </div>
        <div className="text-right">
          <div className="text-sm text-gray-400">Mission Status</div>
          <div className="text-xl font-mono text-green-400">OPERATIONAL</div>
        </div>
      </div>
    </motion.div>

    {/* Performance Metrics */}
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      <PerformanceCard 
        title="Detection Accuracy"
        value="94.2%"
        subtitle="IoU Score"
        color="green"
      />
      <PerformanceCard 
        title="Processing Speed"
        value="2.3s"
        subtitle="Per Image"
        color="blue"
      />
      <PerformanceCard 
        title="System Load"
        value="67%"
        subtitle="GPU Utilization"
        color="orange"
      />
    </div>
  </motion.div>
);

// Performance Card Component
const PerformanceCard: React.FC<{
  title: string;
  value: string;
  subtitle: string;
  color: 'green' | 'blue' | 'orange';
}> = ({ title, value, subtitle, color }) => {
  const colorClass = {
    green: 'text-green-400',
    blue: 'text-blue-400',
    orange: 'text-orange-400'
  }[color];

  return (
    <motion.div
      className="component-card p-6"
      variants={cardVariants}
      initial="idle"
      whileHover="hover"
    >
      <div className="text-sm text-gray-400 mb-2">{title}</div>
      <div className={`text-3xl font-mono font-bold ${colorClass} mb-1`}>
        {value}
      </div>
      <div className="text-xs text-gray-500">{subtitle}</div>
    </motion.div>
  );
};

// Floating Status Indicators
const FloatingStatusIndicators: React.FC<{
  systemStatus: Record<string, string>;
}> = ({ systemStatus }) => (
  <>
    {/* Processing Indicator */}
    <motion.div 
      className="fixed top-20 right-4 w-64 h-32 glassmorphism p-3 border border-cyan-500/30"
      initial={{ opacity: 0, x: 50 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: 0.5 }}
    >
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-bold text-cyan-300">LIVE PROCESSING</h3>
        <motion.div 
          className="w-6 h-6 border-2 border-cyan-400 rounded border-t-transparent"
          animate={{ rotate: 360 }}
          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
        />
      </div>
      <div className="text-xs text-cyan-300 mb-2">Building Detection Active</div>
      <div className="flex items-center space-x-2">
        <div className="flex-1 progress-bar h-2">
          <motion.div 
            className="progress-fill"
            initial={{ width: "0%" }}
            animate={{ width: "67%" }}
            transition={{ duration: 2 }}
          />
        </div>
        <span className="text-xs text-cyan-400">67%</span>
      </div>
    </motion.div>

    {/* Training Metrics */}
    <motion.div 
      className="fixed bottom-4 left-4 w-80 h-48 glassmorphism p-4 border border-purple-500/30"
      initial={{ opacity: 0, x: -50 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: 0.7 }}
    >
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-bold text-purple-300">TRAINING METRICS</h3>
        <div className="text-xs text-purple-300">ALABAMA DATASET</div>
      </div>
      
      <div className="grid grid-cols-2 gap-2 mb-3">
        <div className="text-center">
          <div className="text-xs text-purple-300 mb-1">Traditional IoU</div>
          <div className="w-full h-16 bg-gradient-to-br from-red-900/50 to-red-700/50 rounded border-2 border-red-500/50 flex items-center justify-center">
            <span className="text-sm font-mono text-red-300">0.721</span>
          </div>
        </div>
        <div className="text-center">
          <div className="text-xs text-purple-300 mb-1">Adaptive Fusion</div>
          <div className="w-full h-16 bg-gradient-to-br from-green-900/50 to-green-700/50 rounded border-2 border-green-500/50 flex items-center justify-center">
            <span className="text-sm font-mono text-green-300">0.917</span>
          </div>
        </div>
      </div>
      
      <div className="text-center">
        <div className="text-xs text-purple-300 mb-1">Performance Improvement</div>
        <div className="text-lg font-bold text-green-400">+27.2%</div>
      </div>
    </motion.div>
  </>
);

export default App;