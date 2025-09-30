'use client';

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import * as THREE from 'three';

// NASA-style Launch Sequence Animation
interface LaunchSequenceProps {
  onLaunchComplete?: () => void;
}

function LaunchSequence({ onLaunchComplete }: LaunchSequenceProps) {
  const [countdown, setCountdown] = useState(10);
  const [phase, setPhase] = useState('PRE-LAUNCH');

  useEffect(() => {
    const interval = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          setPhase('SYSTEMS NOMINAL');
          if (onLaunchComplete) {
            setTimeout(onLaunchComplete, 1000);
          }
          clearInterval(interval);
          return 0;
        }
        return prev - 1;
      });
    }, 300);

    return () => clearInterval(interval);
  }, [onLaunchComplete]);

  return (
    <div className="launch-sequence">
      <div className="mission-patch">
        <div className="patch-ring">
          <div className="patch-inner">
            <span className="mission-name">GEOAI</span>
            <span className="mission-subtitle">BUILDING FOOTPRINT</span>
          </div>
        </div>
      </div>
      
      <div className="launch-status">
        <h1 className="launch-title">GEOAI MISSION CONTROL</h1>
        <div className="countdown-display">
          <span className="countdown-number">{countdown.toString().padStart(2, '0')}</span>
          <span className="countdown-label">SYSTEM INITIALIZATION</span>
        </div>
        <div className="phase-indicator">{phase}</div>
      </div>

      <div className="system-checks">
        <div className="check-item">
          <span className="check-dot"></span>
          <span>ML PIPELINE</span>
          <span className="check-status">✓ READY</span>
        </div>
        <div className="check-item">
          <span className="check-dot"></span>
          <span>SATELLITE NETWORK</span>
          <span className="check-status">✓ ONLINE</span>
        </div>
        <div className="check-item">
          <span className="check-dot"></span>
          <span>PROCESSING CORES</span>
          <span className="check-status">✓ NOMINAL</span>
        </div>
      </div>

      <style jsx>{`
        .launch-sequence {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          min-height: 100vh;
          background: radial-gradient(circle, #001122 0%, #000000 100%);
          color: #00ff41;
          font-family: 'Orbitron', 'Courier New', monospace;
        }

        .mission-patch {
          margin-bottom: 3rem;
        }

        .patch-ring {
          width: 200px;
          height: 200px;
          border: 3px solid #00ff41;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          animation: rotate 10s linear infinite;
          box-shadow: 0 0 30px #00ff41;
        }

        .patch-inner {
          text-align: center;
          padding: 2rem;
        }

        .mission-name {
          display: block;
          font-size: 2rem;
          font-weight: bold;
          color: #ffffff;
        }

        .mission-subtitle {
          display: block;
          font-size: 0.8rem;
          color: #00ff41;
          margin-top: 0.5rem;
        }

        .countdown-display {
          text-align: center;
          margin: 2rem 0;
        }

        .countdown-number {
          font-size: 6rem;
          font-weight: bold;
          color: #ff6600;
          text-shadow: 0 0 20px #ff6600;
          display: block;
        }

        .countdown-label {
          font-size: 1.2rem;
          color: #ffffff;
          letter-spacing: 3px;
        }

        .phase-indicator {
          font-size: 1.5rem;
          color: #00ff41;
          text-align: center;
          letter-spacing: 2px;
          margin: 2rem 0;
        }

        .system-checks {
          margin-top: 3rem;
        }

        .check-item {
          display: flex;
          align-items: center;
          margin: 1rem 0;
          padding: 0.5rem 1rem;
          background: rgba(0, 255, 65, 0.1);
          border: 1px solid rgba(0, 255, 65, 0.3);
          border-radius: 5px;
          min-width: 300px;
        }

        .check-dot {
          width: 10px;
          height: 10px;
          background: #00ff41;
          border-radius: 50%;
          margin-right: 1rem;
          animation: pulse 2s infinite;
        }

        .check-status {
          margin-left: auto;
          color: #00ff41;
          font-weight: bold;
        }

        @keyframes rotate {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.7; }
        }
      `}</style>
    </div>
  );
}

// NASA-style Header Component
interface HeaderProps {
  missionTime?: number;
}

function Header({ missionTime = 0 }: HeaderProps) {
  return (
    <header className="nasa-header">
      <div className="mission-info">
        <div className="nasa-logo">
          <span className="logo-text">GEOAI</span>
          <span className="logo-subtitle">MISSION CONTROL</span>
        </div>
        <div className="mission-time">
          <LiveClock missionTime={missionTime} />
        </div>
      </div>
      
      <div className="status-bar">
        <StatusIndicator label="COMM" status="NOMINAL" color="#00ff41" />
        <StatusIndicator label="PWR" status="OPTIMAL" color="#00ff41" />
        <StatusIndicator label="PROC" status="ACTIVE" color="#ffa500" />
        <StatusIndicator label="NET" status="SECURE" color="#00ff41" />
      </div>
    </header>
  );
}

// Live Mission Clock
interface LiveClockProps {
  missionTime?: number;
}

function LiveClock({ missionTime = 0 }: LiveClockProps) {
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const formatMissionTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="mission-clock">
      <div className="time-display">
        <span className="time-value">{formatMissionTime(missionTime)}</span>
        <span className="time-label">MISSION TIME</span>
      </div>
      <div className="date-display">
        <span className="date-value">{time.toISOString().substr(0, 10)}</span>
        <span className="date-label">MISSION DATE</span>
      </div>
    </div>
  );
}

// Status Indicator Component
function StatusIndicator({ label, status, color }: { label: string; status: string; color: string }) {
  return (
    <div className="status-indicator">
      <div className="status-light" style={{ backgroundColor: color }}></div>
      <div className="status-info">
        <span className="status-label">{label}</span>
        <span className="status-value">{status}</span>
      </div>
    </div>
  );
}

// Navigation Panel Component
function NavigationPanel({ activeModule, setActiveModule }: { activeModule: string; setActiveModule: (module: string) => void }) {
  const modules = [
    { id: 'mission-control', name: 'MISSION CONTROL', icon: '🚀' },
    { id: 'earth-observation', name: 'EARTH OBS', icon: '🌍' },
    { id: 'ml-processing', name: 'ML CORE', icon: '🧠' },
    { id: 'data-analysis', name: 'DATA ANALYSIS', icon: '📊' },
    { id: 'satellite-network', name: 'SAT NETWORK', icon: '🛰️' },
    { id: 'building-detection', name: 'BUILDING SCAN', icon: '🏢' }
  ];

  return (
    <nav className="navigation-panel">
      <div className="nav-title">COMMAND MODULES</div>
      {modules.map(module => (
        <motion.div
          key={module.id}
          className={`nav-module ${activeModule === module.id ? 'active' : ''}`}
          onClick={() => setActiveModule(module.id)}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <span className="module-icon">{module.icon}</span>
          <span className="module-name">{module.name}</span>
          {activeModule === module.id && <div className="active-indicator" />}
        </motion.div>
      ))}
    </nav>
  );
}

// Main Display Area
function MainDisplay({ activeModule }: { activeModule: string }) {
  const renderModule = () => {
    switch (activeModule) {
      case 'mission-control':
        return <MissionControlModule />;
      case 'earth-observation':
        return <EarthObservationModule />;
      case 'ml-processing':
        return <MLProcessingModule />;
      case 'data-analysis':
        return <DataAnalysisModule />;
      case 'satellite-network':
        return <SatelliteNetworkModule />;
      case 'building-detection':
        return <BuildingDetectionModule />;
      default:
        return <MissionControlModule />;
    }
  };

  return (
    <main className="main-display">
      <AnimatePresence mode="wait">
        <motion.div
          key={activeModule}
          initial={{ opacity: 0, x: 100 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -100 }}
          transition={{ duration: 0.5 }}
        >
          {renderModule()}
        </motion.div>
      </AnimatePresence>
    </main>
  );
}

// Mission Control Module
function MissionControlModule() {
  return (
    <div className="mission-control-module">
      <div className="module-header">
        <h2>🚀 MISSION CONTROL CENTER</h2>
        <div className="mission-status">STATUS: OPERATIONAL</div>
      </div>
      
      <div className="control-grid">
        <div className="control-panel">
          <h3>SYSTEM OVERVIEW</h3>
          <div className="metrics-grid">
            <Metric label="Building Footprints Processed" value="1,347,892" trend="+12%" />
            <Metric label="ML Models Active" value="3" trend="STABLE" />
            <Metric label="Processing Speed" value="18.7x" trend="+4.2%" />
            <Metric label="Accuracy (IoU)" value="94.98%" trend="+0.3%" />
          </div>
        </div>
        
        <div className="world-map-panel">
          <h3>GLOBAL OPERATIONS</h3>
          <WorldMap />
        </div>
        
        <div className="recent-activity">
          <h3>MISSION LOG</h3>
          <ActivityFeed />
        </div>
      </div>
    </div>
  );
}

// Additional modules (simplified for brevity)
function EarthObservationModule() {
  return (
    <div className="earth-obs-module">
      <h2>🌍 EARTH OBSERVATION SYSTEM</h2>
      <p>Satellite feeds and global coverage monitoring</p>
    </div>
  );
}

function MLProcessingModule() {
  return (
    <div className="ml-processing-module">
      <h2>🧠 MACHINE LEARNING CORE</h2>
      <p>Neural network processing and model performance</p>
    </div>
  );
}

function DataAnalysisModule() {
  return (
    <div className="data-analysis-module">
      <h2>📊 DATA ANALYSIS CENTER</h2>
      <p>Analytics and performance metrics</p>
    </div>
  );
}

function SatelliteNetworkModule() {
  return (
    <div className="satellite-network-module">
      <h2>🛰️ SATELLITE NETWORK</h2>
      <p>Satellite network monitoring and control</p>
    </div>
  );
}

function BuildingDetectionModule() {
  return (
    <div className="building-detection-module">
      <h2>🏢 BUILDING DETECTION</h2>
      <p>Live building footprint scanning and analysis</p>
    </div>
  );
}

// Metric Component
function Metric({ label, value, trend }: { label: string; value: string; trend: string }) {
  return (
    <div className="metric-card">
      <span className="metric-value">{value}</span>
      <span className="metric-label">{label}</span>
      <span className="metric-trend">{trend}</span>
    </div>
  );
}

// WorldMap Component
function WorldMap() {
  return (
    <div className="world-map">
      <div className="map-placeholder">
        [GLOBAL SATELLITE COVERAGE MAP]
      </div>
    </div>
  );
}

// Activity Feed Component
function ActivityFeed() {
  const activities = [
    "ML Model accuracy improved to 94.98%",
    "Satellite feed from Region-7 restored",
    "Processing pipeline optimization complete",
    "New building cluster detected in urban area"
  ];

  return (
    <div className="activity-feed">
      {activities.map((activity, index) => (
        <div key={index} className="activity-item">
          <span className="activity-time">{new Date().toLocaleTimeString()}</span>
          <span className="activity-text">{activity}</span>
        </div>
      ))}
    </div>
  );
}

export { LaunchSequence, Header, NavigationPanel, MainDisplay };