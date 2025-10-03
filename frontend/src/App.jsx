import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './App.css';
import Dashboard from './components/Dashboard';
import ProcessMap from './components/ProcessMap';
import ResultsViewer from './components/ResultsViewer';
import Settings from './components/Settings';
import About from './components/About';
import { fetchApiStatus } from './services/api';

function App() {
  const [apiStatus, setApiStatus] = useState({
    status: 'unknown',
    gpuAvailable: false,
    gpuCount: 0,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkApiStatus = async () => {
      try {
        const status = await fetchApiStatus();
        setApiStatus(status);
        toast.success(`API Connected: ${status.gpuAvailable ? `${status.gpuCount} GPUs available` : 'CPU mode'}`);
      } catch (error) {
        console.error('API connection error:', error);
        toast.error('Failed to connect to API server');
        setApiStatus({
          status: 'error',
          gpuAvailable: false,
          gpuCount: 0,
        });
      } finally {
        setLoading(false);
      }
    };

    checkApiStatus();
  }, []);

  return (
    <Router>
      <div className="app">
        <header className="app-header">
          <h1>Fusing Brains & Boundaries</h1>
          <p className="app-subtitle">GPU-Accelerated Building Footprint Extraction System</p>
          <nav className="app-nav">
            <Link to="/" className="nav-link">Dashboard</Link>
            <Link to="/process" className="nav-link">Process Map</Link>
            <Link to="/results" className="nav-link">Results</Link>
            <Link to="/settings" className="nav-link">Settings</Link>
            <Link to="/about" className="nav-link">About</Link>
          </nav>
          <div className={`api-status ${apiStatus.status === 'running' ? 'status-online' : 'status-offline'}`}>
            API Status: {loading ? 'Connecting...' : (apiStatus.status === 'running' ? 'Online' : 'Offline')}
            {apiStatus.gpuAvailable && !loading && (
              <span className="gpu-badge">
                {apiStatus.gpuCount} GPU{apiStatus.gpuCount !== 1 ? 's' : ''} Available
              </span>
            )}
          </div>
        </header>

        <main className="app-content">
          <Routes>
            <Route path="/" element={<Dashboard apiStatus={apiStatus} />} />
            <Route path="/process" element={<ProcessMap apiStatus={apiStatus} />} />
            <Route path="/results" element={<ResultsViewer />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </main>

        <footer className="app-footer">
          <p>© 2025 USA GeoAI Research - Fusing Brains & Boundaries</p>
        </footer>
        
        <ToastContainer position="bottom-right" />
      </div>
    </Router>
  );
}

export default App;