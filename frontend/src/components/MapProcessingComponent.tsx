// Map Processing Component with Authentication
'use client';

import React, { useState, useEffect, useRef } from 'react';
import { withAuth, API_KEYS } from './AuthService';

const MapProcessingComponent = () => {
  const [mapData, setMapData] = useState(null);
  const [processingStatus, setProcessingStatus] = useState('idle');
  const [uploadedImage, setUploadedImage] = useState(null);
  const [processedResult, setProcessedResult] = useState(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setUploadedImage(e.target.result);
        processMapImage(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const processMapImage = async (imageData) => {
    setProcessingStatus('processing');
    
    try {
      const response = await fetch('http://localhost:8002/api/v1/map/process', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': API_KEYS.MAP_PROCESSING
        },
        body: JSON.stringify({
          image_data: imageData,
          processing_type: 'satellite_analysis',
          normalize: true,
          extract_features: true
        })
      });

      if (response.ok) {
        const result = await response.json();
        setProcessedResult(result);
        setProcessingStatus('completed');
        drawProcessedMap(result);
      } else {
        setProcessingStatus('error');
      }
    } catch (error) {
      console.error('Processing error:', error);
      setProcessingStatus('error');
    }
  };

  const drawProcessedMap = (result) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = 512;
    canvas.height = 512;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw processed features
    if (result.features) {
      result.features.forEach((feature, index) => {
        ctx.fillStyle = `hsl(${(index * 60) % 360}, 70%, 50%)`;
        ctx.fillRect(feature.x, feature.y, feature.width, feature.height);
      });
    }

    // Draw processing overlay
    ctx.fillStyle = 'rgba(0, 255, 136, 0.1)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  };

  const generateSampleSatelliteMap = () => {
    // Generate synthetic satellite data
    const sampleData = {
      coordinates: { lat: 32.3668, lon: -86.2999 }, // Montgomery, Alabama
      features: Array.from({ length: 20 }, (_, i) => ({
        x: Math.random() * 400 + 50,
        y: Math.random() * 400 + 50,
        width: Math.random() * 30 + 10,
        height: Math.random() * 30 + 10,
        type: ['building', 'road', 'vegetation'][Math.floor(Math.random() * 3)]
      })),
      confidence: Math.random() * 0.3 + 0.7
    };

    setProcessedResult(sampleData);
    setProcessingStatus('completed');
    drawProcessedMap(sampleData);
  };

  return (
    <div className="bg-gradient-to-br from-blue-900/30 to-green-900/30 p-6 rounded-lg border border-blue-500/30">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-bold text-blue-300">🗺️ Map Processing Component</h3>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
          <span className="text-green-400 text-sm">AUTHENTICATED</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload Section */}
        <div className="space-y-4">
          <div className="border-2 border-dashed border-blue-400/50 rounded-lg p-6 text-center">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleImageUpload}
              accept="image/*"
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
            >
              📁 Upload Satellite Image
            </button>
            <p className="text-blue-300 text-sm mt-2">
              Support: JPG, PNG, GeoTIFF
            </p>
          </div>

          <button
            onClick={generateSampleSatelliteMap}
            className="w-full bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg transition-colors"
          >
            🛰️ Generate Sample Alabama Map
          </button>

          {uploadedImage && (
            <div className="border border-gray-600 rounded-lg p-2">
              <img
                src={uploadedImage}
                alt="Uploaded"
                className="w-full h-32 object-cover rounded"
              />
            </div>
          )}
        </div>

        {/* Processing Results */}
        <div className="space-y-4">
          <div className="bg-black/30 rounded-lg p-4 border border-gray-600">
            <h4 className="text-blue-300 font-semibold mb-2">Processing Status</h4>
            <div className="flex items-center space-x-2">
              {processingStatus === 'processing' && (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-400"></div>
                  <span className="text-blue-400">Processing...</span>
                </>
              )}
              {processingStatus === 'completed' && (
                <>
                  <div className="w-4 h-4 bg-green-400 rounded-full"></div>
                  <span className="text-green-400">Completed</span>
                </>
              )}
              {processingStatus === 'error' && (
                <>
                  <div className="w-4 h-4 bg-red-400 rounded-full"></div>
                  <span className="text-red-400">Error</span>
                </>
              )}
              {processingStatus === 'idle' && (
                <>
                  <div className="w-4 h-4 bg-gray-400 rounded-full"></div>
                  <span className="text-gray-400">Waiting for input</span>
                </>
              )}
            </div>
          </div>

          <canvas
            ref={canvasRef}
            className="w-full border border-gray-600 rounded-lg bg-black/50"
            style={{ maxHeight: '256px' }}
          />

          {processedResult && (
            <div className="bg-black/30 rounded-lg p-4 border border-gray-600">
              <h4 className="text-blue-300 font-semibold mb-2">Analysis Results</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-300">Features Detected:</span>
                  <span className="text-green-400">{processedResult.features?.length || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-300">Confidence:</span>
                  <span className="text-green-400">{((processedResult.confidence || 0) * 100).toFixed(1)}%</span>
                </div>
                {processedResult.coordinates && (
                  <div className="flex justify-between">
                    <span className="text-gray-300">Location:</span>
                    <span className="text-blue-400">
                      {processedResult.coordinates.lat.toFixed(4)}, {processedResult.coordinates.lon.toFixed(4)}
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default withAuth(MapProcessingComponent, 'MapProcessing', API_KEYS.MAP_PROCESSING);