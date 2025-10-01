'use client';

import React, { useEffect, useRef, useState } from 'react';

interface PipelineStep {
  id: string;
  name: string;
  processing_time: number;
  status: 'pending' | 'processing' | 'completed' | 'error';
}

interface PipelineMetrics {
  totalTime: number;
  accuracy: number;
  buildingsDetected: number;
  improvementPercent: number;
}

const LivePipelineVisualizer: React.FC = () => {
  const [currentStep, setCurrentStep] = useState(-1);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedCity, setSelectedCity] = useState('birmingham');
  const [metrics, setMetrics] = useState<PipelineMetrics>({
    totalTime: 0,
    accuracy: 0,
    buildingsDetected: 0,
    improvementPercent: 0
  });

  const [steps, setSteps] = useState<PipelineStep[]>([
    { id: 'detection', name: 'Building Detection', processing_time: 0, status: 'pending' },
    { id: 'regularization_rt', name: 'RT Regularization', processing_time: 0, status: 'pending' },
    { id: 'regularization_rr', name: 'RR Regularization', processing_time: 0, status: 'pending' },
    { id: 'regularization_fer', name: 'FER Regularization', processing_time: 0, status: 'pending' },
    { id: 'rl_fusion', name: 'RL Adaptive Fusion', processing_time: 0, status: 'pending' },
    { id: 'lapnet_refinement', name: 'LapNet Refinement', processing_time: 0, status: 'pending' },
    { id: 'visualization', name: 'Visualization', processing_time: 0, status: 'pending' }
  ]);

  const cities = {
    birmingham: { name: 'Birmingham, AL', buildings: 156421, accuracy: 0.912 },
    montgomery: { name: 'Montgomery, AL', buildings: 98742, accuracy: 0.897 },
    mobile: { name: 'Mobile, AL', buildings: 87634, accuracy: 0.884 },
    huntsville: { name: 'Huntsville, AL', buildings: 124563, accuracy: 0.923 },
    tuscaloosa: { name: 'Tuscaloosa, AL', buildings: 65432, accuracy: 0.901 }
  };

  const startPipeline = async () => {
    if (isProcessing) return;
    
    setIsProcessing(true);
    setCurrentStep(0);
    
    // Reset all steps
    setSteps(prev => prev.map(step => ({ ...step, status: 'pending', processing_time: 0 })));
    setMetrics({ totalTime: 0, accuracy: 0, buildingsDetected: 0, improvementPercent: 0 });

    // Process each step
    for (let i = 0; i < steps.length; i++) {
      setCurrentStep(i);
      
      // Update step status to processing
      setSteps(prev => prev.map((step, index) => 
        index === i ? { ...step, status: 'processing' } : step
      ));

      // Simulate processing time
      const processingTime = Math.random() * 3000 + 1000; // 1-4 seconds
      await new Promise(resolve => setTimeout(resolve, processingTime));

      // Update step completion
      setSteps(prev => prev.map((step, index) => 
        index === i 
          ? { ...step, status: 'completed', processing_time: processingTime / 1000 }
          : step
      ));

      // Update metrics progressively
      const cityData = cities[selectedCity as keyof typeof cities];
      const progress = (i + 1) / steps.length;
      
      setMetrics(prev => ({
        totalTime: prev.totalTime + processingTime / 1000,
        accuracy: cityData.accuracy * progress,
        buildingsDetected: Math.floor(cityData.buildings * progress),
        improvementPercent: progress * 15.2 // Simulated improvement
      }));
    }

    setIsProcessing(false);
    setCurrentStep(-1);
  };

  const resetPipeline = () => {
    setIsProcessing(false);
    setCurrentStep(-1);
    setSteps(prev => prev.map(step => ({ ...step, status: 'pending', processing_time: 0 })));
    setMetrics({ totalTime: 0, accuracy: 0, buildingsDetected: 0, improvementPercent: 0 });
  };

  const getStepIcon = (status: string) => {
    switch (status) {
      case 'processing': return '⏳';
      case 'completed': return '✅';
      case 'error': return '❌';
      default: return '⏸️';
    }
  };

  return (
    <div className="w-full bg-gray-900 text-white rounded-lg overflow-hidden">
      {/* Header */}
      <div className="p-4 bg-gray-800 border-b border-gray-700">
        <div className="flex justify-between items-center">
          <h3 className="text-xl font-bold">🔄 Live Processing Pipeline</h3>
          <div className="flex gap-3 items-center">
            <select 
              value={selectedCity}
              onChange={(e) => setSelectedCity(e.target.value)}
              className="bg-gray-700 text-white px-3 py-1 rounded border border-gray-600"
              disabled={isProcessing}
            >
              {Object.entries(cities).map(([key, city]) => (
                <option key={key} value={key}>{city.name}</option>
              ))}
            </select>
            <button 
              onClick={startPipeline}
              disabled={isProcessing}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded text-white"
            >
              {isProcessing ? 'Processing...' : 'Start Processing'}
            </button>
            <button 
              onClick={resetPipeline}
              disabled={isProcessing}
              className="px-4 py-2 bg-gray-600 hover:bg-gray-700 disabled:bg-gray-700 rounded text-white"
            >
              Reset
            </button>
          </div>
        </div>
      </div>

      <div className="p-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Pipeline Steps */}
          <div className="space-y-3">
            <h4 className="text-lg font-semibold mb-4">Processing Steps</h4>
            {steps.map((step, index) => (
              <div 
                key={step.id}
                className={`flex items-center p-3 rounded-lg border ${
                  currentStep === index 
                    ? 'border-blue-500 bg-blue-900/30' 
                    : step.status === 'completed'
                    ? 'border-green-500 bg-green-900/30'
                    : 'border-gray-600 bg-gray-800'
                }`}
              >
                <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center mr-3 text-sm font-bold">
                  {index + 1}
                </div>
                <div className="flex-1">
                  <div className="flex justify-between items-center mb-1">
                    <h5 className="font-medium">{step.name}</h5>
                    <span className="text-sm flex items-center gap-1">
                      {getStepIcon(step.status)}
                      <span className={`
                        ${step.status === 'processing' ? 'text-blue-400' : ''}
                        ${step.status === 'completed' ? 'text-green-400' : ''}
                        ${step.status === 'pending' ? 'text-gray-400' : ''}
                      `}>
                        {step.status}
                      </span>
                    </span>
                  </div>
                  {step.status === 'processing' && (
                    <div className="w-full bg-gray-700 rounded-full h-2 mb-1">
                      <div 
                        className="bg-blue-600 h-2 rounded-full animate-pulse"
                        style={{ width: '60%' }}
                      />
                    </div>
                  )}
                  <div className="text-xs text-gray-400">
                    Time: {step.processing_time.toFixed(1)}s
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Metrics and Visualization */}
          <div className="space-y-6">
            {/* Image Visualization */}
            <div>
              <h4 className="text-lg font-semibold mb-4">Processing Visualization</h4>
              <div className="grid grid-cols-1 gap-4">
                <div className="bg-gray-800 rounded-lg p-4">
                  <div className="text-sm text-gray-400 mb-2">Original Image</div>
                  <div className="h-32 bg-gray-700 rounded flex items-center justify-center">
                    <div className="text-gray-500">Satellite Image: {cities[selectedCity as keyof typeof cities].name}</div>
                  </div>
                </div>
                <div className="bg-gray-800 rounded-lg p-4">
                  <div className="text-sm text-gray-400 mb-2">Current Processing</div>
                  <div className="h-32 bg-gray-700 rounded flex items-center justify-center">
                    <div className="text-gray-500">
                      {isProcessing && currentStep >= 0 
                        ? `Processing: ${steps[currentStep]?.name}`
                        : 'Ready to process'
                      }
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Metrics */}
            <div>
              <h4 className="text-lg font-semibold mb-4">Real-time Metrics</h4>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-800 rounded-lg p-4 text-center">
                  <h5 className="text-sm text-gray-400 mb-1">Buildings</h5>
                  <div className="text-2xl font-bold text-blue-400">
                    {metrics.buildingsDetected.toLocaleString()}
                  </div>
                </div>
                <div className="bg-gray-800 rounded-lg p-4 text-center">
                  <h5 className="text-sm text-gray-400 mb-1">Accuracy</h5>
                  <div className="text-2xl font-bold text-green-400">
                    {(metrics.accuracy * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="bg-gray-800 rounded-lg p-4 text-center">
                  <h5 className="text-sm text-gray-400 mb-1">Total Time</h5>
                  <div className="text-2xl font-bold text-yellow-400">
                    {metrics.totalTime.toFixed(1)}s
                  </div>
                </div>
                <div className="bg-gray-800 rounded-lg p-4 text-center">
                  <h5 className="text-sm text-gray-400 mb-1">Improvement</h5>
                  <div className="text-2xl font-bold text-purple-400">
                    +{metrics.improvementPercent.toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Status Footer */}
        <div className="mt-6 p-3 bg-gray-800 rounded-lg">
          <div className="text-center text-gray-400">
            {isProcessing 
              ? `Processing ${cities[selectedCity as keyof typeof cities].name} - Step ${currentStep + 1}/${steps.length}`
              : 'Ready to process'
            }
          </div>
        </div>
      </div>
    </div>
  );
};

export default LivePipelineVisualizer;