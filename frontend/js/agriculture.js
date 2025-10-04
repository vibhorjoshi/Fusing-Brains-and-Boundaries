// Agriculture Module - USA Agriculture & Building Detection Interface
class AgricultureModule {
    constructor() {
        this.selectedStates = [];
        this.isProcessing = false;
        this.processedStates = 0;
        this.totalStates = 28;
        this.currentIteration = 20;
        this.activeJobs = new Map();
        this.usStates = this.initializeStates();
        this.init();
    }

    initializeStates() {
        return [
            'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut',
            'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
            'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan',
            'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada'
        ];
    }

    init() {
        this.populateStateSelectors();
        this.initControls();
        this.initUSAMap();
        this.initPerformanceMetrics();
        console.log('🌾 Agriculture Module initialized');
    }

    populateStateSelectors() {
        const stateSelect = document.getElementById('stateSelect');
        const multiStateSelect = document.getElementById('multiStateSelect');

        if (stateSelect) {
            this.usStates.forEach(state => {
                const option = document.createElement('option');
                option.value = state.toLowerCase().replace(/\s+/g, '-');
                option.textContent = state;
                stateSelect.appendChild(option);
            });
        }

        if (multiStateSelect) {
            this.usStates.forEach(state => {
                const option = document.createElement('option');
                option.value = state.toLowerCase().replace(/\s+/g, '-');
                option.textContent = state;
                multiStateSelect.appendChild(option);
            });
        }
    }

    initControls() {
        // Single state test
        const startStateTest = document.getElementById('startStateTest');
        startStateTest?.addEventListener('click', () => this.startSingleStateTest());

        // Batch processing
        const startBatchTest = document.getElementById('startBatchTest');
        startBatchTest?.addEventListener('click', () => this.startBatchTest());

        const stopAllTests = document.getElementById('stopAllTests');
        stopAllTests?.addEventListener('click', () => this.stopAllTests());

        // Iteration slider
        const iterSlider = document.getElementById('iterSlider');
        const iterValue = document.getElementById('iterValue');
        
        iterSlider?.addEventListener('input', (e) => {
            this.currentIteration = parseInt(e.target.value);
            if (iterValue) iterValue.textContent = this.currentIteration;
        });

        // Multi-select state selector
        const multiStateSelect = document.getElementById('multiStateSelect');
        multiStateSelect?.addEventListener('change', (e) => {
            this.selectedStates = Array.from(e.target.selectedOptions).map(option => option.value);
            console.log('🎯 Selected states:', this.selectedStates);
        });
    }

    initUSAMap() {
        const mapContainer = document.getElementById('usaMap');
        if (!mapContainer) return;

        // Create simplified USA map visualization
        this.createUSAMapVisualization(mapContainer);
        this.updateMapStats();
    }

    createUSAMapVisualization(container) {
        // Create a grid-based representation of US states
        container.innerHTML = `
            <div class="usa-grid-map">
                <div class="map-legend">
                    <div class="legend-item">
                        <div class="legend-color unprocessed"></div>
                        <span>Unprocessed</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color processing"></div>
                        <span>Processing</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color completed"></div>
                        <span>Completed</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color high-performance"></div>
                        <span>High Performance</span>
                    </div>
                </div>
                <div class="states-grid" id="statesGrid">
                    ${this.usStates.map((state, index) => `
                        <div class="state-tile unprocessed" 
                             data-state="${state.toLowerCase().replace(/\s+/g, '-')}"
                             title="${state}">
                            <span class="state-name">${state.substring(0, 3).toUpperCase()}</span>
                            <div class="state-progress"></div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;

        // Add click handlers for state tiles
        const stateTiles = container.querySelectorAll('.state-tile');
        stateTiles.forEach(tile => {
            tile.addEventListener('click', () => {
                const state = tile.getAttribute('data-state');
                this.selectSingleState(state);
            });
        });
    }

    selectSingleState(stateValue) {
        const stateSelect = document.getElementById('stateSelect');
        if (stateSelect) {
            stateSelect.value = stateValue;
            
            // Highlight selected state on map
            document.querySelectorAll('.state-tile').forEach(tile => {
                tile.classList.remove('selected');
            });
            
            const selectedTile = document.querySelector(`[data-state="${stateValue}"]`);
            if (selectedTile) {
                selectedTile.classList.add('selected');
            }
        }
    }

    initPerformanceMetrics() {
        // Initialize crop performance bars with demo data
        this.updateCropPerformance({
            corn: 0.92,
            soybeans: 0.89,
            wheat: 0.86,
            cotton: 0.91
        });

        // Start performance metrics updates
        this.startMetricsUpdates();
    }

    startMetricsUpdates() {
        // Update metrics every 2 seconds when processing
        this.metricsInterval = setInterval(() => {
            if (this.isProcessing) {
                this.updateLiveMetrics();
            }
        }, 2000);
    }

    async startSingleStateTest() {
        const stateSelect = document.getElementById('stateSelect');
        const selectedState = stateSelect?.value;

        if (!selectedState) {
            window.geoaiApp.showNotification('Please select a state', 'warning');
            return;
        }

        try {
            this.isProcessing = true;
            this.updateControlsState(true);
            
            const stateName = this.getStateNameFromValue(selectedState);
            window.geoaiApp.showNotification(`Starting processing for ${stateName}`, 'info');
            
            // Update state tile to processing
            this.updateStateTileStatus(selectedState, 'processing');
            
            // Start processing
            const result = await this.processState(selectedState, {
                iterations: this.currentIteration,
                realtimeTraining: document.getElementById('realtimeTraining')?.checked || false,
                buildingDetection: document.querySelector('[data-type="building"]')?.checked || true,
                cropDetection: document.querySelector('[data-type="crop"]')?.checked || true
            });
            
            // Update results
            this.handleStateProcessingComplete(selectedState, result);
            
        } catch (error) {
            window.geoaiApp.handleError(error, 'processing state');
            this.updateStateTileStatus(selectedState, 'error');
        } finally {
            this.isProcessing = false;
            this.updateControlsState(false);
        }
    }

    async startBatchTest() {
        if (this.selectedStates.length === 0) {
            window.geoaiApp.showNotification('Please select states for batch processing', 'warning');
            return;
        }

        try {
            this.isProcessing = true;
            this.updateControlsState(true);
            this.processedStates = 0;
            
            const parallelProcessing = document.getElementById('parallelProcessing')?.checked || false;
            const agricultureFocus = document.getElementById('agricultureFocus')?.checked || false;
            
            window.geoaiApp.showNotification(
                `Starting batch processing for ${this.selectedStates.length} states`, 
                'info'
            );

            if (parallelProcessing) {
                await this.processStatesInParallel();
            } else {
                await this.processStatesSequentially();
            }
            
        } catch (error) {
            window.geoaiApp.handleError(error, 'batch processing');
        } finally {
            this.isProcessing = false;
            this.updateControlsState(false);
        }
    }

    async processStatesInParallel() {
        const promises = this.selectedStates.map(state => 
            this.processStateWithRetry(state)
        );
        
        const results = await Promise.allSettled(promises);
        
        results.forEach((result, index) => {
            const state = this.selectedStates[index];
            if (result.status === 'fulfilled') {
                this.handleStateProcessingComplete(state, result.value);
            } else {
                this.updateStateTileStatus(state, 'error');
                console.error(`🔴 Failed to process ${state}:`, result.reason);
            }
        });
    }

    async processStatesSequentially() {
        for (const state of this.selectedStates) {
            try {
                this.updateStateTileStatus(state, 'processing');
                const result = await this.processState(state);
                this.handleStateProcessingComplete(state, result);
            } catch (error) {
                this.updateStateTileStatus(state, 'error');
                console.error(`🔴 Failed to process ${state}:`, error);
            }
        }
    }

    async processStateWithRetry(state, maxRetries = 3) {
        for (let i = 0; i < maxRetries; i++) {
            try {
                this.updateStateTileStatus(state, 'processing');
                return await this.processState(state);
            } catch (error) {
                if (i === maxRetries - 1) throw error;
                console.warn(`🔄 Retry ${i + 1} for ${state}:`, error.message);
                await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
            }
        }
    }

    async processState(stateValue, options = {}) {
        const stateName = this.getStateNameFromValue(stateValue);
        
        // Simulate or actual API call
        if (this.isDemoMode()) {
            return await this.simulateStateProcessing(stateName, options);
        } else {
            return await this.callStateProcessingAPI(stateValue, options);
        }
    }

    async simulateStateProcessing(stateName, options = {}) {
        const processingTime = Math.random() * 5000 + 2000; // 2-7 seconds
        
        await new Promise(resolve => setTimeout(resolve, processingTime));
        
        // Generate realistic results
        const basePerformance = 0.85 + Math.random() * 0.12; // 0.85-0.97
        
        return {
            state: stateName,
            building_detection: {
                iou: basePerformance,
                precision: basePerformance - 0.02 + Math.random() * 0.04,
                recall: basePerformance - 0.01 + Math.random() * 0.02,
                buildings_found: Math.floor(Math.random() * 1000 + 500)
            },
            crop_detection: {
                iou: basePerformance - 0.05 + Math.random() * 0.05,
                accuracy: basePerformance - 0.03 + Math.random() * 0.03,
                crop_types: ['corn', 'soybeans', 'wheat', 'cotton'].slice(0, Math.floor(Math.random() * 4) + 1)
            },
            rl_agent: {
                epsilon: 0.1 + Math.random() * 0.3,
                reward: Math.random() * 100 + 50,
                exploration_rate: Math.random() * 0.2
            },
            processing_time: processingTime / 1000,
            iterations: options.iterations || this.currentIteration
        };
    }

    async callStateProcessingAPI(stateValue, options = {}) {
        const response = await window.geoaiApp.apiRequest('/agriculture/process-state', 'POST', {
            state: stateValue,
            ...options
        });
        
        return response;
    }

    handleStateProcessingComplete(stateValue, result) {
        this.processedStates++;
        
        // Update state tile
        const performance = (result.building_detection?.iou || 0) + (result.crop_detection?.iou || 0);
        const status = performance > 1.6 ? 'high-performance' : 'completed';
        this.updateStateTileStatus(stateValue, status);
        
        // Update metrics
        this.updatePerformanceMetrics(result);
        
        // Update map stats
        this.updateMapStats();
        
        const stateName = this.getStateNameFromValue(stateValue);
        window.geoaiApp.showNotification(
            `${stateName} processing completed! IoU: ${(result.building_detection?.iou * 100 || 0).toFixed(1)}%`,
            'success'
        );
    }

    updateStateTileStatus(stateValue, status) {
        const tile = document.querySelector(`[data-state="${stateValue}"]`);
        if (tile) {
            tile.className = `state-tile ${status}`;
            
            if (status === 'processing') {
                tile.innerHTML += '<div class="processing-spinner"></div>';
            } else {
                const spinner = tile.querySelector('.processing-spinner');
                if (spinner) spinner.remove();
            }
        }
    }

    updatePerformanceMetrics(result) {
        // Building Detection
        const buildingPerf = document.getElementById('buildingPerf');
        if (buildingPerf && result.building_detection) {
            const newValue = result.building_detection.iou;
            window.geoaiApp.animateValue(buildingPerf, 
                parseFloat(buildingPerf.textContent) || 0, 
                newValue, 1000
            );
        }

        // Crop Detection
        const cropPerf = document.getElementById('cropPerf');
        if (cropPerf && result.crop_detection) {
            const newValue = result.crop_detection.iou;
            window.geoaiApp.animateValue(cropPerf, 
                parseFloat(cropPerf.textContent) || 0, 
                newValue, 1000
            );
        }

        // RL Agent Epsilon
        const rlEpsilon = document.getElementById('rlEpsilon');
        if (rlEpsilon && result.rl_agent) {
            const newValue = result.rl_agent.epsilon;
            window.geoaiApp.animateValue(rlEpsilon, 
                parseFloat(rlEpsilon.textContent) || 0, 
                newValue, 1000
            );
        }
    }

    updateCropPerformance(cropData) {
        Object.entries(cropData).forEach(([crop, performance]) => {
            const cropItem = document.querySelector(`.crop-item .crop-name:contains('${crop}')`);
            if (cropItem) {
                const progressBar = cropItem.parentElement.querySelector('.crop-progress');
                const valueSpan = cropItem.parentElement.querySelector('.crop-value');
                
                if (progressBar) {
                    window.geoaiApp.animateProgressBar(progressBar, performance * 100);
                }
                if (valueSpan) {
                    valueSpan.textContent = `${Math.round(performance * 100)}%`;
                }
            }
        });
    }

    updateMapStats() {
        const totalStates = document.getElementById('totalStates');
        const processedStates = document.getElementById('processedStates');
        const successRate = document.getElementById('successRate');

        if (totalStates) totalStates.textContent = this.totalStates;
        if (processedStates) processedStates.textContent = this.processedStates;
        
        if (successRate) {
            const rate = this.processedStates > 0 ? (this.processedStates / this.totalStates * 100) : 0;
            successRate.textContent = `${rate.toFixed(1)}%`;
        }
    }

    updateLiveMetrics() {
        // Simulate live metric updates during processing
        const buildingPerf = document.getElementById('buildingPerf');
        const cropPerf = document.getElementById('cropPerf');
        const rlEpsilon = document.getElementById('rlEpsilon');

        if (buildingPerf) {
            const current = parseFloat(buildingPerf.textContent) || 0;
            const variation = (Math.random() - 0.5) * 0.02; // ±1% variation
            const newValue = Math.max(0, Math.min(1, current + variation));
            buildingPerf.textContent = newValue.toFixed(3);
        }

        if (cropPerf) {
            const current = parseFloat(cropPerf.textContent) || 0;
            const variation = (Math.random() - 0.5) * 0.02;
            const newValue = Math.max(0, Math.min(1, current + variation));
            cropPerf.textContent = newValue.toFixed(3);
        }

        if (rlEpsilon) {
            const current = parseFloat(rlEpsilon.textContent) || 0.3;
            const variation = (Math.random() - 0.5) * 0.01;
            const newValue = Math.max(0.1, Math.min(0.5, current + variation));
            rlEpsilon.textContent = newValue.toFixed(3);
        }
    }

    updateControlsState(processing) {
        const controls = [
            'startStateTest',
            'startBatchTest',
            'stateSelect',
            'multiStateSelect'
        ];

        controls.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.disabled = processing;
            }
        });

        const stopBtn = document.getElementById('stopAllTests');
        if (stopBtn) {
            stopBtn.disabled = !processing;
        }
    }

    stopAllTests() {
        this.isProcessing = false;
        this.activeJobs.clear();
        
        // Reset all processing states
        document.querySelectorAll('.state-tile.processing').forEach(tile => {
            tile.classList.remove('processing');
            tile.classList.add('unprocessed');
            const spinner = tile.querySelector('.processing-spinner');
            if (spinner) spinner.remove();
        });

        this.updateControlsState(false);
        window.geoaiApp.showNotification('All tests stopped', 'info');
    }

    getStateNameFromValue(stateValue) {
        return this.usStates.find(state => 
            state.toLowerCase().replace(/\s+/g, '-') === stateValue
        ) || stateValue;
    }

    isDemoMode() {
        // Check if we should use demo mode (no backend connection)
        return !window.geoaiApp.websocket || window.geoaiApp.websocket.readyState !== WebSocket.OPEN;
    }

    // Export functionality
    exportResults() {
        const completedTiles = document.querySelectorAll('.state-tile.completed, .state-tile.high-performance');
        
        if (completedTiles.length === 0) {
            window.geoaiApp.showNotification('No completed states to export', 'warning');
            return;
        }

        const exportData = {
            timestamp: new Date().toISOString(),
            total_states: this.totalStates,
            processed_states: this.processedStates,
            success_rate: (this.processedStates / this.totalStates * 100).toFixed(2),
            configuration: {
                iterations: this.currentIteration,
                realtime_training: document.getElementById('realtimeTraining')?.checked || false,
                parallel_processing: document.getElementById('parallelProcessing')?.checked || false,
                agriculture_focus: document.getElementById('agricultureFocus')?.checked || false
            },
            results: Array.from(completedTiles).map(tile => ({
                state: tile.getAttribute('data-state'),
                status: tile.classList.contains('high-performance') ? 'high-performance' : 'completed'
            }))
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], { 
            type: 'application/json' 
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `usa_agriculture_results_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        window.geoaiApp.showNotification('Results exported successfully!', 'success');
    }

    // Reset functionality
    reset() {
        this.stopAllTests();
        this.processedStates = 0;
        
        // Reset all state tiles
        document.querySelectorAll('.state-tile').forEach(tile => {
            tile.className = 'state-tile unprocessed';
            const spinner = tile.querySelector('.processing-spinner');
            if (spinner) spinner.remove();
        });

        // Reset metrics
        const buildingPerf = document.getElementById('buildingPerf');
        const cropPerf = document.getElementById('cropPerf');
        const rlEpsilon = document.getElementById('rlEpsilon');

        if (buildingPerf) buildingPerf.textContent = '0.000';
        if (cropPerf) cropPerf.textContent = '0.000';
        if (rlEpsilon) rlEpsilon.textContent = '0.300';

        this.updateMapStats();
        window.geoaiApp.showNotification('Agriculture module reset', 'info');
    }
}

// Add CSS for additional agriculture components
const agricultureCSS = `
.usa-grid-map {
    background: rgba(255, 255, 255, 0.1);
    padding: 1rem;
    border-radius: 8px;
}

.map-legend {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.9rem;
}

.legend-color {
    width: 16px;
    height: 16px;
    border-radius: 4px;
}

.legend-color.unprocessed { background: rgba(255, 255, 255, 0.3); }
.legend-color.processing { background: #f39c12; }
.legend-color.completed { background: #27ae60; }
.legend-color.high-performance { background: #8e44ad; }

.states-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(60px, 1fr));
    gap: 0.5rem;
    max-height: 300px;
    overflow-y: auto;
}

.state-tile {
    aspect-ratio: 1;
    background: rgba(255, 255, 255, 0.3);
    border-radius: 4px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.3s ease;
    position: relative;
    border: 2px solid transparent;
}

.state-tile:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

.state-tile.selected {
    border-color: #f39c12;
    background: rgba(243, 156, 18, 0.3);
}

.state-tile.processing {
    background: #f39c12;
    animation: pulse 2s infinite;
}

.state-tile.completed {
    background: #27ae60;
}

.state-tile.high-performance {
    background: #8e44ad;
}

.state-tile.error {
    background: #e74c3c;
}

.state-name {
    font-size: 0.7rem;
    font-weight: bold;
    text-align: center;
}

.processing-spinner {
    position: absolute;
    top: 2px;
    right: 2px;
    width: 12px;
    height: 12px;
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-top: 2px solid white;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
`;

// Inject CSS
const style = document.createElement('style');
style.textContent = agricultureCSS;
document.head.appendChild(style);

// Initialize when DOM is loaded
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AgricultureModule;
}