// Live Pipeline Visualization Module
class LivePipelineVisualizer {
    constructor(containerSelector, config = {}) {
        this.container = document.querySelector(containerSelector);
        this.config = Object.assign({
            animationDuration: 1000,
            showMetrics: true,
            autoUpdate: true,
            updateInterval: 5000,
            theme: 'dark'
        }, config);
        
        this.pipelineSteps = [
            { id: 'detection', name: 'Building Detection', processing_time: 0, status: 'pending' },
            { id: 'regularization_rt', name: 'RT Regularization', processing_time: 0, status: 'pending' },
            { id: 'regularization_rr', name: 'RR Regularization', processing_time: 0, status: 'pending' },
            { id: 'regularization_fer', name: 'FER Regularization', processing_time: 0, status: 'pending' },
            { id: 'rl_fusion', name: 'RL Adaptive Fusion', processing_time: 0, status: 'pending' },
            { id: 'lapnet_refinement', name: 'LapNet Refinement', processing_time: 0, status: 'pending' },
            { id: 'visualization', name: 'Visualization', processing_time: 0, status: 'pending' }
        ];
        
        this.currentStepIndex = -1;
        this.metrics = {
            totalTime: 0,
            accuracy: 0,
            buildingsDetected: 0,
            improvementPercent: 0
        };
        
        this.maskImages = {
            original: null,
            detection: null,
            regularization_rt: null,
            regularization_rr: null,
            regularization_fer: null,
            rl_fusion: null,
            lapnet_refinement: null
        };
        
        this.initialize();
        
        if (this.config.autoUpdate) {
            this.startAutoUpdates();
        }
    }
    
    initialize() {
        if (!this.container) {
            console.error('Pipeline visualization container not found');
            return;
        }
        
        this.createUI();
    }
    
    createUI() {
        // Create pipeline container
        this.container.innerHTML = `
            <div class="pipeline-visualizer ${this.config.theme}">
                <div class="pipeline-header">
                    <h3>🔄 Live Processing Pipeline</h3>
                    <div class="pipeline-controls">
                        <select id="pipeline-demo-city">
                            <option value="birmingham">Birmingham, AL</option>
                            <option value="montgomery">Montgomery, AL</option>
                            <option value="mobile">Mobile, AL</option>
                            <option value="huntsville">Huntsville, AL</option>
                            <option value="tuscaloosa">Tuscaloosa, AL</option>
                        </select>
                        <button id="pipeline-start-btn" class="btn-primary">Start Processing</button>
                        <button id="pipeline-reset-btn" class="btn-secondary">Reset</button>
                    </div>
                </div>
                
                <div class="pipeline-body">
                    <div class="pipeline-steps">
                        ${this.pipelineSteps.map((step, index) => `
                            <div class="pipeline-step" data-step="${step.id}">
                                <div class="step-number">${index + 1}</div>
                                <div class="step-content">
                                    <div class="step-header">
                                        <h4>${step.name}</h4>
                                        <span class="step-status ${step.status}">${step.status}</span>
                                    </div>
                                    <div class="step-progress">
                                        <div class="progress-bar" style="width: 0%"></div>
                                    </div>
                                    <div class="step-metrics">
                                        <span class="processing-time">0.0s</span>
                                        <span class="memory-usage">0.0 MB</span>
                                    </div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                    
                    <div class="pipeline-visualization">
                        <div class="image-container">
                            <div class="image-label">Original Image</div>
                            <div class="image-placeholder" id="original-image">
                                <div class="loading-spinner"></div>
                            </div>
                        </div>
                        <div class="image-container">
                            <div class="image-label">Current Processing</div>
                            <div class="image-placeholder" id="processing-image">
                                <div class="no-image">Select a city and start processing</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="pipeline-metrics" id="pipeline-metrics">
                        <div class="metric-card">
                            <h4>Buildings</h4>
                            <div class="metric-value" id="buildings-detected">0</div>
                        </div>
                        <div class="metric-card">
                            <h4>Accuracy</h4>
                            <div class="metric-value" id="detection-accuracy">0.0%</div>
                        </div>
                        <div class="metric-card">
                            <h4>Time</h4>
                            <div class="metric-value" id="total-time">0.0s</div>
                        </div>
                        <div class="metric-card">
                            <h4>Improvement</h4>
                            <div class="metric-value" id="improvement-percent">+0.0%</div>
                        </div>
                    </div>
                </div>
                
                <div class="pipeline-footer">
                    <div class="pipeline-status" id="pipeline-status">Ready to process</div>
                </div>
            </div>
        `;
        
        // Add event listeners
        document.getElementById('pipeline-start-btn').addEventListener('click', () => this.startPipeline());
        document.getElementById('pipeline-reset-btn').addEventListener('click', () => this.resetPipeline());
        document.getElementById('pipeline-demo-city').addEventListener('change', () => this.loadCityImage());
        
        // Apply styles
        this.applyStyles();
    }
    
    applyStyles() {
        const styleElement = document.createElement('style');
        styleElement.textContent = `
            .pipeline-visualizer {
                background: rgba(10, 10, 26, 0.8);
                border: 1px solid rgba(100, 255, 218, 0.2);
                border-radius: 12px;
                padding: 20px;
                color: white;
                margin-bottom: 20px;
            }
            
            .pipeline-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }
            
            .pipeline-controls {
                display: flex;
                gap: 10px;
            }
            
            .pipeline-controls select, .pipeline-controls button {
                padding: 8px 12px;
                border-radius: 4px;
                border: none;
            }
            
            .btn-primary {
                background: #64ffda;
                color: #0a0a1a;
                font-weight: bold;
                cursor: pointer;
            }
            
            .btn-secondary {
                background: rgba(255, 255, 255, 0.2);
                color: white;
                cursor: pointer;
            }
            
            .pipeline-body {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }
            
            .pipeline-steps {
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            
            .pipeline-step {
                display: flex;
                gap: 10px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 8px;
                padding: 10px;
                transition: all 0.3s ease;
            }
            
            .pipeline-step.active {
                background: rgba(100, 255, 218, 0.1);
                border-left: 4px solid #64ffda;
            }
            
            .pipeline-step.completed {
                background: rgba(0, 200, 83, 0.1);
                border-left: 4px solid #00c853;
            }
            
            .step-number {
                width: 28px;
                height: 28px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
            }
            
            .pipeline-step.active .step-number {
                background: #64ffda;
                color: #0a0a1a;
            }
            
            .pipeline-step.completed .step-number {
                background: #00c853;
                color: white;
            }
            
            .step-content {
                flex: 1;
            }
            
            .step-header {
                display: flex;
                justify-content: space-between;
                margin-bottom: 5px;
            }
            
            .step-status {
                font-size: 0.8rem;
                padding: 2px 6px;
                border-radius: 4px;
                background: rgba(255, 255, 255, 0.1);
            }
            
            .step-status.pending {
                background: rgba(255, 255, 255, 0.1);
            }
            
            .step-status.processing {
                background: rgba(100, 255, 218, 0.3);
                color: #64ffda;
            }
            
            .step-status.completed {
                background: rgba(0, 200, 83, 0.3);
                color: #00c853;
            }
            
            .step-progress {
                height: 6px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 3px;
                margin-bottom: 5px;
                overflow: hidden;
            }
            
            .progress-bar {
                height: 100%;
                background: #64ffda;
                width: 0;
                transition: width 0.5s ease;
            }
            
            .step-metrics {
                display: flex;
                justify-content: space-between;
                font-size: 0.8rem;
                color: rgba(255, 255, 255, 0.6);
            }
            
            .pipeline-visualization {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
                grid-row: span 2;
            }
            
            .image-container {
                background: rgba(0, 0, 0, 0.2);
                border-radius: 8px;
                overflow: hidden;
                display: flex;
                flex-direction: column;
            }
            
            .image-label {
                padding: 8px;
                background: rgba(0, 0, 0, 0.4);
                font-size: 0.9rem;
                text-align: center;
            }
            
            .image-placeholder {
                height: 200px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: #0a0a1a;
            }
            
            .loading-spinner {
                border: 4px solid rgba(255, 255, 255, 0.1);
                border-radius: 50%;
                border-top: 4px solid #64ffda;
                width: 30px;
                height: 30px;
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .no-image {
                color: rgba(255, 255, 255, 0.4);
                text-align: center;
                padding: 20px;
            }
            
            .pipeline-metrics {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 10px;
                margin-top: 20px;
                grid-column: span 2;
            }
            
            .metric-card {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 8px;
                padding: 15px;
                text-align: center;
            }
            
            .metric-card h4 {
                font-size: 0.9rem;
                margin-bottom: 5px;
                color: rgba(255, 255, 255, 0.7);
            }
            
            .metric-value {
                font-size: 1.5rem;
                font-weight: bold;
                color: #64ffda;
            }
            
            .pipeline-footer {
                margin-top: 20px;
                text-align: center;
                padding-top: 10px;
                border-top: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .pipeline-status {
                font-style: italic;
                color: rgba(255, 255, 255, 0.7);
            }
        `;
        
        document.head.appendChild(styleElement);
    }
    
    // Start the pipeline processing visualization
    startPipeline() {
        if (this.currentStepIndex >= 0) {
            console.warn('Pipeline already running');
            return;
        }
        
        const citySelect = document.getElementById('pipeline-demo-city');
        const cityName = citySelect.options[citySelect.selectedIndex].text;
        
        document.getElementById('pipeline-status').textContent = `Processing ${cityName}...`;
        
        // Reset metrics
        this.metrics = {
            totalTime: 0,
            accuracy: 0,
            buildingsDetected: 0,
            improvementPercent: 0
        };
        
        this.updateMetricsDisplay();
        
        // Start with the first step
        this.currentStepIndex = 0;
        this.processNextStep();
    }
    
    // Process the next pipeline step
    processNextStep() {
        if (this.currentStepIndex >= this.pipelineSteps.length) {
            this.completePipeline();
            return;
        }
        
        const currentStep = this.pipelineSteps[this.currentStepIndex];
        
        // Update UI to show current step
        this.updateStepUI(currentStep.id, 'processing');
        
        // Simulate processing time
        const processingTime = 0.5 + Math.random() * 2.5;
        currentStep.processing_time = processingTime;
        
        // Update progress bar animation
        this.animateProgress(currentStep.id);
        
        // Simulate step completion
        setTimeout(() => {
            // Step completed
            this.updateStepUI(currentStep.id, 'completed');
            
            // Update metrics
            this.metrics.totalTime += processingTime;
            
            if (currentStep.id === 'detection') {
                this.metrics.buildingsDetected = Math.floor(50 + Math.random() * 200);
                this.metrics.accuracy = 0.65 + Math.random() * 0.1;
            } else if (currentStep.id === 'rl_fusion') {
                this.metrics.accuracy += 0.05 + Math.random() * 0.03;
                this.metrics.improvementPercent = ((this.metrics.accuracy / 0.65) - 1) * 100;
            } else if (currentStep.id === 'lapnet_refinement') {
                this.metrics.accuracy += 0.02 + Math.random() * 0.02;
                this.metrics.improvementPercent = ((this.metrics.accuracy / 0.65) - 1) * 100;
            }
            
            this.updateMetricsDisplay();
            
            // Update the processing image
            this.updateProcessingImage(currentStep.id);
            
            // Move to next step
            this.currentStepIndex++;
            this.processNextStep();
            
        }, processingTime * 1000);
    }
    
    // Complete the pipeline processing
    completePipeline() {
        document.getElementById('pipeline-status').textContent = `Processing complete! Found ${this.metrics.buildingsDetected} buildings with ${(this.metrics.accuracy * 100).toFixed(1)}% accuracy`;
        this.currentStepIndex = -1;
    }
    
    // Reset the pipeline to initial state
    resetPipeline() {
        this.currentStepIndex = -1;
        
        // Reset all steps
        this.pipelineSteps.forEach(step => {
            this.updateStepUI(step.id, 'pending');
        });
        
        // Reset metrics
        this.metrics = {
            totalTime: 0,
            accuracy: 0,
            buildingsDetected: 0,
            improvementPercent: 0
        };
        
        this.updateMetricsDisplay();
        
        // Reset images
        document.getElementById('processing-image').innerHTML = `<div class="no-image">Select a city and start processing</div>`;
        
        // Reset status
        document.getElementById('pipeline-status').textContent = 'Ready to process';
    }
    
    // Update the UI for a specific step
    updateStepUI(stepId, status) {
        const stepElement = document.querySelector(`.pipeline-step[data-step="${stepId}"]`);
        
        if (!stepElement) return;
        
        // Remove all status classes
        stepElement.classList.remove('active', 'completed');
        
        // Add appropriate class
        if (status === 'processing') {
            stepElement.classList.add('active');
        } else if (status === 'completed') {
            stepElement.classList.add('completed');
        }
        
        // Update status label
        const statusElement = stepElement.querySelector('.step-status');
        if (statusElement) {
            statusElement.textContent = status;
            statusElement.className = `step-status ${status}`;
        }
        
        // Update processing time if completed
        if (status === 'completed') {
            const step = this.pipelineSteps.find(s => s.id === stepId);
            if (step) {
                const timeElement = stepElement.querySelector('.processing-time');
                if (timeElement) {
                    timeElement.textContent = `${step.processing_time.toFixed(1)}s`;
                }
                
                // Add random memory usage
                const memoryElement = stepElement.querySelector('.memory-usage');
                if (memoryElement) {
                    const memoryUsage = (Math.random() * 500 + 100).toFixed(1);
                    memoryElement.textContent = `${memoryUsage} MB`;
                }
            }
        }
    }
    
    // Animate progress bar for a step
    animateProgress(stepId) {
        const progressBar = document.querySelector(`.pipeline-step[data-step="${stepId}"] .progress-bar`);
        
        if (!progressBar) return;
        
        // Reset progress
        progressBar.style.width = '0%';
        
        // Animate to 100%
        setTimeout(() => {
            progressBar.style.width = '100%';
        }, 50);
    }
    
    // Update metrics display
    updateMetricsDisplay() {
        document.getElementById('buildings-detected').textContent = this.metrics.buildingsDetected;
        document.getElementById('detection-accuracy').textContent = `${(this.metrics.accuracy * 100).toFixed(1)}%`;
        document.getElementById('total-time').textContent = `${this.metrics.totalTime.toFixed(1)}s`;
        document.getElementById('improvement-percent').textContent = `+${this.metrics.improvementPercent.toFixed(1)}%`;
    }
    
    // Load a city image (simulated)
    loadCityImage() {
        const citySelect = document.getElementById('pipeline-demo-city');
        const cityName = citySelect.options[citySelect.selectedIndex].text;
        
        // Show loading spinner
        document.getElementById('original-image').innerHTML = `<div class="loading-spinner"></div>`;
        
        // Simulate image loading
        setTimeout(() => {
            // Generate a random color for this city (for demo purposes)
            const hue = Math.floor(Math.random() * 360);
            
            // Create a canvas for the image
            const canvas = document.createElement('canvas');
            canvas.width = 300;
            canvas.height = 200;
            const ctx = canvas.getContext('2d');
            
            // Draw a gradient background
            const gradient = ctx.createLinearGradient(0, 0, 300, 200);
            gradient.addColorStop(0, `hsl(${hue}, 70%, 20%)`);
            gradient.addColorStop(1, `hsl(${hue}, 70%, 40%)`);
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, 300, 200);
            
            // Draw some random "buildings"
            ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
            for (let i = 0; i < 50; i++) {
                const x = Math.random() * 280;
                const y = Math.random() * 180;
                const width = 5 + Math.random() * 20;
                const height = 10 + Math.random() * 30;
                ctx.fillRect(x, y, width, height);
            }
            
            // Add city name
            ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
            ctx.font = '16px Arial';
            ctx.fillText(cityName, 10, 30);
            
            // Display the image
            document.getElementById('original-image').innerHTML = '';
            document.getElementById('original-image').appendChild(canvas);
            
            // Store this as the original image
            this.maskImages.original = canvas;
            
            // Reset processing image
            document.getElementById('processing-image').innerHTML = `<div class="no-image">Select a city and start processing</div>`;
            
            // Update status
            document.getElementById('pipeline-status').textContent = `Loaded ${cityName}. Ready to process.`;
            
        }, 1000);
    }
    
    // Update processing image based on current step
    updateProcessingImage(stepId) {
        if (!this.maskImages.original) return;
        
        // Create a canvas for the processed image
        const canvas = document.createElement('canvas');
        canvas.width = 300;
        canvas.height = 200;
        const ctx = canvas.getContext('2d');
        
        // Start with the original image
        ctx.drawImage(this.maskImages.original, 0, 0);
        
        // Apply processing effect based on step
        switch(stepId) {
            case 'detection':
                // Add red rectangles
                ctx.strokeStyle = 'rgba(255, 50, 50, 0.7)';
                ctx.lineWidth = 2;
                for (let i = 0; i < 25; i++) {
                    const x = Math.random() * 250;
                    const y = Math.random() * 150;
                    const width = 10 + Math.random() * 30;
                    const height = 15 + Math.random() * 40;
                    ctx.strokeRect(x, y, width, height);
                }
                break;
                
            case 'regularization_rt':
                // Add red rectangles with better alignment
                ctx.strokeStyle = 'rgba(255, 120, 50, 0.7)';
                ctx.lineWidth = 2;
                for (let i = 0; i < 25; i++) {
                    const x = Math.floor(Math.random() * 24) * 10;
                    const y = Math.floor(Math.random() * 15) * 10;
                    const width = Math.floor(2 + Math.random() * 4) * 10;
                    const height = Math.floor(2 + Math.random() * 5) * 10;
                    ctx.strokeRect(x, y, width, height);
                }
                break;
                
            case 'regularization_rr':
                // Add even better aligned shapes
                ctx.strokeStyle = 'rgba(255, 200, 50, 0.7)';
                ctx.lineWidth = 2;
                for (let i = 0; i < 25; i++) {
                    const x = Math.floor(Math.random() * 24) * 10;
                    const y = Math.floor(Math.random() * 15) * 10;
                    const width = Math.floor(2 + Math.random() * 4) * 10;
                    const height = Math.floor(2 + Math.random() * 5) * 10;
                    
                    ctx.beginPath();
                    ctx.moveTo(x, y);
                    ctx.lineTo(x + width, y);
                    ctx.lineTo(x + width, y + height);
                    ctx.lineTo(x, y + height);
                    ctx.closePath();
                    ctx.stroke();
                }
                break;
                
            case 'regularization_fer':
                // Add green more complex polygons
                ctx.strokeStyle = 'rgba(100, 255, 100, 0.7)';
                ctx.lineWidth = 2;
                for (let i = 0; i < 20; i++) {
                    const x = Math.floor(Math.random() * 24) * 10;
                    const y = Math.floor(Math.random() * 15) * 10;
                    const width = Math.floor(2 + Math.random() * 4) * 10;
                    const height = Math.floor(2 + Math.random() * 5) * 10;
                    
                    // Draw slightly more complex shape
                    ctx.beginPath();
                    ctx.moveTo(x, y);
                    ctx.lineTo(x + width, y);
                    ctx.lineTo(x + width, y + height);
                    ctx.lineTo(x + width/2, y + height + 10);
                    ctx.lineTo(x, y + height);
                    ctx.closePath();
                    ctx.stroke();
                }
                break;
                
            case 'rl_fusion':
                // Add blue good looking polygons
                ctx.strokeStyle = 'rgba(50, 150, 255, 0.8)';
                ctx.lineWidth = 2;
                for (let i = 0; i < 18; i++) {
                    const x = Math.floor(Math.random() * 24) * 10;
                    const y = Math.floor(Math.random() * 15) * 10;
                    const width = Math.floor(2 + Math.random() * 4) * 10;
                    const height = Math.floor(2 + Math.random() * 5) * 10;
                    
                    // Draw slightly more complex shape with fill
                    ctx.fillStyle = 'rgba(50, 150, 255, 0.2)';
                    ctx.beginPath();
                    ctx.moveTo(x, y);
                    ctx.lineTo(x + width, y);
                    ctx.lineTo(x + width, y + height);
                    ctx.lineTo(x, y + height);
                    ctx.closePath();
                    ctx.fill();
                    ctx.stroke();
                }
                break;
                
            case 'lapnet_refinement':
                // Add cyan precise polygons
                ctx.strokeStyle = 'rgba(0, 255, 255, 0.9)';
                ctx.lineWidth = 2;
                for (let i = 0; i < 15; i++) {
                    const x = Math.floor(Math.random() * 24) * 10;
                    const y = Math.floor(Math.random() * 15) * 10;
                    const width = Math.floor(2 + Math.random() * 4) * 10;
                    const height = Math.floor(2 + Math.random() * 5) * 10;
                    
                    // Draw more precise shape with fill
                    ctx.fillStyle = 'rgba(0, 255, 255, 0.3)';
                    ctx.beginPath();
                    
                    // Create a more complex polygon
                    const points = [];
                    const sides = Math.floor(4 + Math.random() * 4);
                    for (let j = 0; j < sides; j++) {
                        const angle = (j / sides) * Math.PI * 2;
                        const radius = width/2 * (0.8 + Math.random() * 0.4);
                        points.push({
                            x: x + width/2 + Math.cos(angle) * radius,
                            y: y + height/2 + Math.sin(angle) * radius
                        });
                    }
                    
                    ctx.moveTo(points[0].x, points[0].y);
                    for (let j = 1; j < points.length; j++) {
                        ctx.lineTo(points[j].x, points[j].y);
                    }
                    ctx.closePath();
                    ctx.fill();
                    ctx.stroke();
                }
                break;
                
            case 'visualization':
                // Final visualization with labels and details
                ctx.strokeStyle = 'rgba(0, 255, 255, 0.9)';
                ctx.lineWidth = 2;
                for (let i = 0; i < 15; i++) {
                    const x = Math.floor(Math.random() * 24) * 10;
                    const y = Math.floor(Math.random() * 15) * 10;
                    const width = Math.floor(2 + Math.random() * 4) * 10;
                    const height = Math.floor(2 + Math.random() * 5) * 10;
                    
                    // Draw more precise shape with fill
                    ctx.fillStyle = 'rgba(0, 255, 255, 0.3)';
                    ctx.beginPath();
                    
                    // Create a more complex polygon
                    const points = [];
                    const sides = Math.floor(4 + Math.random() * 4);
                    for (let j = 0; j < sides; j++) {
                        const angle = (j / sides) * Math.PI * 2;
                        const radius = width/2 * (0.8 + Math.random() * 0.4);
                        points.push({
                            x: x + width/2 + Math.cos(angle) * radius,
                            y: y + height/2 + Math.sin(angle) * radius
                        });
                    }
                    
                    ctx.moveTo(points[0].x, points[0].y);
                    for (let j = 1; j < points.length; j++) {
                        ctx.lineTo(points[j].x, points[j].y);
                    }
                    ctx.closePath();
                    ctx.fill();
                    ctx.stroke();
                    
                    // Add building number
                    ctx.fillStyle = 'white';
                    ctx.font = '10px Arial';
                    ctx.fillText(`#${i+1}`, x + width/2 - 5, y + height/2 + 3);
                }
                
                // Add overlay info
                ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
                ctx.fillRect(0, 0, 300, 30);
                ctx.fillStyle = 'white';
                ctx.font = '12px Arial';
                ctx.fillText(`Buildings: ${this.metrics.buildingsDetected} | IoU: ${(this.metrics.accuracy * 100).toFixed(1)}%`, 10, 20);
                
                break;
        }
        
        // Store this as the processed image for this step
        this.maskImages[stepId] = canvas;
        
        // Display the image
        document.getElementById('processing-image').innerHTML = '';
        document.getElementById('processing-image').appendChild(canvas);
    }
    
    // Start auto updates for demo purposes
    startAutoUpdates() {
        if (!this.config.autoUpdate) return;
        
        setInterval(() => {
            // If pipeline is not running, randomly start it
            if (this.currentStepIndex === -1 && Math.random() < 0.2) {
                // Select a random city
                const citySelect = document.getElementById('pipeline-demo-city');
                const randomIndex = Math.floor(Math.random() * citySelect.options.length);
                citySelect.selectedIndex = randomIndex;
                
                // Load the city image
                this.loadCityImage();
                
                // Start processing after a delay
                setTimeout(() => this.startPipeline(), 1500);
            }
        }, this.config.updateInterval);
    }
}

// Initialize when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    if (document.getElementById('pipeline-visualization')) {
        window.pipelineVisualizer = new LivePipelineVisualizer('#pipeline-visualization');
    }
});