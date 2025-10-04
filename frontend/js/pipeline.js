// Pipeline Module - AI Building Footprint Extraction Interface
class PipelineModule {
    constructor() {
        this.currentMethod = 'rl-fusion';
        this.uploadedImage = null;
        this.processingJob = null;
        this.results = null;
        this.init();
    }

    init() {
        this.initImageUpload();
        this.initMethodSelection();
        this.initTabs();
        this.initProcessButton();
        console.log('🚀 Pipeline Module initialized');
    }

    initImageUpload() {
        const uploadArea = document.getElementById('uploadArea');
        const imageInput = document.getElementById('imageInput');

        if (!uploadArea || !imageInput) return;

        // Click to upload
        uploadArea.addEventListener('click', () => {
            imageInput.click();
        });

        // File input change
        imageInput.addEventListener('change', (e) => {
            this.handleFileSelection(e.target.files[0]);
        });

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            this.handleFileSelection(e.dataTransfer.files[0]);
        });
    }

    initMethodSelection() {
        const methodCards = document.querySelectorAll('.method-card');
        
        methodCards.forEach(card => {
            card.addEventListener('click', () => {
                // Remove active class from all cards
                methodCards.forEach(c => c.classList.remove('active'));
                
                // Add active class to clicked card
                card.classList.add('active');
                
                // Update selected method
                const radio = card.querySelector('input[type="radio"]');
                if (radio) {
                    radio.checked = true;
                    this.currentMethod = radio.value;
                    console.log('🧠 Selected method:', this.currentMethod);
                }
            });
        });
    }

    initTabs() {
        const tabBtns = document.querySelectorAll('.tab-btn');
        const tabPanes = document.querySelectorAll('.tab-pane');

        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const targetTab = btn.getAttribute('data-tab');
                
                // Update active button
                tabBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                // Update active pane
                tabPanes.forEach(pane => {
                    pane.classList.remove('active');
                    if (pane.id === targetTab) {
                        pane.classList.add('active');
                    }
                });
            });
        });
    }

    initProcessButton() {
        const processBtn = document.getElementById('processBtn');
        
        processBtn?.addEventListener('click', () => {
            if (this.uploadedImage && this.currentMethod) {
                this.processImage();
            }
        });
    }

    handleFileSelection(file) {
        if (!file) return;

        try {
            // Validate file
            window.geoaiApp.validateImageFile(file);
            
            this.uploadedImage = file;
            this.displayUploadedImage(file);
            this.enableProcessButton();
            
            window.geoaiApp.showNotification('Image uploaded successfully!', 'success');
        } catch (error) {
            window.geoaiApp.handleError(error, 'uploading image');
        }
    }

    displayUploadedImage(file) {
        const reader = new FileReader();
        const originalImage = document.getElementById('originalImage');
        
        reader.onload = (e) => {
            if (originalImage) {
                originalImage.innerHTML = `
                    <img src="${e.target.result}" alt="Uploaded Image" style="width: 100%; height: 100%; object-fit: cover; border-radius: 8px;">
                `;
            }
        };
        
        reader.readAsDataURL(file);
    }

    enableProcessButton() {
        const processBtn = document.getElementById('processBtn');
        if (processBtn) {
            processBtn.disabled = false;
            processBtn.classList.add('enabled');
        }
    }

    async processImage() {
        if (!this.uploadedImage || !this.currentMethod) {
            window.geoaiApp.showNotification('Please upload an image and select a method', 'warning');
            return;
        }

        try {
            this.showProcessingOverlay();
            
            // Prepare form data
            const formData = new FormData();
            formData.append('image', this.uploadedImage);
            formData.append('method', this.currentMethod);
            
            // Start processing
            const response = await fetch('/api/v1/process-image', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Processing failed: ${response.statusText}`);
            }

            const result = await response.json();
            
            if (result.job_id) {
                this.processingJob = result.job_id;
                this.pollProcessingStatus();
            } else {
                // Immediate result
                this.handleProcessingComplete(result);
            }
            
        } catch (error) {
            this.hideProcessingOverlay();
            window.geoaiApp.handleError(error, 'processing image');
        }
    }

    showProcessingOverlay() {
        const overlay = document.getElementById('processingOverlay');
        if (overlay) {
            overlay.classList.add('active');
            this.updateProcessingProgress(0, 'Initializing...');
        }
    }

    hideProcessingOverlay() {
        const overlay = document.getElementById('processingOverlay');
        if (overlay) {
            overlay.classList.remove('active');
        }
    }

    updateProcessingProgress(percent, status) {
        const progressFill = document.getElementById('processingProgress');
        const progressPercent = document.getElementById('processingPercent');
        const overlay = document.getElementById('processingOverlay');
        
        if (progressFill) {
            progressFill.style.width = `${percent}%`;
        }
        
        if (progressPercent) {
            progressPercent.textContent = `${Math.round(percent)}%`;
        }
        
        if (overlay && status) {
            const statusElement = overlay.querySelector('p');
            if (statusElement) {
                statusElement.textContent = status;
            }
        }
    }

    async pollProcessingStatus() {
        if (!this.processingJob) return;

        try {
            const response = await fetch(`/api/v1/job-status/${this.processingJob}`);
            const status = await response.json();
            
            this.updateProcessingProgress(status.progress || 0, status.status || 'Processing...');
            
            if (status.completed) {
                this.handleProcessingComplete(status.result);
            } else if (status.error) {
                throw new Error(status.error);
            } else {
                // Continue polling
                setTimeout(() => this.pollProcessingStatus(), 1000);
            }
            
        } catch (error) {
            this.hideProcessingOverlay();
            window.geoaiApp.handleError(error, 'checking processing status');
        }
    }

    handleProcessingComplete(result) {
        this.hideProcessingOverlay();
        this.results = result;
        
        // Display results
        this.displayResults(result);
        
        // Show success notification
        window.geoaiApp.showNotification('Image processing completed!', 'success');
        
        // Switch to visualization tab
        this.switchToTab('visualization');
    }

    displayResults(result) {
        // Display result image
        const resultImage = document.getElementById('resultImage');
        if (resultImage && result.result_image) {
            resultImage.innerHTML = `
                <img src="data:image/png;base64,${result.result_image}" 
                     alt="Detection Results" 
                     style="width: 100%; height: 100%; object-fit: cover; border-radius: 8px;">
            `;
        }

        // Update metrics
        this.updateMetrics(result.metrics || {});
        
        // Update analysis
        this.updateAnalysis(result.analysis || {});
    }

    updateMetrics(metrics) {
        // Detection Accuracy
        const accuracyValue = document.getElementById('accuracyValue');
        const accuracyProgress = document.getElementById('accuracyProgress');
        if (accuracyValue && metrics.accuracy !== undefined) {
            accuracyValue.textContent = `${(metrics.accuracy * 100).toFixed(1)}%`;
            if (accuracyProgress) {
                window.geoaiApp.animateProgressBar(accuracyProgress, metrics.accuracy * 100);
            }
        }

        // IoU Score
        const iouValue = document.getElementById('iouValue');
        const iouProgress = document.getElementById('iouProgress');
        if (iouValue && metrics.iou !== undefined) {
            iouValue.textContent = `${(metrics.iou * 100).toFixed(1)}%`;
            if (iouProgress) {
                window.geoaiApp.animateProgressBar(iouProgress, metrics.iou * 100);
            }
        }

        // Building Count
        const buildingCount = document.getElementById('buildingCount');
        if (buildingCount && metrics.building_count !== undefined) {
            window.geoaiApp.animateValue(buildingCount, 0, metrics.building_count);
        }

        // Processing Time
        const processingTime = document.getElementById('processingTime');
        if (processingTime && metrics.processing_time !== undefined) {
            processingTime.textContent = window.geoaiApp.formatTime(metrics.processing_time);
        }
    }

    updateAnalysis(analysis) {
        const analysisDetails = document.getElementById('analysisDetails');
        if (!analysisDetails) return;

        let html = '<div class="analysis-grid">';
        
        if (analysis.detections) {
            html += `
                <div class="analysis-item">
                    <h5>🏢 Building Detections</h5>
                    <p>Total buildings: <strong>${analysis.detections.total || 0}</strong></p>
                    <p>Average confidence: <strong>${((analysis.detections.avg_confidence || 0) * 100).toFixed(1)}%</strong></p>
                    <p>Large buildings: <strong>${analysis.detections.large_buildings || 0}</strong></p>
                    <p>Small buildings: <strong>${analysis.detections.small_buildings || 0}</strong></p>
                </div>
            `;
        }

        if (analysis.method_performance) {
            html += `
                <div class="analysis-item">
                    <h5>🧠 Method Performance</h5>
                    <p>Selected method: <strong>${analysis.method_performance.method || this.currentMethod}</strong></p>
                    <p>Precision: <strong>${((analysis.method_performance.precision || 0) * 100).toFixed(1)}%</strong></p>
                    <p>Recall: <strong>${((analysis.method_performance.recall || 0) * 100).toFixed(1)}%</strong></p>
                    <p>F1-Score: <strong>${((analysis.method_performance.f1_score || 0) * 100).toFixed(1)}%</strong></p>
                </div>
            `;
        }

        if (analysis.image_properties) {
            html += `
                <div class="analysis-item">
                    <h5>📸 Image Properties</h5>
                    <p>Resolution: <strong>${analysis.image_properties.width || 0} × ${analysis.image_properties.height || 0}</strong></p>
                    <p>File size: <strong>${window.geoaiApp.formatBytes(analysis.image_properties.size || 0)}</strong></p>
                    <p>Format: <strong>${analysis.image_properties.format || 'Unknown'}</strong></p>
                </div>
            `;
        }

        html += '</div>';
        analysisDetails.innerHTML = html;

        // Update performance chart if available
        this.updatePerformanceChart(analysis.performance_data);
    }

    updatePerformanceChart(performanceData) {
        const canvas = document.getElementById('performanceChart');
        if (!canvas || !performanceData) return;

        const chartConfig = {
            type: 'radar',
            data: {
                labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Speed', 'IoU'],
                datasets: [{
                    label: this.currentMethod.toUpperCase(),
                    data: [
                        (performanceData.accuracy || 0) * 100,
                        (performanceData.precision || 0) * 100,
                        (performanceData.recall || 0) * 100,
                        (performanceData.f1_score || 0) * 100,
                        (performanceData.speed || 0) * 20, // Normalize speed to 0-100 scale
                        (performanceData.iou || 0) * 100
                    ],
                    backgroundColor: 'rgba(52, 152, 219, 0.2)',
                    borderColor: 'rgba(52, 152, 219, 1)',
                    pointBackgroundColor: 'rgba(52, 152, 219, 1)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(52, 152, 219, 1)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            stepSize: 20
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    title: {
                        display: true,
                        text: 'Performance Metrics'
                    }
                }
            }
        };

        // Destroy existing chart and create new one
        window.geoaiApp.destroyChart('performanceChart');
        window.geoaiApp.createChart('performanceChart', chartConfig);
    }

    switchToTab(tabId) {
        const tabBtn = document.querySelector(`[data-tab="${tabId}"]`);
        if (tabBtn) {
            tabBtn.click();
        }
    }

    // Demo mode for when backend is not available
    async demoMode() {
        this.showProcessingOverlay();
        
        // Simulate processing steps
        const steps = [
            { progress: 10, status: 'Loading image...' },
            { progress: 25, status: 'Initializing Mask R-CNN...' },
            { progress: 45, status: 'Applying regularization techniques...' },
            { progress: 70, status: 'Running RL fusion...' },
            { progress: 90, status: 'Generating results...' },
            { progress: 100, status: 'Complete!' }
        ];

        for (const step of steps) {
            this.updateProcessingProgress(step.progress, step.status);
            await new Promise(resolve => setTimeout(resolve, 800));
        }

        // Generate demo results
        const demoResults = {
            result_image: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMzAwIiBoZWlnaHQ9IjMwMCIgZmlsbD0iIzM0OTVlYiIvPjx0ZXh0IHg9IjE1MCIgeT0iMTUwIiBmb250LWZhbWlseT0iQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTYiIGZpbGw9IndoaXRlIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+RGVtbyBSZXN1bHQ8L3RleHQ+PC9zdmc+',
            metrics: {
                accuracy: 0.957,
                iou: 0.891,
                building_count: 23,
                processing_time: 2.3
            },
            analysis: {
                detections: {
                    total: 23,
                    avg_confidence: 0.94,
                    large_buildings: 8,
                    small_buildings: 15
                },
                method_performance: {
                    method: this.currentMethod,
                    precision: 0.943,
                    recall: 0.971,
                    f1_score: 0.957
                },
                image_properties: {
                    width: 512,
                    height: 512,
                    size: this.uploadedImage?.size || 245760,
                    format: 'PNG'
                },
                performance_data: {
                    accuracy: 0.957,
                    precision: 0.943,
                    recall: 0.971,
                    f1_score: 0.957,
                    speed: 4.2,
                    iou: 0.891
                }
            }
        };

        this.handleProcessingComplete(demoResults);
    }

    // Export results
    exportResults() {
        if (!this.results) {
            window.geoaiApp.showNotification('No results to export', 'warning');
            return;
        }

        const exportData = {
            timestamp: new Date().toISOString(),
            method: this.currentMethod,
            metrics: this.results.metrics,
            analysis: this.results.analysis
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], { 
            type: 'application/json' 
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `geoai_results_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        window.geoaiApp.showNotification('Results exported successfully!', 'success');
    }

    // Reset pipeline
    reset() {
        this.uploadedImage = null;
        this.results = null;
        this.processingJob = null;

        // Reset UI
        const originalImage = document.getElementById('originalImage');
        const resultImage = document.getElementById('resultImage');
        const processBtn = document.getElementById('processBtn');

        if (originalImage) {
            originalImage.innerHTML = `
                <i class="fas fa-image"></i>
                <p>Upload an image to see results</p>
            `;
        }

        if (resultImage) {
            resultImage.innerHTML = `
                <i class="fas fa-building"></i>
                <p>Results will appear here</p>
            `;
        }

        if (processBtn) {
            processBtn.disabled = true;
            processBtn.classList.remove('enabled');
        }

        // Reset metrics
        ['accuracyValue', 'iouValue', 'buildingCount', 'processingTime'].forEach(id => {
            const element = document.getElementById(id);
            if (element) element.textContent = '--';
        });

        // Reset progress bars
        ['accuracyProgress', 'iouProgress'].forEach(id => {
            const element = document.getElementById(id);
            if (element) element.style.width = '0%';
        });

        window.geoaiApp.showNotification('Pipeline reset', 'info');
    }
}

// Initialize when DOM is loaded
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PipelineModule;
}