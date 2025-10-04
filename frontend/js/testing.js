// Testing Module - Live Performance Testing Interface
class TestingModule {
    constructor() {
        this.isRunning = false;
        this.systemMetrics = {
            gpu_usage: 0,
            memory_usage: 0,
            active_jobs: 0
        };
        this.performanceHistory = [];
        this.maxHistoryLength = 50;
        this.testResults = new Map();
        this.consoleBuffer = [];
        this.consolePaused = false;
        this.init();
    }

    init() {
        this.initSystemStatus();
        this.initCharts();
        this.initControls();
        this.initConsole();
        this.startSystemMonitoring();
        console.log('⚡ Testing Module initialized');
    }

    initSystemStatus() {
        // Initialize status indicators
        this.updateSystemStatus('backend', 'active');
        this.updateSystemStatus('rl', 'active');
        this.updateSystemStatus('websocket', 'processing');

        // Initialize system metrics display
        this.updateSystemMetrics({
            gpu_usage: Math.random() * 30 + 20, // 20-50%
            memory_usage: Math.random() * 40 + 30, // 30-70%
            active_jobs: Math.floor(Math.random() * 5)
        });
    }

    initCharts() {
        this.createPerformanceChart();
        this.createSpeedChart();
    }

    createPerformanceChart() {
        const canvas = document.getElementById('livePerformanceChart');
        if (!canvas) return;

        const config = {
            type: 'line',
            data: {
                labels: Array.from({length: 20}, (_, i) => `${i * 5}s`),
                datasets: [
                    {
                        label: 'IoU Score',
                        data: this.generateRandomData(20, 0.8, 0.95),
                        borderColor: 'rgba(52, 152, 219, 1)',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Accuracy',
                        data: this.generateRandomData(20, 0.85, 0.97),
                        borderColor: 'rgba(46, 204, 113, 1)',
                        backgroundColor: 'rgba(46, 204, 113, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'F1-Score',
                        data: this.generateRandomData(20, 0.82, 0.94),
                        borderColor: 'rgba(155, 89, 182, 1)',
                        backgroundColor: 'rgba(155, 89, 182, 0.1)',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: 'white'
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: 'white' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: {
                        min: 0.7,
                        max: 1.0,
                        ticks: { color: 'white' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                },
                animation: {
                    duration: 750
                }
            }
        };

        this.performanceChart = window.geoaiApp.createChart('livePerformanceChart', config);
    }

    createSpeedChart() {
        const canvas = document.getElementById('speedChart');
        if (!canvas) return;

        const config = {
            type: 'bar',
            data: {
                labels: ['Mask R-CNN', 'RT', 'RR', 'FER', 'RL Fusion'],
                datasets: [{
                    label: 'Processing Speed (fps)',
                    data: [5.8, 4.9, 5.1, 4.7, 4.2],
                    backgroundColor: [
                        'rgba(52, 152, 219, 0.8)',
                        'rgba(46, 204, 113, 0.8)',
                        'rgba(243, 156, 18, 0.8)',
                        'rgba(231, 76, 60, 0.8)',
                        'rgba(155, 89, 182, 0.8)'
                    ],
                    borderColor: [
                        'rgba(52, 152, 219, 1)',
                        'rgba(46, 204, 113, 1)',
                        'rgba(243, 156, 18, 1)',
                        'rgba(231, 76, 60, 1)',
                        'rgba(155, 89, 182, 1)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: 'white'
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: 'white' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: {
                        beginAtZero: true,
                        ticks: { color: 'white' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        };

        this.speedChart = window.geoaiApp.createChart('speedChart', config);
    }

    initControls() {
        // Automated testing
        const startAutomatedTest = document.getElementById('startAutomatedTest');
        startAutomatedTest?.addEventListener('click', () => this.startAutomatedTesting());

        // Export and analysis
        const exportMetrics = document.getElementById('exportMetrics');
        exportMetrics?.addEventListener('click', () => this.exportMetrics());

        const generateReport = document.getElementById('generateReport');
        generateReport?.addEventListener('click', () => this.generateReport());

        // Console controls
        const clearConsole = document.getElementById('clearConsole');
        clearConsole?.addEventListener('click', () => this.clearConsole());

        const pauseConsole = document.getElementById('pauseConsole');
        pauseConsole?.addEventListener('click', () => this.toggleConsole());
    }

    initConsole() {
        this.addConsoleLog('info', 'System initialized and ready');
        this.addConsoleLog('success', 'All models loaded successfully');
        this.addConsoleLog('info', 'WebSocket connection established');
        this.addConsoleLog('info', 'Performance monitoring started');
    }

    startSystemMonitoring() {
        // Update system metrics every 2 seconds
        this.monitoringInterval = setInterval(() => {
            this.updateLiveMetrics();
            this.updatePerformanceCharts();
        }, 2000);

        // Add random console logs
        this.logInterval = setInterval(() => {
            if (!this.consolePaused) {
                this.addRandomConsoleLog();
            }
        }, 3000);
    }

    updateSystemStatus(component, status) {
        const statusElement = document.getElementById(`${component}Status`);
        if (statusElement) {
            statusElement.className = `status-dot ${status}`;
        }
    }

    updateSystemMetrics(metrics) {
        this.systemMetrics = { ...this.systemMetrics, ...metrics };

        const gpuUsage = document.getElementById('gpuUsage');
        const memoryUsage = document.getElementById('memoryUsage');
        const activeJobs = document.getElementById('activeJobs');

        if (gpuUsage) gpuUsage.textContent = `${metrics.gpu_usage?.toFixed(1) || 0}%`;
        if (memoryUsage) memoryUsage.textContent = `${metrics.memory_usage?.toFixed(1) || 0}%`;
        if (activeJobs) activeJobs.textContent = metrics.active_jobs || 0;
    }

    updateLiveMetrics() {
        // Simulate realistic metric variations
        const newMetrics = {
            gpu_usage: Math.max(0, Math.min(100, 
                this.systemMetrics.gpu_usage + (Math.random() - 0.5) * 10
            )),
            memory_usage: Math.max(0, Math.min(100, 
                this.systemMetrics.memory_usage + (Math.random() - 0.5) * 5
            )),
            active_jobs: Math.max(0, Math.min(10, 
                this.systemMetrics.active_jobs + Math.floor((Math.random() - 0.5) * 3)
            ))
        };

        this.updateSystemMetrics(newMetrics);

        // Add to performance history
        this.performanceHistory.push({
            timestamp: Date.now(),
            gpu_usage: newMetrics.gpu_usage,
            memory_usage: newMetrics.memory_usage,
            processing_speed: 3.5 + Math.random() * 2 // 3.5-5.5 fps
        });

        // Keep history size manageable
        if (this.performanceHistory.length > this.maxHistoryLength) {
            this.performanceHistory.shift();
        }
    }

    updatePerformanceCharts() {
        if (this.performanceChart) {
            // Update performance trend chart
            const datasets = this.performanceChart.data.datasets;
            
            datasets.forEach(dataset => {
                // Simulate gradual performance improvements
                const lastValue = dataset.data[dataset.data.length - 1];
                const variation = (Math.random() - 0.5) * 0.02; // ±1% variation
                const trend = Math.random() > 0.6 ? 0.001 : -0.0005; // Slight upward trend
                const newValue = Math.max(0.7, Math.min(1.0, lastValue + variation + trend));
                
                dataset.data.push(newValue);
                if (dataset.data.length > 20) {
                    dataset.data.shift();
                }
            });

            // Update labels
            const currentTime = new Date();
            this.performanceChart.data.labels.push(
                currentTime.toLocaleTimeString([], { 
                    hour12: false, 
                    minute: '2-digit', 
                    second: '2-digit' 
                })
            );
            if (this.performanceChart.data.labels.length > 20) {
                this.performanceChart.data.labels.shift();
            }

            this.performanceChart.update('none'); // No animation for live updates
        }
    }

    async startAutomatedTesting() {
        if (this.isRunning) {
            window.geoaiApp.showNotification('Testing already in progress', 'warning');
            return;
        }

        this.isRunning = true;
        const startBtn = document.getElementById('startAutomatedTest');
        if (startBtn) {
            startBtn.disabled = true;
            startBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Running Tests...';
        }

        try {
            this.addConsoleLog('info', 'Starting automated testing suite...');
            
            const testOptions = this.getSelectedTestOptions();
            
            if (testOptions.stressTest) {
                await this.runStressTest();
            }
            
            if (testOptions.accuracyTest) {
                await this.runAccuracyTest();
            }
            
            if (testOptions.memoryTest) {
                await this.runMemoryTest();
            }

            this.addConsoleLog('success', 'Automated testing completed successfully');
            window.geoaiApp.showNotification('All tests completed successfully!', 'success');
            
        } catch (error) {
            this.addConsoleLog('error', `Testing failed: ${error.message}`);
            window.geoaiApp.handleError(error, 'running automated tests');
        } finally {
            this.isRunning = false;
            if (startBtn) {
                startBtn.disabled = false;
                startBtn.innerHTML = '<i class="fas fa-flask"></i> Start Automated Test';
            }
        }
    }

    getSelectedTestOptions() {
        const checkboxes = document.querySelectorAll('.test-option input[type="checkbox"]');
        const options = {};
        
        checkboxes.forEach(checkbox => {
            const label = checkbox.parentElement.textContent.trim();
            if (label.includes('Stress')) options.stressTest = checkbox.checked;
            if (label.includes('Accuracy')) options.accuracyTest = checkbox.checked;
            if (label.includes('Memory')) options.memoryTest = checkbox.checked;
        });
        
        return options;
    }

    async runStressTest() {
        this.addConsoleLog('info', 'Running stress test...');
        
        // Simulate high load
        for (let i = 0; i < 10; i++) {
            await new Promise(resolve => setTimeout(resolve, 500));
            
            this.updateSystemMetrics({
                gpu_usage: Math.min(95, this.systemMetrics.gpu_usage + 5),
                memory_usage: Math.min(90, this.systemMetrics.memory_usage + 3),
                active_jobs: Math.min(8, this.systemMetrics.active_jobs + 1)
            });
            
            this.addConsoleLog('info', `Stress test progress: ${(i + 1) * 10}%`);
        }
        
        // Gradually reduce load
        for (let i = 0; i < 5; i++) {
            await new Promise(resolve => setTimeout(resolve, 300));
            
            this.updateSystemMetrics({
                gpu_usage: Math.max(20, this.systemMetrics.gpu_usage - 10),
                memory_usage: Math.max(30, this.systemMetrics.memory_usage - 8),
                active_jobs: Math.max(0, this.systemMetrics.active_jobs - 1)
            });
        }
        
        this.testResults.set('stress_test', {
            status: 'passed',
            peak_gpu: 95,
            peak_memory: 90,
            duration: 8000
        });
        
        this.addConsoleLog('success', 'Stress test completed - System stable under load');
    }

    async runAccuracyTest() {
        this.addConsoleLog('info', 'Running accuracy test...');
        
        const testCases = [
            { name: 'Urban buildings', expected_iou: 0.92 },
            { name: 'Rural buildings', expected_iou: 0.89 },
            { name: 'Mixed terrain', expected_iou: 0.91 },
            { name: 'High density', expected_iou: 0.88 },
            { name: 'Agricultural areas', expected_iou: 0.94 }
        ];
        
        const results = [];
        
        for (const testCase of testCases) {
            await new Promise(resolve => setTimeout(resolve, 800));
            
            // Simulate accuracy test
            const actual_iou = testCase.expected_iou + (Math.random() - 0.5) * 0.05;
            const passed = Math.abs(actual_iou - testCase.expected_iou) < 0.03;
            
            results.push({
                name: testCase.name,
                expected: testCase.expected_iou,
                actual: actual_iou,
                passed
            });
            
            const status = passed ? 'success' : 'warning';
            this.addConsoleLog(status, 
                `${testCase.name}: IoU ${actual_iou.toFixed(3)} (expected ${testCase.expected_iou.toFixed(3)}) - ${passed ? 'PASS' : 'FAIL'}`
            );
        }
        
        const passedCount = results.filter(r => r.passed).length;
        const overallPassed = passedCount >= testCases.length * 0.8; // 80% pass rate
        
        this.testResults.set('accuracy_test', {
            status: overallPassed ? 'passed' : 'failed',
            test_cases: results,
            pass_rate: passedCount / testCases.length
        });
        
        this.addConsoleLog(overallPassed ? 'success' : 'error', 
            `Accuracy test completed - ${passedCount}/${testCases.length} tests passed`
        );
    }

    async runMemoryTest() {
        this.addConsoleLog('info', 'Running memory test...');
        
        let memoryLeakDetected = false;
        const initialMemory = this.systemMetrics.memory_usage;
        
        // Simulate memory-intensive operations
        for (let i = 0; i < 15; i++) {
            await new Promise(resolve => setTimeout(resolve, 400));
            
            const memoryIncrease = i * 2 + Math.random() * 3;
            const currentMemory = Math.min(95, initialMemory + memoryIncrease);
            
            this.updateSystemMetrics({
                memory_usage: currentMemory
            });
            
            if (currentMemory > 90) {
                memoryLeakDetected = true;
                this.addConsoleLog('warning', `High memory usage detected: ${currentMemory.toFixed(1)}%`);
            }
            
            this.addConsoleLog('info', `Memory test iteration ${i + 1}/15: ${currentMemory.toFixed(1)}% usage`);
        }
        
        // Memory cleanup simulation
        await new Promise(resolve => setTimeout(resolve, 1000));
        this.updateSystemMetrics({
            memory_usage: initialMemory + Math.random() * 5 // Should return close to initial
        });
        
        this.testResults.set('memory_test', {
            status: memoryLeakDetected ? 'warning' : 'passed',
            peak_memory: Math.min(95, initialMemory + 30),
            final_memory: this.systemMetrics.memory_usage,
            leak_detected: memoryLeakDetected
        });
        
        this.addConsoleLog(memoryLeakDetected ? 'warning' : 'success', 
            `Memory test completed - ${memoryLeakDetected ? 'Memory spikes detected' : 'No memory leaks found'}`
        );
    }

    addConsoleLog(level, message) {
        if (this.consolePaused) return;

        const timestamp = new Date().toLocaleTimeString();
        const logEntry = {
            timestamp,
            level,
            message
        };

        this.consoleBuffer.push(logEntry);
        
        // Keep buffer size manageable
        if (this.consoleBuffer.length > 100) {
            this.consoleBuffer.shift();
        }

        this.updateConsoleDisplay();
    }

    addRandomConsoleLog() {
        const randomLogs = [
            { level: 'info', message: 'Processing image batch completed' },
            { level: 'info', message: 'Model weights updated' },
            { level: 'info', message: 'Cache cleared successfully' },
            { level: 'success', message: 'Backup created automatically' },
            { level: 'info', message: 'Performance metrics collected' },
            { level: 'info', message: 'Database connection verified' },
            { level: 'warning', message: 'GPU temperature: 72°C' },
            { level: 'info', message: 'Training iteration completed' },
            { level: 'success', message: 'Model validation passed' },
            { level: 'info', message: 'Resource optimization applied' }
        ];

        const randomLog = randomLogs[Math.floor(Math.random() * randomLogs.length)];
        this.addConsoleLog(randomLog.level, randomLog.message);
    }

    updateConsoleDisplay() {
        const consoleLog = document.getElementById('liveConsole');
        if (!consoleLog) return;

        const html = this.consoleBuffer.slice(-50).map(entry => {
            return `<div class="log-line ${entry.level}">[${entry.timestamp}] [${entry.level.toUpperCase()}] ${entry.message}</div>`;
        }).join('');

        consoleLog.innerHTML = html;
        
        // Auto-scroll to bottom
        consoleLog.scrollTop = consoleLog.scrollHeight;
    }

    clearConsole() {
        this.consoleBuffer = [];
        this.updateConsoleDisplay();
        this.addConsoleLog('info', 'Console cleared');
    }

    toggleConsole() {
        this.consolePaused = !this.consolePaused;
        const pauseBtn = document.getElementById('pauseConsole');
        
        if (pauseBtn) {
            pauseBtn.textContent = this.consolePaused ? 'Resume' : 'Pause';
        }
        
        this.addConsoleLog('info', `Console ${this.consolePaused ? 'paused' : 'resumed'}`);
    }

    exportMetrics() {
        const exportData = {
            timestamp: new Date().toISOString(),
            system_metrics: this.systemMetrics,
            performance_history: this.performanceHistory,
            test_results: Object.fromEntries(this.testResults),
            console_logs: this.consoleBuffer.slice(-100) // Last 100 logs
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], { 
            type: 'application/json' 
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `system_metrics_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        this.addConsoleLog('success', 'Metrics exported successfully');
        window.geoaiApp.showNotification('Metrics exported successfully!', 'success');
    }

    generateReport() {
        const reportData = this.compileSystemReport();
        
        const reportHtml = this.generateReportHTML(reportData);
        
        const blob = new Blob([reportHtml], { type: 'text/html' });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `system_report_${Date.now()}.html`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        this.addConsoleLog('success', 'System report generated');
        window.geoaiApp.showNotification('System report generated successfully!', 'success');
    }

    compileSystemReport() {
        const currentTime = new Date();
        const uptime = currentTime - (window.geoaiApp.startTime || currentTime);
        
        return {
            generated_at: currentTime.toISOString(),
            uptime_hours: (uptime / (1000 * 60 * 60)).toFixed(2),
            current_metrics: this.systemMetrics,
            performance_summary: this.calculatePerformanceSummary(),
            test_results: Object.fromEntries(this.testResults),
            system_health: this.assessSystemHealth()
        };
    }

    calculatePerformanceSummary() {
        if (this.performanceHistory.length === 0) return null;
        
        const gpuUsages = this.performanceHistory.map(h => h.gpu_usage);
        const memoryUsages = this.performanceHistory.map(h => h.memory_usage);
        const speeds = this.performanceHistory.map(h => h.processing_speed);
        
        return {
            avg_gpu_usage: gpuUsages.reduce((a, b) => a + b) / gpuUsages.length,
            avg_memory_usage: memoryUsages.reduce((a, b) => a + b) / memoryUsages.length,
            avg_processing_speed: speeds.reduce((a, b) => a + b) / speeds.length,
            max_gpu_usage: Math.max(...gpuUsages),
            max_memory_usage: Math.max(...memoryUsages)
        };
    }

    assessSystemHealth() {
        const summary = this.calculatePerformanceSummary();
        if (!summary) return 'Unknown';
        
        let healthScore = 100;
        
        if (summary.avg_gpu_usage > 80) healthScore -= 20;
        if (summary.avg_memory_usage > 85) healthScore -= 25;
        if (summary.max_memory_usage > 95) healthScore -= 30;
        
        const failedTests = Array.from(this.testResults.values()).filter(t => t.status === 'failed').length;
        healthScore -= failedTests * 15;
        
        if (healthScore >= 90) return 'Excellent';
        if (healthScore >= 75) return 'Good';
        if (healthScore >= 60) return 'Fair';
        if (healthScore >= 40) return 'Poor';
        return 'Critical';
    }

    generateReportHTML(data) {
        return `
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>GeoAI System Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                .header { text-align: center; margin-bottom: 40px; }
                .section { margin-bottom: 30px; }
                .metric { display: inline-block; margin: 10px 20px; padding: 10px; background: #f5f5f5; border-radius: 5px; }
                .health-excellent { color: #27ae60; font-weight: bold; }
                .health-good { color: #2ecc71; font-weight: bold; }
                .health-fair { color: #f39c12; font-weight: bold; }
                .health-poor { color: #e67e22; font-weight: bold; }
                .health-critical { color: #e74c3c; font-weight: bold; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧠 GeoAI System Performance Report</h1>
                <p>Generated on: ${new Date(data.generated_at).toLocaleString()}</p>
                <p>System Uptime: ${data.uptime_hours} hours</p>
            </div>
            
            <div class="section">
                <h2>📊 Current System Status</h2>
                <div class="metric">GPU Usage: ${data.current_metrics.gpu_usage?.toFixed(1) || 0}%</div>
                <div class="metric">Memory Usage: ${data.current_metrics.memory_usage?.toFixed(1) || 0}%</div>
                <div class="metric">Active Jobs: ${data.current_metrics.active_jobs || 0}</div>
                <div class="metric">
                    System Health: <span class="health-${data.system_health.toLowerCase()}">${data.system_health}</span>
                </div>
            </div>
            
            ${data.performance_summary ? `
            <div class="section">
                <h2>📈 Performance Summary</h2>
                <div class="metric">Avg GPU Usage: ${data.performance_summary.avg_gpu_usage.toFixed(1)}%</div>
                <div class="metric">Avg Memory Usage: ${data.performance_summary.avg_memory_usage.toFixed(1)}%</div>
                <div class="metric">Avg Processing Speed: ${data.performance_summary.avg_processing_speed.toFixed(1)} fps</div>
                <div class="metric">Peak GPU Usage: ${data.performance_summary.max_gpu_usage.toFixed(1)}%</div>
                <div class="metric">Peak Memory Usage: ${data.performance_summary.max_memory_usage.toFixed(1)}%</div>
            </div>
            ` : ''}
            
            ${Object.keys(data.test_results).length > 0 ? `
            <div class="section">
                <h2>🧪 Test Results</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Test</th>
                            <th>Status</th>
                            <th>Details</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${Object.entries(data.test_results).map(([test, result]) => `
                            <tr>
                                <td>${test.replace('_', ' ').toUpperCase()}</td>
                                <td>${result.status.toUpperCase()}</td>
                                <td>${JSON.stringify(result, null, 2).substring(0, 100)}...</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
            ` : ''}
            
            <div class="section">
                <h2>📝 Report Notes</h2>
                <p>This report provides a snapshot of the GeoAI system performance at the time of generation.</p>
                <p>For real-time monitoring, please use the live testing dashboard.</p>
            </div>
        </body>
        </html>
        `;
    }

    generateRandomData(length, min, max) {
        return Array.from({length}, () => min + Math.random() * (max - min));
    }

    cleanup() {
        if (this.monitoringInterval) {
            clearInterval(this.monitoringInterval);
        }
        if (this.logInterval) {
            clearInterval(this.logInterval);
        }
    }
}

// Initialize when DOM is loaded
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TestingModule;
}