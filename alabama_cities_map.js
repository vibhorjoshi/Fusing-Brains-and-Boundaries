// Google Maps Integration for Alabama City Visualization
function initializeAlabamaMap() {
    // Check if Google Maps API is loaded
    if (typeof google === 'undefined' || typeof google.maps === 'undefined') {
        console.error('Google Maps API not loaded');
        document.getElementById('alabama-map-container').innerHTML = 
            '<div class="error-message">Google Maps API not loaded. Please check your API key.</div>';
        return;
    }

    // Alabama center coordinates
    const alabamaCenter = { lat: 32.7794, lng: -86.8287 };
    
    // Create the map centered on Alabama
    const map = new google.maps.Map(document.getElementById('alabama-map-container'), {
        center: alabamaCenter,
        zoom: 7,
        mapTypeId: 'hybrid',
        styles: [
            { elementType: 'geometry', stylers: [{ color: '#242f3e' }] },
            { elementType: 'labels.text.stroke', stylers: [{ color: '#242f3e' }] },
            { elementType: 'labels.text.fill', stylers: [{ color: '#746855' }] },
            { featureType: 'administrative.locality', elementType: 'labels.text.fill', stylers: [{ color: '#d59563' }] },
            { featureType: 'poi', elementType: 'labels.text.fill', stylers: [{ color: '#d59563' }] },
            { featureType: 'poi.park', elementType: 'geometry', stylers: [{ color: '#263c3f' }] },
            { featureType: 'poi.park', elementType: 'labels.text.fill', stylers: [{ color: '#6b9a76' }] },
            { featureType: 'road', elementType: 'geometry', stylers: [{ color: '#38414e' }] },
            { featureType: 'road', elementType: 'geometry.stroke', stylers: [{ color: '#212a37' }] },
            { featureType: 'road', elementType: 'labels.text.fill', stylers: [{ color: '#9ca5b3' }] },
            { featureType: 'road.highway', elementType: 'geometry', stylers: [{ color: '#746855' }] },
            { featureType: 'road.highway', elementType: 'geometry.stroke', stylers: [{ color: '#1f2835' }] },
            { featureType: 'road.highway', elementType: 'labels.text.fill', stylers: [{ color: '#f3d19c' }] },
            { featureType: 'transit', elementType: 'geometry', stylers: [{ color: '#2f3948' }] },
            { featureType: 'transit.station', elementType: 'labels.text.fill', stylers: [{ color: '#d59563' }] },
            { featureType: 'water', elementType: 'geometry', stylers: [{ color: '#17263c' }] },
            { featureType: 'water', elementType: 'labels.text.fill', stylers: [{ color: '#515c6d' }] },
            { featureType: 'water', elementType: 'labels.text.stroke', stylers: [{ color: '#17263c' }] }
        ]
    });

    // Alabama cities with performance data
    const alabamaCities = [
        { name: "Birmingham", lat: 33.5207, lng: -86.8025, buildings: 156421, accuracy: 0.912, processingTime: 26.4 },
        { name: "Montgomery", lat: 32.3792, lng: -86.3077, buildings: 98742, accuracy: 0.897, processingTime: 18.7 },
        { name: "Mobile", lat: 30.6954, lng: -88.0399, buildings: 87634, accuracy: 0.884, processingTime: 17.2 },
        { name: "Huntsville", lat: 34.7304, lng: -86.5861, buildings: 124563, accuracy: 0.923, processingTime: 22.9 },
        { name: "Tuscaloosa", lat: 33.2098, lng: -87.5692, buildings: 65432, accuracy: 0.901, processingTime: 14.8 },
        { name: "Auburn", lat: 32.6099, lng: -85.4808, buildings: 42387, accuracy: 0.894, processingTime: 10.5 },
        { name: "Dothan", lat: 31.2232, lng: -85.3905, buildings: 38921, accuracy: 0.878, processingTime: 9.7 },
        { name: "Hoover", lat: 33.4054, lng: -86.8114, buildings: 47532, accuracy: 0.907, processingTime: 12.3 },
        { name: "Decatur", lat: 34.6059, lng: -86.9834, buildings: 36281, accuracy: 0.889, processingTime: 8.9 },
        { name: "Madison", lat: 34.6992, lng: -86.7484, buildings: 41756, accuracy: 0.917, processingTime: 10.8 }
    ];

    const infoWindow = new google.maps.InfoWindow();
    const markers = [];
    
    // Create markers for each city with custom colors based on accuracy
    alabamaCities.forEach(city => {
        // Calculate color based on accuracy (green to red)
        const hue = (city.accuracy - 0.85) * 1200; // Convert 0.85-0.95 to 0-120 (red to green in HSL)
        const color = `hsl(${hue}, 100%, 50%)`;
        
        // Create marker
        const marker = new google.maps.Marker({
            position: { lat: city.lat, lng: city.lng },
            map: map,
            title: city.name,
            icon: {
                path: google.maps.SymbolPath.CIRCLE,
                scale: 10,
                fillColor: color,
                fillOpacity: 0.7,
                strokeWeight: 1,
                strokeColor: '#ffffff'
            }
        });
        
        // Create info window content
        const contentString = `
            <div class="info-window">
                <h3>${city.name}, Alabama</h3>
                <p><strong>Buildings Detected:</strong> ${city.buildings.toLocaleString()}</p>
                <p><strong>Detection Accuracy:</strong> ${(city.accuracy * 100).toFixed(1)}%</p>
                <p><strong>Processing Time:</strong> ${city.processingTime.toFixed(1)} seconds</p>
                <p><a href="#" onclick="loadCityData('${city.name}')">View Detailed Analysis</a></p>
            </div>
        `;
        
        // Add click event
        marker.addListener('click', () => {
            infoWindow.setContent(contentString);
            infoWindow.open(map, marker);
        });
        
        markers.push(marker);
    });
    
    // Create heatmap layer
    const heatmapData = alabamaCities.map(city => ({
        location: new google.maps.LatLng(city.lat, city.lng),
        weight: city.buildings / 10000 // Normalize weight
    }));
    
    const heatmap = new google.maps.visualization.HeatmapLayer({
        data: heatmapData,
        map: map,
        radius: 30,
        opacity: 0.7
    });
    
    // Add controls for toggling visualization
    const visualizationControl = document.getElementById('visualization-control');
    if (visualizationControl) {
        visualizationControl.addEventListener('change', function() {
            const value = this.value;
            
            // Hide all markers
            markers.forEach(marker => marker.setVisible(false));
            heatmap.setMap(null);
            
            if (value === 'markers') {
                markers.forEach(marker => marker.setVisible(true));
            } else if (value === 'heatmap') {
                heatmap.setMap(map);
            } else if (value === 'both') {
                markers.forEach(marker => marker.setVisible(true));
                heatmap.setMap(map);
            }
        });
    }
}

// Function to load city data when a city is clicked
function loadCityData(cityName) {
    document.getElementById('city-performance-title').innerText = `${cityName}, Alabama - Performance Metrics`;
    document.getElementById('selected-city').innerText = cityName;
    
    // Get metrics container
    const metricsContainer = document.getElementById('city-performance-metrics');
    
    // Simulate loading data
    metricsContainer.innerHTML = '<div class="loading-spinner"></div><p>Loading detailed metrics...</p>';
    
    // Fetch city data from API (simulated)
    setTimeout(() => {
        fetchCityPerformanceData(cityName)
            .then(data => {
                // Update metrics
                updateCityPerformanceMetrics(data);
                // Update performance charts
                updateCityPerformanceCharts(data);
            })
            .catch(error => {
                metricsContainer.innerHTML = `<div class="error-message">Error loading data: ${error.message}</div>`;
            });
    }, 1000);
    
    // Show the city data panel
    document.getElementById('city-data-panel').classList.add('visible');
}

// Simulated API call to get city performance data
async function fetchCityPerformanceData(cityName) {
    // This would be replaced with a real API call
    // For now, return simulated data
    return {
        cityName: cityName,
        buildingCount: Math.floor(40000 + Math.random() * 100000),
        accuracy: 0.85 + Math.random() * 0.1,
        processingTime: 8 + Math.random() * 20,
        modelPerformance: {
            baseline: 0.78 + Math.random() * 0.05,
            adaptive: 0.85 + Math.random() * 0.05,
            refined: 0.88 + Math.random() * 0.07
        },
        timeSeriesData: Array(12).fill(0).map((_, i) => ({
            timestamp: `2025-0${Math.floor(i/3)+1}-${(i%3+1)*10}`,
            baseline: 0.75 + Math.random() * 0.05 + i*0.005,
            adaptive: 0.82 + Math.random() * 0.05 + i*0.005,
            refined: 0.85 + Math.random() * 0.05 + i*0.006
        })),
        buildingSizes: {
            small: Math.floor(Math.random() * 40000),
            medium: Math.floor(Math.random() * 50000),
            large: Math.floor(Math.random() * 30000)
        },
        processingStages: [
            { name: "Image Acquisition", time: 1 + Math.random() * 2 },
            { name: "Preprocessing", time: 2 + Math.random() * 3 },
            { name: "Detection", time: 3 + Math.random() * 5 },
            { name: "Regularization", time: 2 + Math.random() * 4 },
            { name: "Refinement", time: 1 + Math.random() * 3 },
            { name: "Post-processing", time: 1 + Math.random() * 2 }
        ]
    };
}

// Update city performance metrics UI
function updateCityPerformanceMetrics(data) {
    const metricsContainer = document.getElementById('city-performance-metrics');
    
    metricsContainer.innerHTML = `
        <div class="metrics-grid">
            <div class="metric-card">
                <h4>Buildings Detected</h4>
                <div class="metric-value">${data.buildingCount.toLocaleString()}</div>
            </div>
            <div class="metric-card">
                <h4>Detection Accuracy</h4>
                <div class="metric-value">${(data.accuracy * 100).toFixed(1)}%</div>
            </div>
            <div class="metric-card">
                <h4>Processing Time</h4>
                <div class="metric-value">${data.processingTime.toFixed(1)}s</div>
            </div>
            <div class="metric-card">
                <h4>Improvement vs. Baseline</h4>
                <div class="metric-value">+${((data.accuracy - data.modelPerformance.baseline) * 100).toFixed(1)}%</div>
            </div>
        </div>
    `;
}

// Update city performance charts
function updateCityPerformanceCharts(data) {
    // Performance comparison chart
    const performanceCompCtx = document.getElementById('performance-comparison-chart').getContext('2d');
    
    if (window.performanceComparisonChart) {
        window.performanceComparisonChart.destroy();
    }
    
    window.performanceComparisonChart = new Chart(performanceCompCtx, {
        type: 'bar',
        data: {
            labels: ['Baseline', 'Adaptive Fusion', 'LapNet Refined'],
            datasets: [{
                label: 'IoU Score',
                data: [
                    data.modelPerformance.baseline,
                    data.modelPerformance.adaptive,
                    data.modelPerformance.refined
                ],
                backgroundColor: [
                    'rgba(255, 99, 132, 0.7)',
                    'rgba(54, 162, 235, 0.7)',
                    'rgba(75, 192, 192, 0.7)'
                ],
                borderColor: [
                    'rgb(255, 99, 132)',
                    'rgb(54, 162, 235)',
                    'rgb(75, 192, 192)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false,
                    min: Math.min(data.modelPerformance.baseline, 
                                  data.modelPerformance.adaptive, 
                                  data.modelPerformance.refined) - 0.05,
                    max: Math.max(data.modelPerformance.baseline, 
                                  data.modelPerformance.adaptive, 
                                  data.modelPerformance.refined) + 0.05
                }
            }
        }
    });
    
    // Time series chart
    const timeSeriesCtx = document.getElementById('time-series-chart').getContext('2d');
    
    if (window.timeSeriesChart) {
        window.timeSeriesChart.destroy();
    }
    
    window.timeSeriesChart = new Chart(timeSeriesCtx, {
        type: 'line',
        data: {
            labels: data.timeSeriesData.map(d => d.timestamp),
            datasets: [
                {
                    label: 'Baseline',
                    data: data.timeSeriesData.map(d => d.baseline),
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    fill: true,
                    tension: 0.3
                },
                {
                    label: 'Adaptive Fusion',
                    data: data.timeSeriesData.map(d => d.adaptive),
                    borderColor: 'rgb(54, 162, 235)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    fill: true,
                    tension: 0.3
                },
                {
                    label: 'LapNet Refined',
                    data: data.timeSeriesData.map(d => d.refined),
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    fill: true,
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false,
                    min: 0.7,
                    max: 1.0
                }
            }
        }
    });
    
    // Processing stages chart
    const stagesCtx = document.getElementById('processing-stages-chart').getContext('2d');
    
    if (window.stagesChart) {
        window.stagesChart.destroy();
    }
    
    window.stagesChart = new Chart(stagesCtx, {
        type: 'doughnut',
        data: {
            labels: data.processingStages.map(s => s.name),
            datasets: [{
                label: 'Processing Time (s)',
                data: data.processingStages.map(s => s.time),
                backgroundColor: [
                    'rgba(255, 99, 132, 0.7)',
                    'rgba(54, 162, 235, 0.7)',
                    'rgba(255, 206, 86, 0.7)',
                    'rgba(75, 192, 192, 0.7)',
                    'rgba(153, 102, 255, 0.7)',
                    'rgba(255, 159, 64, 0.7)'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
}

// Initialize on window load if Map container exists
window.addEventListener('load', function() {
    if (document.getElementById('alabama-map-container')) {
        // Wait for Google Maps API to load
        if (typeof google !== 'undefined' && typeof google.maps !== 'undefined') {
            initializeAlabamaMap();
        } else {
            console.log('Waiting for Google Maps API to load...');
            // Check again in 1 second
            setTimeout(() => {
                if (typeof google !== 'undefined' && typeof google.maps !== 'undefined') {
                    initializeAlabamaMap();
                } else {
                    console.error('Google Maps API failed to load within timeout');
                }
            }, 1000);
        }
    }
});