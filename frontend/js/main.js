// Fusing Brains & Boundaries - Main JavaScript Module
class GeoAIApp {
    constructor() {
        this.currentSection = 'home';
        this.apiEndpoint = '/api/v1';
        this.websocket = null;
        this.charts = {};
        this.init();
    }

    init() {
        this.initNavigation();
        this.initSmoothScrolling();
        this.initIntersectionObserver();
        this.initWebSocket();
        console.log('🧠 GeoAI App initialized successfully');
    }

    // Navigation functionality
    initNavigation() {
        const hamburger = document.querySelector('.hamburger');
        const navMenu = document.querySelector('.nav-menu');
        const navLinks = document.querySelectorAll('.nav-link');

        // Mobile menu toggle
        hamburger?.addEventListener('click', () => {
            hamburger.classList.toggle('active');
            navMenu.classList.toggle('active');
        });

        // Close mobile menu when clicking on nav links
        navLinks.forEach(link => {
            link.addEventListener('click', () => {
                hamburger?.classList.remove('active');
                navMenu?.classList.remove('active');
            });
        });

        // Active navigation highlight
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const targetId = link.getAttribute('href').substring(1);
                this.scrollToSection(targetId);
                this.updateActiveNav(link);
            });
        });
    }

    initSmoothScrolling() {
        // Smooth scroll for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', (e) => {
                e.preventDefault();
                const target = document.querySelector(anchor.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    }

    initIntersectionObserver() {
        const sections = document.querySelectorAll('.section');
        const navLinks = document.querySelectorAll('.nav-link');

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const sectionId = entry.target.id;
                    this.currentSection = sectionId;
                    
                    // Update active navigation
                    navLinks.forEach(link => {
                        link.classList.remove('active');
                        if (link.getAttribute('href') === `#${sectionId}`) {
                            link.classList.add('active');
                        }
                    });
                }
            });
        }, {
            threshold: 0.3
        });

        sections.forEach(section => observer.observe(section));
    }

    initWebSocket() {
        try {
            this.websocket = new WebSocket('ws://localhost:8000/ws');
            
            this.websocket.onopen = () => {
                console.log('🔌 WebSocket connected');
                this.updateConnectionStatus('connected');
            };

            this.websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            };

            this.websocket.onclose = () => {
                console.log('🔌 WebSocket disconnected');
                this.updateConnectionStatus('disconnected');
                // Attempt to reconnect after 5 seconds
                setTimeout(() => this.initWebSocket(), 5000);
            };

            this.websocket.onerror = (error) => {
                console.error('🔌 WebSocket error:', error);
                this.updateConnectionStatus('error');
            };
        } catch (error) {
            console.error('🔌 WebSocket initialization failed:', error);
            this.updateConnectionStatus('error');
        }
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'processing_update':
                this.updateProcessingProgress(data.progress, data.status);
                break;
            case 'results':
                this.displayResults(data.results);
                break;
            case 'system_metrics':
                this.updateSystemMetrics(data.metrics);
                break;
            case 'console_log':
                this.addConsoleLog(data.level, data.message);
                break;
            default:
                console.log('📨 Unknown message type:', data.type);
        }
    }

    updateConnectionStatus(status) {
        const statusElements = {
            backend: document.getElementById('backendStatus'),
            websocket: document.getElementById('wsStatus')
        };

        Object.values(statusElements).forEach(element => {
            if (element) {
                element.className = `status-dot ${status}`;
            }
        });
    }

    scrollToSection(sectionId) {
        const section = document.getElementById(sectionId);
        if (section) {
            section.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    }

    updateActiveNav(activeLink) {
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        activeLink.classList.add('active');
    }

    // Utility methods
    formatNumber(num, decimals = 2) {
        return parseFloat(num).toFixed(decimals);
    }

    formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    formatTime(seconds) {
        if (seconds < 60) return `${seconds.toFixed(1)}s`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${(seconds % 60).toFixed(0)}s`;
        return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-${this.getNotificationIcon(type)}"></i>
                <span>${message}</span>
                <button class="notification-close">&times;</button>
            </div>
        `;

        // Add to page
        document.body.appendChild(notification);

        // Auto remove after 5 seconds
        setTimeout(() => {
            notification.remove();
        }, 5000);

        // Manual close
        notification.querySelector('.notification-close').addEventListener('click', () => {
            notification.remove();
        });
    }

    getNotificationIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };
        return icons[type] || 'info-circle';
    }

    // API methods
    async apiRequest(endpoint, method = 'GET', data = null) {
        try {
            const config = {
                method,
                headers: {
                    'Content-Type': 'application/json',
                }
            };

            if (data) {
                config.body = JSON.stringify(data);
            }

            const response = await fetch(`${this.apiEndpoint}${endpoint}`, config);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('🔴 API Request failed:', error);
            this.showNotification(`API Error: ${error.message}`, 'error');
            throw error;
        }
    }

    // Chart utilities
    createChart(canvasId, config) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.warn(`🔴 Canvas with id '${canvasId}' not found`);
            return null;
        }

        const ctx = canvas.getContext('2d');
        this.charts[canvasId] = new Chart(ctx, config);
        return this.charts[canvasId];
    }

    updateChart(canvasId, newData) {
        const chart = this.charts[canvasId];
        if (chart) {
            chart.data = newData;
            chart.update();
        }
    }

    destroyChart(canvasId) {
        const chart = this.charts[canvasId];
        if (chart) {
            chart.destroy();
            delete this.charts[canvasId];
        }
    }

    // Animation utilities
    animateValue(element, start, end, duration = 1000) {
        const startTimestamp = performance.now();
        
        const step = (timestamp) => {
            const elapsed = timestamp - startTimestamp;
            const progress = Math.min(elapsed / duration, 1);
            
            const current = start + (end - start) * this.easeOutCubic(progress);
            element.textContent = this.formatNumber(current);
            
            if (progress < 1) {
                requestAnimationFrame(step);
            }
        };
        
        requestAnimationFrame(step);
    }

    animateProgressBar(element, targetWidth, duration = 1000) {
        const startWidth = 0;
        const startTimestamp = performance.now();
        
        const step = (timestamp) => {
            const elapsed = timestamp - startTimestamp;
            const progress = Math.min(elapsed / duration, 1);
            
            const currentWidth = startWidth + (targetWidth - startWidth) * this.easeOutCubic(progress);
            element.style.width = `${currentWidth}%`;
            
            if (progress < 1) {
                requestAnimationFrame(step);
            }
        };
        
        requestAnimationFrame(step);
    }

    easeOutCubic(t) {
        return 1 - Math.pow(1 - t, 3);
    }

    // Loading states
    showLoading(element, text = 'Loading...') {
        if (element) {
            element.innerHTML = `
                <div class="loading-spinner"></div>
                <span>${text}</span>
            `;
        }
    }

    hideLoading(element, originalContent = '') {
        if (element) {
            element.innerHTML = originalContent;
        }
    }

    // Error handling
    handleError(error, context = '') {
        console.error(`🔴 Error ${context}:`, error);
        this.showNotification(`Error ${context}: ${error.message}`, 'error');
    }

    // Data validation
    validateImageFile(file) {
        const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
        const maxSize = 10 * 1024 * 1024; // 10MB

        if (!validTypes.includes(file.type)) {
            throw new Error('Invalid file type. Please upload JPEG, PNG, GIF, or WebP images.');
        }

        if (file.size > maxSize) {
            throw new Error('File too large. Please upload images smaller than 10MB.');
        }

        return true;
    }

    // Local storage utilities
    saveToStorage(key, data) {
        try {
            localStorage.setItem(`geoai_${key}`, JSON.stringify(data));
        } catch (error) {
            console.warn('🔴 Failed to save to localStorage:', error);
        }
    }

    loadFromStorage(key, defaultValue = null) {
        try {
            const stored = localStorage.getItem(`geoai_${key}`);
            return stored ? JSON.parse(stored) : defaultValue;
        } catch (error) {
            console.warn('🔴 Failed to load from localStorage:', error);
            return defaultValue;
        }
    }

    clearStorage(key = null) {
        if (key) {
            localStorage.removeItem(`geoai_${key}`);
        } else {
            // Clear all GeoAI related storage
            Object.keys(localStorage).forEach(key => {
                if (key.startsWith('geoai_')) {
                    localStorage.removeItem(key);
                }
            });
        }
    }

    // Performance monitoring
    measurePerformance(name, fn) {
        const start = performance.now();
        const result = fn();
        const end = performance.now();
        
        console.log(`⚡ ${name} took ${(end - start).toFixed(2)}ms`);
        return result;
    }

    async measureAsyncPerformance(name, fn) {
        const start = performance.now();
        const result = await fn();
        const end = performance.now();
        
        console.log(`⚡ ${name} took ${(end - start).toFixed(2)}ms`);
        return result;
    }

    // Debounce utility
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // Throttle utility
    throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        }
    }
}

// Global utility functions
window.scrollToSection = function(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        section.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
};

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.geoaiApp = new GeoAIApp();
    
    // Initialize all modules
    if (typeof PipelineModule !== 'undefined') {
        window.pipelineModule = new PipelineModule();
    }
    
    if (typeof AgricultureModule !== 'undefined') {
        window.agricultureModule = new AgricultureModule();
    }
    
    if (typeof TestingModule !== 'undefined') {
        window.testingModule = new TestingModule();
    }
    
    if (typeof ChartsModule !== 'undefined') {
        window.chartsModule = new ChartsModule();
    }
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GeoAIApp;
}