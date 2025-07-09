/**
 * Dashboard JavaScript - WebSocket Client with Real-time Chart Updates
 * Claude Code Proxy Dashboard
 */

class DashboardManager {
    constructor() {
        this.websocket = null;
        this.charts = {};
        this.metrics = {};
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.reconnectDelay = 1000;
        this.updateBuffer = [];
        this.preferences = this.loadPreferences();

        this.init();
    }

    init() {
        this.setupTheme();
        this.setupEventListeners();
        this.initializeCharts();
        this.showLoading();
        this.connectWebSocket();
    }

    // Theme Management
    setupTheme() {
        const theme = this.preferences.theme || 'light';
        document.documentElement.setAttribute('data-theme', theme);
        this.updateThemeToggle(theme);
    }

    toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', newTheme);
        this.updateThemeToggle(newTheme);
        this.savePreference('theme', newTheme);
        this.updateChartThemes(newTheme);
    }

    updateThemeToggle(theme) {
        const toggleIcon = document.querySelector('.theme-icon');
        toggleIcon.textContent = theme === 'dark' ? '☀️' : '🌙';
    }

    updateChartThemes(theme) {
        const isDark = theme === 'dark';
        const textColor = isDark ? '#ffffff' : '#212529';
        const gridColor = isDark ? '#404040' : '#dee2e6';

        Object.values(this.charts).forEach(chart => {
            if (chart.options.scales) {
                // Update scales colors
                Object.keys(chart.options.scales).forEach(scaleKey => {
                    const scale = chart.options.scales[scaleKey];
                    if (scale.ticks) scale.ticks.color = textColor;
                    if (scale.grid) scale.grid.color = gridColor;
                });
            }

            // Update legend colors
            if (chart.options.plugins && chart.options.plugins.legend) {
                chart.options.plugins.legend.labels.color = textColor;
            }

            chart.update('none');
        });
    }

    // Event Listeners
    setupEventListeners() {
        // Theme toggle
        document.getElementById('themeToggle').addEventListener('click', () => {
            this.toggleTheme();
        });

        // Controls
        document.getElementById('timeRange').addEventListener('change', (e) => {
            this.savePreference('timeRange', e.target.value);
            this.requestDataUpdate();
        });

        document.getElementById('apiFilter').addEventListener('change', (e) => {
            this.savePreference('apiFilter', e.target.value);
            this.requestDataUpdate();
        });

        document.getElementById('endpointFilter').addEventListener('change', (e) => {
            this.savePreference('endpointFilter', e.target.value);
            this.requestDataUpdate();
        });

        document.getElementById('autoRefresh').addEventListener('change', (e) => {
            this.savePreference('autoRefresh', e.target.checked);
        });

        // Export buttons
        document.getElementById('requestRateExport').addEventListener('click', () => {
            this.exportChart('requestRateChart');
        });

        document.getElementById('responseTimeExport').addEventListener('click', () => {
            this.exportChart('responseTimeChart');
        });

        document.getElementById('apiDistributionExport').addEventListener('click', () => {
            this.exportChart('apiDistributionChart');
        });

        document.getElementById('errorRateExport').addEventListener('click', () => {
            this.exportChart('errorRateChart');
        });

        document.getElementById('costBreakdownExport').addEventListener('click', () => {
            this.exportChart('costBreakdownChart');
        });

        // Footer actions
        document.getElementById('resetData').addEventListener('click', () => {
            this.resetDashboard();
        });

        document.getElementById('exportData').addEventListener('click', () => {
            this.exportAllData();
        });

        // Modal events
        document.getElementById('errorModalClose').addEventListener('click', () => {
            this.hideErrorModal();
        });

        document.getElementById('errorRetry').addEventListener('click', () => {
            this.hideErrorModal();
            this.connectWebSocket();
        });

        document.getElementById('errorDismiss').addEventListener('click', () => {
            this.hideErrorModal();
        });

        // Window events
        window.addEventListener('beforeunload', () => {
            if (this.websocket) {
                this.websocket.close();
            }
        });

        window.addEventListener('focus', () => {
            if (!this.isConnected && this.reconnectAttempts < this.maxReconnectAttempts) {
                this.connectWebSocket();
            }
        });
    }

    // WebSocket Management
    connectWebSocket() {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/metrics/ws`;

            this.websocket = new WebSocket(wsUrl);
            this.setupWebSocketHandlers();
            this.updateConnectionStatus('connecting');
        } catch (error) {
            console.error('WebSocket connection error:', error);
            this.showError('Failed to connect to WebSocket', error.message);
            this.scheduleReconnect();
        }
    }

    setupWebSocketHandlers() {
        this.websocket.onopen = () => {
            console.log('WebSocket connected');
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this.updateConnectionStatus('connected');
            this.hideLoading();
            this.sendInitialRequest();
        };

        this.websocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };

        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.showError('WebSocket Error', 'Connection error occurred');
        };

        this.websocket.onclose = (event) => {
            console.log('WebSocket disconnected:', event.code, event.reason);
            this.isConnected = false;
            this.updateConnectionStatus('disconnected');

            if (event.code !== 1000) { // Not a normal closure
                this.scheduleReconnect();
            }
        };
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'metrics_update':
                // Update both metrics and charts
                if (data.data) {
                    this.updateMetrics(data.data);
                }
                if (data.charts) {
                    this.updateChartData(data.charts);
                }
                break;
            case 'chart_data':
                this.updateChartData(data.payload);
                break;
            case 'error':
                this.showError('Server Error', data.message);
                break;
            default:
                console.warn('Unknown message type:', data.type);
        }
    }

    sendInitialRequest() {
        const message = {
            type: 'subscribe',
            filters: {
                timeRange: this.preferences.timeRange || '24h',
                apiFilter: this.preferences.apiFilter || 'all',
                endpointFilter: this.preferences.endpointFilter || 'all'
            }
        };
        this.sendWebSocketMessage(message);
    }

    sendWebSocketMessage(message) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify(message));
        }
    }

    requestDataUpdate() {
        const message = {
            type: 'update_filters',
            filters: {
                timeRange: document.getElementById('timeRange').value,
                apiFilter: document.getElementById('apiFilter').value,
                endpointFilter: document.getElementById('endpointFilter').value
            }
        };
        this.sendWebSocketMessage(message);
    }

    scheduleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts);
            this.reconnectAttempts++;

            setTimeout(() => {
                if (!this.isConnected) {
                    this.connectWebSocket();
                }
            }, delay);
        } else {
            this.showError('Connection Lost', 'Maximum reconnection attempts reached');
        }
    }

    updateConnectionStatus(status) {
        const statusIndicator = document.getElementById('connectionStatus');
        const statusText = document.getElementById('connectionText');

        statusIndicator.className = `status-indicator ${status}`;

        switch (status) {
            case 'connected':
                statusText.textContent = 'Connected';
                break;
            case 'connecting':
                statusText.textContent = 'Connecting...';
                break;
            case 'disconnected':
                statusText.textContent = 'Disconnected';
                break;
        }
    }

    // Chart Management
    initializeCharts() {
        this.initRequestRateChart();
        this.initResponseTimeChart();
        this.initApiDistributionChart();
        this.initErrorRateChart();
        this.initCostBreakdownChart();
    }

    initRequestRateChart() {
        const ctx = document.getElementById('requestRateChart').getContext('2d');
        this.charts.requestRate = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Requests/min',
                    data: [],
                    borderColor: '#007bff',
                    backgroundColor: 'rgba(0, 123, 255, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'minute'
                        },
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Requests per minute'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                animation: {
                    duration: 300
                }
            }
        });
    }

    initResponseTimeChart() {
        const ctx = document.getElementById('responseTimeChart').getContext('2d');
        this.charts.responseTime = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'P50',
                        data: [],
                        borderColor: '#28a745',
                        backgroundColor: 'rgba(40, 167, 69, 0.1)',
                        fill: false
                    },
                    {
                        label: 'P95',
                        data: [],
                        borderColor: '#ffc107',
                        backgroundColor: 'rgba(255, 193, 7, 0.1)',
                        fill: false
                    },
                    {
                        label: 'P99',
                        data: [],
                        borderColor: '#dc3545',
                        backgroundColor: 'rgba(220, 53, 69, 0.1)',
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'minute'
                        },
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Response Time (ms)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                animation: {
                    duration: 300
                }
            }
        });
    }

    initApiDistributionChart() {
        const ctx = document.getElementById('apiDistributionChart').getContext('2d');
        this.charts.apiDistribution = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: ['Anthropic', 'OpenAI'],
                datasets: [{
                    data: [0, 0],
                    backgroundColor: [
                        '#007bff',
                        '#6f42c1'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'bottom'
                    }
                },
                animation: {
                    duration: 300
                }
            }
        });
    }

    initErrorRateChart() {
        const ctx = document.getElementById('errorRateChart').getContext('2d');
        this.charts.errorRate = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Error Rate (%)',
                    data: [],
                    borderColor: '#dc3545',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'minute'
                        },
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Error Rate (%)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                animation: {
                    duration: 300
                }
            }
        });
    }

    initCostBreakdownChart() {
        const ctx = document.getElementById('costBreakdownChart').getContext('2d');
        this.charts.costBreakdown = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Cost (USD)',
                    data: [],
                    backgroundColor: [
                        '#007bff',
                        '#28a745',
                        '#ffc107',
                        '#dc3545',
                        '#6f42c1',
                        '#17a2b8'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Model'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Cost (USD)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                },
                animation: {
                    duration: 300
                }
            }
        });
    }

    // Data Updates
    updateMetrics(metrics) {
        this.metrics = metrics;

        // Update metric cards
        this.updateMetricCard('activeRequests', metrics.activeRequests, metrics.activeRequestsChange);
        this.updateMetricCard('totalRequests', metrics.totalRequests, metrics.totalRequestsChange);
        this.updateMetricCard('errorRate', `${metrics.errorRate.toFixed(1)}%`, metrics.errorRateChange);
        this.updateMetricCard('avgResponseTime', `${metrics.avgResponseTime.toFixed(0)}ms`, metrics.avgResponseTimeChange);
    }

    updateMetricCard(id, value, change) {
        const valueElement = document.getElementById(id);
        const changeElement = document.getElementById(`${id}Change`);

        if (valueElement) {
            valueElement.textContent = value;
        }

        if (changeElement && change !== undefined) {
            changeElement.textContent = change > 0 ? `+${change}` : `${change}`;
            changeElement.className = `metric-change ${change > 0 ? 'positive' : change < 0 ? 'negative' : 'neutral'}`;
        }
    }

    updateChartData(chartData) {
        Object.keys(chartData).forEach(chartName => {
            const chart = this.charts[chartName];
            if (chart && chartData[chartName]) {
                this.updateChart(chart, chartData[chartName]);
            }
        });
    }

    updateChart(chart, data) {
        if (!data) return;

        // Update labels
        if (data.labels) {
            chart.data.labels = data.labels;
        }

        // Update datasets
        if (data.datasets) {
            data.datasets.forEach((dataset, index) => {
                if (chart.data.datasets[index]) {
                    // Update data points
                    chart.data.datasets[index].data = dataset.data;

                    // Update label if provided
                    if (dataset.label) {
                        chart.data.datasets[index].label = dataset.label;
                    }

                    // Update colors if provided
                    if (dataset.borderColor) {
                        chart.data.datasets[index].borderColor = dataset.borderColor;
                    }
                    if (dataset.backgroundColor) {
                        chart.data.datasets[index].backgroundColor = dataset.backgroundColor;
                    }
                } else {
                    // Add new dataset if it doesn't exist
                    chart.data.datasets.push(dataset);
                }
            });

            // Remove extra datasets if new data has fewer
            while (chart.data.datasets.length > data.datasets.length) {
                chart.data.datasets.pop();
            }
        }

        // Update chart with animation disabled for smoother updates
        chart.update('none');
    }

    // Export Functions
    exportChart(chartId) {
        const canvas = document.getElementById(chartId);
        const link = document.createElement('a');
        link.download = `${chartId}_${new Date().toISOString().split('T')[0]}.png`;
        link.href = canvas.toDataURL();
        link.click();
    }

    exportAllData() {
        const data = {
            metrics: this.metrics,
            timestamp: new Date().toISOString(),
            preferences: this.preferences
        };

        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `dashboard_data_${new Date().toISOString().split('T')[0]}.json`;
        link.click();
        URL.revokeObjectURL(url);
    }

    // Utility Functions
    resetDashboard() {
        if (confirm('Are you sure you want to reset the dashboard? This will clear all current data.')) {
            this.sendWebSocketMessage({ type: 'reset_dashboard' });

            // Reset charts
            Object.values(this.charts).forEach(chart => {
                chart.data.labels = [];
                chart.data.datasets.forEach(dataset => {
                    dataset.data = [];
                });
                chart.update();
            });

            // Reset metrics
            this.updateMetricCard('activeRequests', '-', '-');
            this.updateMetricCard('totalRequests', '-', '-');
            this.updateMetricCard('errorRate', '-', '-');
            this.updateMetricCard('avgResponseTime', '-', '-');
        }
    }

    showLoading() {
        const overlay = document.getElementById('loadingOverlay');
        overlay.classList.add('show');
    }

    hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        overlay.classList.remove('show');
    }

    showError(title, message) {
        const modal = document.getElementById('errorModal');
        const messageElement = document.getElementById('errorMessage');

        messageElement.textContent = message;
        modal.classList.add('show');
    }

    hideErrorModal() {
        const modal = document.getElementById('errorModal');
        modal.classList.remove('show');
    }

    // Preferences Management
    loadPreferences() {
        try {
            const stored = localStorage.getItem('dashboard_preferences');
            return stored ? JSON.parse(stored) : {
                theme: 'light',
                timeRange: '24h',
                apiFilter: 'all',
                endpointFilter: 'all',
                autoRefresh: true
            };
        } catch (error) {
            console.error('Error loading preferences:', error);
            return {
                theme: 'light',
                timeRange: '24h',
                apiFilter: 'all',
                endpointFilter: 'all',
                autoRefresh: true
            };
        }
    }

    savePreference(key, value) {
        this.preferences[key] = value;
        try {
            localStorage.setItem('dashboard_preferences', JSON.stringify(this.preferences));
        } catch (error) {
            console.error('Error saving preferences:', error);
        }
    }

    // Initialize controls with saved preferences
    initializeControls() {
        document.getElementById('timeRange').value = this.preferences.timeRange || '24h';
        document.getElementById('apiFilter').value = this.preferences.apiFilter || 'all';
        document.getElementById('endpointFilter').value = this.preferences.endpointFilter || 'all';
        document.getElementById('autoRefresh').checked = this.preferences.autoRefresh !== false;
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const dashboard = new DashboardManager();
    dashboard.initializeControls();
});
