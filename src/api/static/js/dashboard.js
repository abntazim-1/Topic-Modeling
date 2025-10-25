/**
 * Topic Modeling Dashboard JavaScript
 * ==================================
 * 
 * Additional JavaScript functionality for the Topic Modeling Dashboard.
 * This file provides enhanced user interactions and utilities.
 */

// Dashboard utilities and enhancements
class TopicModelingDashboard {
    constructor() {
        this.apiBaseUrl = window.location.origin;
        this.modelsStatus = {
            lda: false,
            nmf: false
        };
        this.predictionHistory = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.checkAPIHealth();
        this.loadModelsStatus();
        this.loadPredictionHistory();
    }

    setupEventListeners() {
        // Auto-save prediction history
        document.addEventListener('beforeunload', () => {
            this.savePredictionHistory();
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                this.submitPrediction();
            }
        });

        // Text area auto-resize
        const textArea = document.getElementById('input-text');
        if (textArea) {
            textArea.addEventListener('input', this.autoResizeTextarea);
        }
    }

    autoResizeTextarea(event) {
        const textarea = event.target;
        textarea.style.height = 'auto';
        textarea.style.height = textarea.scrollHeight + 'px';
    }

    async checkAPIHealth() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            const data = await response.json();
            
            if (data.status === 'healthy') {
                this.updateAPIStatus('connected');
                this.modelsStatus = data.models_loaded;
                this.updateModelStatus();
            } else {
                throw new Error('API not healthy');
            }
        } catch (error) {
            this.updateAPIStatus('disconnected');
            this.showError('Unable to connect to API server');
        }
    }

    updateAPIStatus(status) {
        const statusElement = document.getElementById('api-status');
        if (statusElement) {
            if (status === 'connected') {
                statusElement.innerHTML = '<i class="fas fa-circle text-success me-1"></i>Connected';
            } else {
                statusElement.innerHTML = '<i class="fas fa-circle text-danger me-1"></i>Disconnected';
            }
        }
    }

    async loadModelsStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/models`);
            const data = await response.json();
            
            this.modelsStatus.lda = data.models.lda.loaded;
            this.modelsStatus.nmf = data.models.nmf.loaded;
            this.updateModelStatus();
        } catch (error) {
            console.error('Failed to load models status:', error);
        }
    }

    updateModelStatus() {
        const ldaStatus = document.getElementById('lda-status');
        const nmfStatus = document.getElementById('nmf-status');
        
        if (ldaStatus) {
            if (this.modelsStatus.lda) {
                ldaStatus.textContent = 'Loaded';
                ldaStatus.className = 'model-status status-loaded';
            } else {
                ldaStatus.textContent = 'Not Loaded';
                ldaStatus.className = 'model-status status-not-loaded';
            }
        }
        
        if (nmfStatus) {
            if (this.modelsStatus.nmf) {
                nmfStatus.textContent = 'Loaded';
                nmfStatus.className = 'model-status status-loaded';
            } else {
                nmfStatus.textContent = 'Not Loaded';
                nmfStatus.className = 'model-status status-not-loaded';
            }
        }
    }

    async loadModel(modelType) {
        this.showLoading();
        try {
            const response = await fetch(`${this.apiBaseUrl}/models/${modelType}/load`, {
                method: 'POST'
            });
            
            if (response.ok) {
                this.modelsStatus[modelType] = true;
                this.updateModelStatus();
                this.showSuccess(`${modelType.toUpperCase()} model loaded successfully!`);
            } else {
                const error = await response.json();
                throw new Error(error.error || 'Failed to load model');
            }
        } catch (error) {
            this.showError(`Failed to load ${modelType.toUpperCase()} model: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    async getModelTopics(modelType) {
        if (!this.modelsStatus[modelType]) {
            this.showError(`${modelType.toUpperCase()} model is not loaded. Please load it first.`);
            return;
        }
        
        this.showLoading();
        try {
            const response = await fetch(`${this.apiBaseUrl}/models/${modelType}/topics?num_words=10`);
            const data = await response.json();
            
            if (response.ok) {
                this.displayTopics(data.topics, modelType);
            } else {
                throw new Error(data.error || 'Failed to get topics');
            }
        } catch (error) {
            this.showError(`Failed to get topics: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    displayTopics(topics, modelType) {
        const modal = new bootstrap.Modal(document.getElementById('topicsModal'));
        const modalBody = document.getElementById('topics-modal-body');
        
        let html = `<h5>${modelType.toUpperCase()} Model Topics</h5>`;
        topics.forEach((topic, index) => {
            html += `
                <div class="topic-item">
                    <h6>Topic ${topic.topic_id}</h6>
                    <div class="d-flex flex-wrap gap-1">
                        ${topic.words.map(word => 
                            `<span class="badge bg-primary">${word[0]} (${word[1].toFixed(3)})</span>`
                        ).join('')}
                    </div>
                </div>
            `;
        });
        
        modalBody.innerHTML = html;
        modal.show();
    }

    async predictTopics(text, modelType, numTopics) {
        this.showLoading();
        try {
            const response = await fetch(`${this.apiBaseUrl}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: text,
                    model_type: modelType,
                    num_topics: numTopics
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.displayPredictionResults(data);
                this.addToPredictionHistory(data);
            } else {
                throw new Error(data.error || 'Prediction failed');
            }
        } catch (error) {
            this.showError(`Prediction failed: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    displayPredictionResults(data) {
        const resultsContainer = document.getElementById('prediction-results');
        const resultSection = document.querySelector('.result-section');
        
        let html = `
            <div class="card">
                <div class="card-header d-flex justify-content-between align-items-center">
                    <h5 class="mb-0">Analysis Results</h5>
                    <div>
                        <span class="badge bg-primary">${data.model_type.toUpperCase()}</span>
                        <span class="badge bg-success">${(data.prediction.confidence * 100).toFixed(1)}% Confidence</span>
                    </div>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-6">
                            <h6>Input Text:</h6>
                            <p class="text-muted">${data.input_text.substring(0, 200)}${data.input_text.length > 200 ? '...' : ''}</p>
                        </div>
                        <div class="col-md-6">
                            <h6>Model Information:</h6>
                            <p><strong>Model:</strong> ${data.model_type.toUpperCase()}</p>
                            <p><strong>Dominant Topic:</strong> ${data.prediction.dominant_topic}</p>
                            <p><strong>Confidence:</strong> ${(data.prediction.confidence * 100).toFixed(1)}%</p>
                        </div>
                    </div>
                    <hr>
                    <h6>Topic Distribution:</h6>
                    <div class="row">
        `;
        
        data.prediction.topics.forEach((topic, index) => {
            const percentage = (topic.probability * 100).toFixed(1);
            html += `
                <div class="col-md-6 mb-3">
                    <div class="card">
                        <div class="card-body">
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <h6 class="mb-0">Topic ${topic.topic_id}</h6>
                                <span class="badge bg-info">${percentage}%</span>
                            </div>
                            <div class="progress mb-2" style="height: 6px;">
                                <div class="progress-bar" style="width: ${percentage}%"></div>
                            </div>
                            <div class="d-flex flex-wrap gap-1">
                                ${topic.topic_words.map(word => 
                                    `<span class="badge bg-secondary">${word[0]} (${word[1].toFixed(3)})</span>`
                                ).join('')}
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });
        
        html += `
                    </div>
                </div>
            </div>
        `;
        
        resultsContainer.innerHTML = html;
        resultSection.style.display = 'block';
        resultSection.scrollIntoView({ behavior: 'smooth' });
    }

    addToPredictionHistory(prediction) {
        this.predictionHistory.push({
            timestamp: new Date().toISOString(),
            input_text: prediction.input_text,
            model_type: prediction.model_type,
            dominant_topic: prediction.prediction.dominant_topic,
            confidence: prediction.prediction.confidence
        });
        
        // Keep only last 10 predictions
        if (this.predictionHistory.length > 10) {
            this.predictionHistory = this.predictionHistory.slice(-10);
        }
    }

    loadPredictionHistory() {
        const history = localStorage.getItem('topicModelingHistory');
        if (history) {
            this.predictionHistory = JSON.parse(history);
        }
    }

    savePredictionHistory() {
        localStorage.setItem('topicModelingHistory', JSON.stringify(this.predictionHistory));
    }

    showLoading() {
        document.querySelector('.loading-spinner').style.display = 'block';
    }

    hideLoading() {
        document.querySelector('.loading-spinner').style.display = 'none';
    }

    showError(message) {
        const errorElement = document.getElementById('error-message');
        const alertElement = document.querySelector('.error-alert');
        
        if (errorElement) {
            errorElement.textContent = message;
        }
        if (alertElement) {
            alertElement.style.display = 'block';
            setTimeout(() => {
                alertElement.style.display = 'none';
            }, 5000);
        }
    }

    showSuccess(message) {
        // Create a success notification
        const notification = document.createElement('div');
        notification.className = 'alert alert-success alert-dismissible fade show position-fixed';
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; max-width: 400px;';
        notification.innerHTML = `
            <i class="fas fa-check-circle me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }

    scrollToSection(sectionId) {
        document.getElementById(sectionId).scrollIntoView({ behavior: 'smooth' });
    }

    submitPrediction() {
        const form = document.getElementById('prediction-form');
        if (form) {
            form.dispatchEvent(new Event('submit'));
        }
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.dashboard = new TopicModelingDashboard();
});

// Global functions for HTML onclick handlers
function loadModel(modelType) {
    if (window.dashboard) {
        window.dashboard.loadModel(modelType);
    }
}

function getModelTopics(modelType) {
    if (window.dashboard) {
        window.dashboard.getModelTopics(modelType);
    }
}

function scrollToSection(sectionId) {
    if (window.dashboard) {
        window.dashboard.scrollToSection(sectionId);
    }
}
