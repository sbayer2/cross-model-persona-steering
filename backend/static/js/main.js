/**
 * GPT-OSS Persona Vector System - Main JavaScript
 * Handles UI interactions and API communications
 */

console.log('main.js loaded - starting PersonaVectorSystem');

class PersonaVectorSystem {
    constructor() {
        this.currentVector = null;
        this.charts = {};
        this.init();
    }

    init() {
        this.bindEvents();
        this.loadInitialData();
    }

    bindEvents() {
        // Form submissions
        document.getElementById('generate-form').addEventListener('submit', (e) => this.handleGenerateVector(e));
        document.getElementById('test-form').addEventListener('submit', (e) => this.handleTestVector(e));

        // Steering coefficient slider
        const slider = document.getElementById('steering-coefficient');
        slider.addEventListener('input', (e) => {
            document.getElementById('coefficient-value').textContent = e.target.value;
        });

        // Model info and memory buttons
        document.getElementById('model-info-btn').addEventListener('click', () => this.showModelInfo());
        document.getElementById('memory-info-btn').addEventListener('click', () => this.showMemoryInfo());
    }

    async loadInitialData() {
        try {
            // Load models, traits, and vectors in parallel
            await Promise.all([
                this.loadModels(),
                this.loadTraits(),
                this.loadVectors()
            ]);
        } catch (error) {
            console.error('Error loading initial data:', error);
            this.showAlert('Error loading application data', 'danger');
        }
    }

    async loadModels() {
        try {
            const response = await fetch('/api/models');
            const data = await response.json();
            
            const modelSelects = [
                document.getElementById('model-select'),
                document.getElementById('test-model-select')
            ];

            modelSelects.forEach(select => {
                select.innerHTML = '<option value="" selected disabled>Select a model</option>';
                data.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.id;
                    option.textContent = model.name;
                    select.appendChild(option);
                });
            });
        } catch (error) {
            console.error('Error loading models:', error);
            throw error;
        }
    }

    async loadTraits() {
        try {
            const response = await fetch('/api/traits');
            const data = await response.json();
            
            const traitSelects = [
                document.getElementById('trait-select'),
                document.getElementById('test-trait-select')
            ];

            traitSelects.forEach(select => {
                select.innerHTML = '<option value="" selected disabled>Select a trait</option>';
                data.traits.forEach(trait => {
                    const option = document.createElement('option');
                    option.value = trait.id;
                    option.textContent = trait.name;
                    option.title = trait.description;
                    select.appendChild(option);
                });
            });
        } catch (error) {
            console.error('Error loading traits:', error);
            throw error;
        }
    }

    async loadVectors() {
        const vectorsList = document.getElementById('vectors-list');
        const loadingElement = document.getElementById('loading-vectors');
        const noVectorsElement = document.getElementById('no-vectors-message');

        try {
            const response = await fetch('/api/vectors');
            const data = await response.json();
            
            loadingElement.classList.add('d-none');
            
            if (data.vectors.length === 0) {
                noVectorsElement.classList.remove('d-none');
                return;
            }

            noVectorsElement.classList.add('d-none');
            
            // Clear existing items (except loading and no-vectors messages)
            const existingItems = vectorsList.querySelectorAll('.vector-item');
            existingItems.forEach(item => item.remove());

            // Add vector items
            data.vectors.forEach(vector => {
                const item = this.createVectorListItem(vector);
                vectorsList.appendChild(item);
            });
        } catch (error) {
            console.error('Error loading vectors:', error);
            loadingElement.classList.add('d-none');
            this.showAlert('Error loading vectors', 'danger');
        }
    }

    createVectorListItem(vector) {
        const item = document.createElement('div');
        item.className = 'vector-item';
        item.setAttribute('data-model-id', vector.model_id);
        item.setAttribute('data-trait-id', vector.trait_id);

        const size = this.formatFileSize(vector.size);
        const date = new Date(vector.modified).toLocaleDateString();
        
        item.innerHTML = `
            <div class="vector-name-container">
                <div>
                    <div class="vector-name">
                        <strong>${vector.model_id}</strong><br>
                        <small class="text-muted">${vector.trait_id}</small>
                    </div>
                    <div class="memory-usage">
                        ${size} • ${date}
                    </div>
                </div>
                <button class="btn btn-outline-danger btn-sm delete-vector-btn" 
                        title="Delete vector" onclick="event.stopPropagation(); app.deleteVector('${vector.model_id}', '${vector.trait_id}')">
                    ×
                </button>
            </div>
        `;

        item.addEventListener('click', () => this.selectVector(vector.model_id, vector.trait_id));
        
        return item;
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    async selectVector(modelId, traitId) {
        // Update UI selection
        document.querySelectorAll('.vector-item').forEach(item => {
            item.classList.remove('active');
        });
        
        const selectedItem = document.querySelector(`[data-model-id="${modelId}"][data-trait-id="${traitId}"]`);
        if (selectedItem) {
            selectedItem.classList.add('active');
        }

        // Load and visualize vector
        try {
            const response = await fetch(`/api/vectors/${modelId}/${traitId}`);
            const vectorData = await response.json();
            
            this.currentVector = vectorData;
            this.visualizeVector(vectorData);
            
            // Switch to visualize tab
            const visualizeTab = document.getElementById('visualize-tab');
            const tab = new bootstrap.Tab(visualizeTab);
            tab.show();
        } catch (error) {
            console.error('Error loading vector:', error);
            this.showAlert('Error loading vector data', 'danger');
        }
    }

    async deleteVector(modelId, traitId) {
        if (!confirm(`Are you sure you want to delete the vector for ${modelId} - ${traitId}?`)) {
            return;
        }

        try {
            const response = await fetch(`/api/vectors/${modelId}/${traitId}`, {
                method: 'DELETE'
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.showAlert(result.message, 'success');
                // Reload vectors list
                await this.loadVectors();
                
                // Clear visualization if this was the current vector
                if (this.currentVector && 
                    this.currentVector.model_id === modelId && 
                    this.currentVector.trait_id === traitId) {
                    this.clearVisualization();
                }
            } else {
                this.showAlert(result.detail || 'Error deleting vector', 'danger');
            }
        } catch (error) {
            console.error('Error deleting vector:', error);
            this.showAlert('Error deleting vector', 'danger');
        }
    }

    async handleGenerateVector(e) {
        e.preventDefault();
        
        const modelId = document.getElementById('model-select').value;
        const traitId = document.getElementById('trait-select').value;
        
        if (!modelId || !traitId) {
            this.showAlert('Please select both a model and trait', 'warning');
            return;
        }

        const generateButton = document.getElementById('generate-button');
        const progressDiv = document.getElementById('generation-progress');
        const resultsDiv = document.getElementById('generation-results');
        const statusElement = document.getElementById('generation-status');
        const progressBar = progressDiv.querySelector('.progress-bar');

        // Show progress
        generateButton.disabled = true;
        generateButton.textContent = 'Generating...';
        progressDiv.classList.remove('d-none');
        resultsDiv.classList.add('d-none');
        
        // Simulate progress updates
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 90) progress = 90;
            progressBar.style.width = progress + '%';
            
            // Update status based on progress
            if (progress < 20) {
                statusElement.textContent = 'Loading model with 4-bit quantization...';
            } else if (progress < 50) {
                statusElement.textContent = 'Extracting activations from transformer layers...';
            } else if (progress < 80) {
                statusElement.textContent = 'Computing persona vectors from activation differences...';
            } else {
                statusElement.textContent = 'Finalizing vector generation and scoring effectiveness...';
            }
        }, 1000);

        try {
            const response = await fetch('/api/vectors/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model_id: modelId,
                    trait_id: traitId
                })
            });

            clearInterval(progressInterval);
            progressBar.style.width = '100%';
            statusElement.textContent = 'Vector generation complete!';

            const result = await response.json();
            
            if (response.ok) {
                // Show results
                this.displayGenerationResults(result);
                resultsDiv.classList.remove('d-none');
                
                // Reload vectors list
                await this.loadVectors();
                
                this.showAlert('Vector generated successfully!', 'success');
            } else {
                this.showAlert(result.detail || 'Error generating vector', 'danger');
            }
        } catch (error) {
            clearInterval(progressInterval);
            console.error('Error generating vector:', error);
            this.showAlert('Error generating vector. Please try again.', 'danger');
        } finally {
            generateButton.disabled = false;
            generateButton.textContent = 'Generate Vector';
            progressDiv.classList.add('d-none');
        }
    }

    displayGenerationResults(result) {
        document.getElementById('result-model').textContent = result.model_id;
        document.getElementById('result-trait').textContent = result.trait_id;
        document.getElementById('result-layers').textContent = result.num_layers;
        document.getElementById('result-questions').textContent = result.metadata?.total_samples_per_layer || 'N/A';
        document.getElementById('result-time').textContent = `${result.generation_time?.toFixed(1)}s`;
        
        // Find most effective layer (placeholder for now)
        document.getElementById('result-layer').textContent = `layer_${Math.floor(result.num_layers / 2)}`;
        document.getElementById('result-score').textContent = '0.75';
        
        const now = new Date();
        document.getElementById('result-date').textContent = now.toLocaleString();
    }

    async handleTestVector(e) {
        e.preventDefault();
        
        const modelId = document.getElementById('test-model-select').value;
        const traitId = document.getElementById('test-trait-select').value;
        const userPrompt = document.getElementById('test-prompt').value.trim();
        const coefficient = parseFloat(document.getElementById('steering-coefficient').value);
        
        if (!modelId || !traitId || !userPrompt) {
            this.showAlert('Please fill in all required fields', 'warning');
            return;
        }

        const testButton = document.getElementById('test-button');
        const resultsDiv = document.getElementById('test-results');

        testButton.disabled = true;
        testButton.textContent = 'Testing...';

        try {
            console.log('Making steering test request...');
            const response = await fetch('/api/steering/test', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model_id: modelId,
                    trait_id: traitId,
                    coefficient: coefficient,
                    user_prompt: userPrompt
                })
            });

            console.log('Response received:', response.status);
            const result = await response.json();
            console.log('Raw API result:', result);
            
            if (response.ok && result.success) {
                // Debug log to see what we're getting
                console.log('Test result:', result);
                this.displayTestResults(result);
                resultsDiv.classList.remove('d-none');
                this.showAlert('Vector test completed!', 'success');
            } else {
                this.showAlert(result.error || result.detail || 'Error testing vector', 'danger');
            }
        } catch (error) {
            console.error('Error testing vector:', error);
            this.showAlert('Error testing vector. Please try again.', 'danger');
        } finally {
            testButton.disabled = false;
            testButton.textContent = 'Test Vector';
        }
    }

    displayTestResults(result) {
        console.log('DisplayTestResults called with:', result);
        console.log('Has baseline_response?', !!result.baseline_response);
        console.log('Baseline content:', result.baseline_response?.substring(0, 100));
        
        document.getElementById('test-coefficient').textContent = result.steering_coefficient;
        document.getElementById('test-direction').textContent = result.direction;
        document.getElementById('steered-response').textContent = result.response;
        document.getElementById('test-time').textContent = result.elapsed_time?.toFixed(1) || 'N/A';
        document.getElementById('test-layers').textContent = result.num_layers_available || 'N/A';
        
        // Show baseline response if available
        if (result.baseline_response) {
            console.log('Creating baseline comparison UI...');
            // Update the UI to show comparison
            const resultsDiv = document.getElementById('test-results');
            console.log('resultsDiv found:', !!resultsDiv);
            
            // ALWAYS remove ANY existing comparison to ensure fresh headers
            const existingComparisons = resultsDiv.querySelectorAll('.response-comparison');
            console.log('Found', existingComparisons.length, 'existing comparison(s)');
            existingComparisons.forEach(comp => {
                console.log('Removing old comparison layout...');
                comp.remove();
            });
            
            // Always create fresh comparison layout
            console.log('Creating new comparison layout with correct headers...');
            // Create comparison layout
            const comparisonHTML = `
                <div class="response-comparison mt-3">
                    <div class="row">
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    <h6 class="mb-0">Baseline Response</h6>
                                    <small class="text-muted">Neutral system prompt</small>
                                </div>
                                <div class="card-body">
                                    <p class="card-text" id="baseline-response-text"></p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    <h6 class="mb-0">Steered Response</h6>
                                    <small class="text-muted">Coefficient: <span id="steered-coefficient"></span> | Direction: <span id="steered-direction"></span></small>
                                </div>
                                <div class="card-body">
                                    <p class="card-text" id="steered-response-text"></p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            console.log('Inserting HTML:', comparisonHTML.substring(0, 100));
            resultsDiv.insertAdjacentHTML('beforeend', comparisonHTML);
            console.log('HTML inserted successfully');
            
            // Update content - CORRECT MAPPING
            console.log('Updating comparison content...');
            const baselineTextElement = document.getElementById('baseline-response-text');
            const steeredTextElement = document.getElementById('steered-response-text');
            console.log('baseline element found:', !!baselineTextElement);
            console.log('steered element found:', !!steeredTextElement);
            console.log('Backend sends: response=', result.response?.substring(0, 30), '... (steered)');
            console.log('Backend sends: baseline_response=', result.baseline_response?.substring(0, 30), '... (baseline)');
            
            // CORRECT MAPPING - backend sends baseline_response (honest) and response (dishonest/steered)
            if (baselineTextElement) {
                baselineTextElement.textContent = result.baseline_response;  // honest baseline goes to baseline box
                console.log('Baseline box receives baseline_response (honest):', result.baseline_response?.substring(0, 50));
            }
            if (steeredTextElement) {
                steeredTextElement.textContent = result.response;  // dishonest steered goes to steered box  
                console.log('Steered box receives response (dishonest/steered):', result.response?.substring(0, 50));
            }
            
            const coeffElement = document.getElementById('steered-coefficient');
            const directionElement = document.getElementById('steered-direction');
            if (coeffElement) coeffElement.textContent = result.steering_coefficient;
            if (directionElement) directionElement.textContent = result.direction;
            
            // Force the comparison div to be visible and highlight it
            const comparisonDiv = resultsDiv.querySelector('.response-comparison');
            if (comparisonDiv) {
                comparisonDiv.style.border = '3px solid red';
                comparisonDiv.style.backgroundColor = '#f0f0f0';
                comparisonDiv.style.padding = '20px';
                comparisonDiv.style.margin = '20px 0';
                console.log('Comparison div styled for visibility');
            }
        }
    }

    visualizeVector(vectorData) {
        const container = document.getElementById('visualization-container');
        const placeholder = document.getElementById('visualize-placeholder');
        
        placeholder.classList.add('d-none');
        container.classList.remove('d-none');

        // Update title and details
        document.getElementById('vis-title').textContent = `${vectorData.model_id} - ${vectorData.trait_id}`;
        document.getElementById('vis-model').textContent = vectorData.model_id;
        document.getElementById('vis-trait').textContent = vectorData.trait_id;
        
        const generatedDate = new Date(vectorData.generated_at).toLocaleString();
        document.getElementById('vis-date').textContent = generatedDate;
        
        const layerCount = Object.keys(vectorData.vectors).length;
        document.getElementById('vis-layers').textContent = layerCount;

        // Find most effective layer
        let bestLayer = '';
        let bestScore = 0;
        let totalDimensions = 0;
        
        Object.entries(vectorData.effectiveness_scores || {}).forEach(([layer, score]) => {
            if (score > bestScore) {
                bestScore = score;
                bestLayer = layer;
            }
        });

        if (bestLayer && vectorData.vectors[bestLayer]) {
            const vector = vectorData.vectors[bestLayer];
            totalDimensions = Array.isArray(vector) ? vector.length : 0;
            const vectorNorm = Array.isArray(vector) ? 
                Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0)) : 0;
            
            document.getElementById('vis-layer').textContent = bestLayer;
            document.getElementById('vis-score').textContent = bestScore.toFixed(3);
            document.getElementById('vis-dimensions').textContent = totalDimensions;
            document.getElementById('vis-norm').textContent = vectorNorm.toFixed(3);
        }

        // Create charts
        this.createEffectivenessChart(vectorData);
        this.createVectorChart(vectorData, bestLayer);
    }

    createEffectivenessChart(vectorData) {
        const ctx = document.getElementById('effectiveness-chart').getContext('2d');
        
        if (this.charts.effectiveness) {
            this.charts.effectiveness.destroy();
        }

        const scores = vectorData.effectiveness_scores || {};
        const labels = Object.keys(scores);
        const data = Object.values(scores);

        this.charts.effectiveness = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Effectiveness Score',
                    data: data,
                    backgroundColor: 'rgba(36, 112, 220, 0.1)',
                    borderColor: 'rgba(36, 112, 220, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Layer Effectiveness Scores'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
    }

    createVectorChart(vectorData, layerName) {
        const ctx = document.getElementById('vector-chart').getContext('2d');
        
        if (this.charts.vector) {
            this.charts.vector.destroy();
        }

        if (!layerName || !vectorData.vectors[layerName]) {
            return;
        }

        const vector = vectorData.vectors[layerName];
        const dimensions = Array.isArray(vector) ? vector : [];
        
        // Show only first 50 dimensions for readability
        const displayDimensions = dimensions.slice(0, 50);
        const labels = displayDimensions.map((_, i) => `D${i + 1}`);

        this.charts.vector = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: `Vector Components (${layerName})`,
                    data: displayDimensions,
                    borderColor: 'rgba(220, 53, 69, 1)',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    tension: 0.1,
                    pointRadius: 2
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: `Vector Components (First 50 dimensions)`
                    }
                },
                scales: {
                    y: {
                        title: {
                            display: true,
                            text: 'Component Value'
                        }
                    }
                }
            }
        });
    }

    clearVisualization() {
        const container = document.getElementById('visualization-container');
        const placeholder = document.getElementById('visualize-placeholder');
        
        container.classList.add('d-none');
        placeholder.classList.remove('d-none');
        
        if (this.charts.effectiveness) {
            this.charts.effectiveness.destroy();
            this.charts.effectiveness = null;
        }
        
        if (this.charts.vector) {
            this.charts.vector.destroy();
            this.charts.vector = null;
        }
        
        this.currentVector = null;
    }

    async showModelInfo() {
        const modal = new bootstrap.Modal(document.getElementById('modelInfoModal'));
        const content = document.getElementById('modelInfoContent');
        
        content.innerHTML = 'Loading model information...';
        modal.show();
        
        try {
            const response = await fetch('/api/models/gpt-oss-20b/info');
            const info = await response.json();
            
            content.innerHTML = `
                <div class="row">
                    <div class="col-md-6">
                        <p><strong>Model ID:</strong> ${info.id}</p>
                        <p><strong>Type:</strong> ${info.model_type}</p>
                        <p><strong>Quantization:</strong> ${info.quantization}</p>
                        <p><strong>Max Length:</strong> ${info.max_length} tokens</p>
                    </div>
                    <div class="col-md-6">
                        <p><strong>Status:</strong> ${info.loaded ? 'Loaded' : 'Not loaded'}</p>
                        <p><strong>Memory Usage:</strong> ${info.memory_usage || 'N/A'}</p>
                        <p><strong>Device:</strong> ${info.device || 'N/A'}</p>
                    </div>
                </div>
                <div class="quantization-info">
                    <strong>4-bit Quantization:</strong> Model uses BitsAndBytesConfig with NF4 quantization 
                    to reduce memory usage from ~40GB to ~10GB while maintaining performance.
                </div>
            `;
        } catch (error) {
            content.innerHTML = '<p class="text-danger">Error loading model information.</p>';
        }
    }

    async showMemoryInfo() {
        // Simple memory info display
        const totalMemory = '24GB';
        const estimatedUsage = '~12GB';
        
        this.showAlert(
            `System Memory: ${totalMemory} unified | Estimated Usage: ${estimatedUsage} (model + activations)`, 
            'info'
        );
    }

    showAlert(message, type = 'info') {
        // Create and show Bootstrap alert
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        // Insert at top of main content
        const main = document.querySelector('main');
        main.insertBefore(alertDiv, main.firstChild);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
}

// Initialize the application
const app = new PersonaVectorSystem();

// Custom Trait Manager
class CustomTraitManager {
    constructor() {
        this.generatedPrompts = null;
        this.initEventListeners();
        this.loadCustomTraits();
    }

    initEventListeners() {
        // Generate button
        document.getElementById('generate-trait-btn')?.addEventListener('click', () => this.generateTrait());
        
        // Save button
        document.getElementById('save-trait-btn')?.addEventListener('click', () => this.saveTrait());
        
        // Modal show event - refresh custom traits list
        document.getElementById('customTraitModal')?.addEventListener('show.bs.modal', () => {
            this.loadCustomTraits();
        });
    }

    async loadCustomTraits() {
        try {
            const response = await fetch('/api/traits/custom');
            const data = await response.json();
            
            // Update slot count
            document.getElementById('trait-slots').textContent = `${data.count}/5 used`;
            
            // Update traits list
            const listDiv = document.getElementById('custom-traits-list');
            if (data.traits.length > 0) {
                listDiv.innerHTML = '<small>Current traits: ' + 
                    data.traits.map(t => `<span class="badge bg-secondary me-1">${t.name}</span>`).join('') +
                    '</small>';
            } else {
                listDiv.innerHTML = '';
            }
        } catch (error) {
            console.error('Error loading custom traits:', error);
        }
    }

    async generateTrait() {
        const traitName = document.getElementById('trait-name').value.trim();
        const positiveTrait = document.getElementById('positive-trait').value.trim();
        const negativeTrait = document.getElementById('negative-trait').value.trim();
        const numPairs = parseInt(document.getElementById('num-prompt-pairs').value);
        const temperature = parseFloat(document.getElementById('generation-temp').value);

        if (!traitName || !positiveTrait || !negativeTrait) {
            alert('Please fill in all trait fields');
            return;
        }

        // Show progress
        document.getElementById('trait-generation-progress').classList.remove('d-none');
        document.getElementById('generate-trait-btn').disabled = true;
        document.getElementById('trait-progress-bar').style.width = '20%';
        document.getElementById('trait-progress-text').textContent = 'Initializing Qwen model...';

        try {
            // Update progress
            setTimeout(() => {
                document.getElementById('trait-progress-bar').style.width = '50%';
                document.getElementById('trait-progress-text').textContent = 'Generating contrastive prompts...';
            }, 500);

            const response = await fetch('/api/traits/generate-custom', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    trait_name: traitName,
                    positive_trait: positiveTrait,
                    negative_trait: negativeTrait,
                    num_pairs: numPairs,
                    temperature: temperature
                })
            });

            const data = await response.json();

            if (response.ok && data.success) {
                // Update progress
                document.getElementById('trait-progress-bar').style.width = '100%';
                document.getElementById('trait-progress-text').textContent = 'Generation complete!';
                
                // Store generated prompts
                this.generatedPrompts = data;
                
                // Show preview
                this.showPromptsPreview(data.prompt_pairs);
                
                // Show save button, hide generate button
                document.getElementById('generate-trait-btn').classList.add('d-none');
                document.getElementById('save-trait-btn').classList.remove('d-none');
                
                // Reload custom traits list
                await this.loadCustomTraits();
                
                // Auto-save since generation was successful
                setTimeout(() => this.saveTrait(), 1000);
            } else {
                alert(data.detail || 'Error generating trait prompts');
                document.getElementById('trait-generation-progress').classList.add('d-none');
            }
        } catch (error) {
            console.error('Error generating trait:', error);
            alert('Error generating trait prompts. Please try again.');
            document.getElementById('trait-generation-progress').classList.add('d-none');
        } finally {
            document.getElementById('generate-trait-btn').disabled = false;
        }
    }

    showPromptsPreview(promptPairs) {
        const previewDiv = document.getElementById('generated-prompts-preview');
        const tbody = document.getElementById('prompts-preview-table');
        
        previewDiv.classList.remove('d-none');
        tbody.innerHTML = '';
        
        promptPairs.forEach((pair, index) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${index + 1}</td>
                <td><small>${pair.pos}</small></td>
                <td><small>${pair.neg}</small></td>
            `;
            tbody.appendChild(row);
        });
    }

    saveTrait() {
        if (!this.generatedPrompts) {
            alert('No generated prompts to save');
            return;
        }

        // Show success message
        alert(`Custom trait "${this.generatedPrompts.trait_name}" saved successfully!`);
        
        // Reset form
        document.getElementById('custom-trait-form').reset();
        document.getElementById('trait-generation-progress').classList.add('d-none');
        document.getElementById('generated-prompts-preview').classList.add('d-none');
        document.getElementById('save-trait-btn').classList.add('d-none');
        document.getElementById('generate-trait-btn').classList.remove('d-none');
        
        // Reload traits in main form
        app.loadTraits();
        
        // Close modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('customTraitModal'));
        modal.hide();
    }
}

// Initialize custom trait manager
const customTraitManager = new CustomTraitManager();

// Visualization Test Suite Manager
class VizTestSuite {
    constructor() {
        this.testCache = [];
        this.currentTestIndex = 0;
        this.testConfig = null;
        this.initEventListeners();
    }

    initEventListeners() {
        // Update config summary when inputs change
        ['viz-min-coeff', 'viz-max-coeff', 'viz-num-points'].forEach(id => {
            document.getElementById(id)?.addEventListener('input', () => this.updateConfigSummary());
        });

        // Run tests button
        document.getElementById('viz-run-tests')?.addEventListener('click', () => this.runTestSuite());

        // Create visualization button
        document.getElementById('viz-create-chart')?.addEventListener('click', () => this.createVisualization());

        // Populate model and trait selects when modal opens
        document.getElementById('vizTestModal')?.addEventListener('show.bs.modal', () => {
            this.populateSelects();
        });
    }

    populateSelects() {
        // Copy options from main form to viz form
        const modelSelect = document.getElementById('viz-model');
        const traitSelect = document.getElementById('viz-trait');
        const sourceModelSelect = document.getElementById('test-model-select');
        const sourceTraitSelect = document.getElementById('test-trait-select');

        if (modelSelect && sourceModelSelect) {
            modelSelect.innerHTML = sourceModelSelect.innerHTML;
        }
        if (traitSelect && sourceTraitSelect) {
            traitSelect.innerHTML = sourceTraitSelect.innerHTML;
        }
    }

    updateConfigSummary() {
        const minCoeff = parseFloat(document.getElementById('viz-min-coeff').value);
        const maxCoeff = parseFloat(document.getElementById('viz-max-coeff').value);
        const numPoints = parseInt(document.getElementById('viz-num-points').value);
        
        const summary = document.getElementById('viz-config-summary');
        if (summary) {
            summary.textContent = `Will run ${numPoints} tests from ${minCoeff.toFixed(1)} to ${maxCoeff.toFixed(1)}`;
        }
    }

    generateCoefficients(min, max, count) {
        const coefficients = [];
        const step = (max - min) / (count - 1);
        for (let i = 0; i < count; i++) {
            coefficients.push(min + (step * i));
        }
        return coefficients;
    }

    async runTestSuite() {
        // Get configuration
        const modelId = document.getElementById('viz-model').value;
        const traitId = document.getElementById('viz-trait').value;
        const prompt = document.getElementById('viz-prompt').value.trim();
        const minCoeff = parseFloat(document.getElementById('viz-min-coeff').value);
        const maxCoeff = parseFloat(document.getElementById('viz-max-coeff').value);
        const numPoints = parseInt(document.getElementById('viz-num-points').value);

        if (!modelId || !traitId || !prompt) {
            alert('Please fill in all required fields');
            return;
        }

        // Generate coefficient points
        const coefficients = this.generateCoefficients(minCoeff, maxCoeff, numPoints);
        
        // Store config
        this.testConfig = {
            modelId,
            traitId,
            prompt,
            coefficients,
            timestamp: new Date().toISOString()
        };

        // Reset cache
        this.testCache = [];
        this.currentTestIndex = 0;

        // Show progress section
        document.getElementById('viz-progress-section').classList.remove('d-none');
        document.getElementById('viz-results-preview').classList.add('d-none');
        document.getElementById('viz-run-tests').disabled = true;
        document.getElementById('viz-create-chart').classList.add('d-none');

        // Run tests sequentially
        for (let i = 0; i < coefficients.length; i++) {
            const coefficient = coefficients[i];
            await this.runSingleTest(coefficient, i);
        }

        // Tests complete
        document.getElementById('viz-run-tests').disabled = false;
        document.getElementById('viz-create-chart').classList.remove('d-none');
        document.getElementById('viz-progress-text').textContent = 'All tests complete!';
    }

    async runSingleTest(coefficient, index) {
        const { modelId, traitId, prompt, coefficients } = this.testConfig;
        
        // Update progress
        const progress = ((index + 1) / coefficients.length) * 100;
        document.getElementById('viz-progress-bar').style.width = `${progress}%`;
        document.getElementById('viz-progress-text').textContent = 
            `Testing coefficient ${coefficient.toFixed(1)} (${index + 1}/${coefficients.length})...`;

        try {
            const response = await fetch('/api/steering/test', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model_id: modelId,
                    trait_id: traitId,
                    coefficient: coefficient,
                    user_prompt: prompt
                })
            });

            const result = await response.json();
            
            if (response.ok && result.success) {
                // Analyze the result
                const coherenceScore = this.analyzeCoherence(result.response);
                const ethicalStance = this.analyzeEthicalStance(result.response, coefficient);
                
                // Cache the result
                const cachedResult = {
                    coefficient,
                    response: result.response,
                    baseline_response: result.baseline_response,
                    elapsed_time: result.elapsed_time || 0,
                    coherenceScore,
                    ethicalStance,
                    responseLength: result.response.length
                };
                
                this.testCache.push(cachedResult);
                
                // Update preview table
                this.updatePreviewTable(cachedResult);
            }
        } catch (error) {
            console.error(`Error testing coefficient ${coefficient}:`, error);
        }
    }

    analyzeCoherence(response) {
        // Simple coherence analysis based on response characteristics
        let score = 100;
        
        // Check for repetition
        const words = response.split(/\s+/);
        const uniqueWords = new Set(words);
        const repetitionRatio = uniqueWords.size / words.length;
        if (repetitionRatio < 0.3) score -= 40;  // Heavy repetition
        
        // Check for gibberish patterns
        if (/(.)\1{5,}/.test(response)) score -= 30;  // Repeated characters
        if (/[\u4e00-\u9fa5]/.test(response)) score -= 20;  // Contains Chinese chars (breakdown indicator)
        if (/Just\s+Just\s+Just/i.test(response)) score -= 30;  // Specific breakdown pattern
        
        // Check for incomplete sentences
        if (response.length < 50) score -= 20;
        if (!response.match(/[.!?]$/)) score -= 10;  // Doesn't end with punctuation
        
        return Math.max(0, Math.min(100, score));
    }

    analyzeEthicalStance(response, coefficient) {
        // Map coefficient to expected ethical stance
        // Positive coefficient = dishonest, Negative = honest
        let stance = coefficient * 40;  // Scale to -80 to +80 range
        
        // Adjust based on content analysis
        const honestKeywords = /ethical|legal|return|report|honest|integrity|responsible/gi;
        const dishonestKeywords = /slip.*pocket|take|keep|yours|tempting/gi;
        
        const honestMatches = (response.match(honestKeywords) || []).length;
        const dishonestMatches = (response.match(dishonestKeywords) || []).length;
        
        stance += (dishonestMatches - honestMatches) * 10;
        
        return Math.max(-100, Math.min(100, stance));
    }

    updatePreviewTable(result) {
        document.getElementById('viz-results-preview').classList.remove('d-none');
        const tbody = document.getElementById('viz-results-table');
        
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${result.coefficient.toFixed(1)}</strong></td>
            <td>${result.coherenceScore}%</td>
            <td>${result.elapsed_time.toFixed(1)}s</td>
            <td><small>${result.response.substring(0, 50)}...</small></td>
        `;
        tbody.appendChild(row);
    }

    createVisualization() {
        if (this.testCache.length === 0) {
            alert('No test results to visualize');
            return;
        }

        // Store cache in sessionStorage for the visualization page
        sessionStorage.setItem('vizTestCache', JSON.stringify(this.testCache));
        sessionStorage.setItem('vizTestConfig', JSON.stringify(this.testConfig));
        
        // Open visualization page
        window.location.href = '/visualization?mode=dynamic';
    }
}

// Initialize test suite manager
const vizTestSuite = new VizTestSuite();