/**
 * Persona Vector Steering Visualization
 * Creates an interactive chart showing the "thermostat" effect of steering coefficients
 */

class PersonaVectorVisualization {
    constructor() {
        this.chart = null;
        this.mode = this.getMode();
        this.initVisualization();
    }

    getMode() {
        // Check if we're in dynamic mode (from test suite)
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get('mode') === 'dynamic' ? 'dynamic' : 'static';
    }

    initVisualization() {
        // Create visualization container if it doesn't exist
        const container = document.getElementById('steering-visualization');
        if (!container) {
            console.log('Visualization container not found');
            return;
        }

        const canvas = document.createElement('canvas');
        canvas.id = 'thermostat-chart';
        container.appendChild(canvas);

        if (this.mode === 'dynamic') {
            this.loadDynamicData();
        } else {
            this.createThermostatChart();
        }
    }

    loadDynamicData() {
        // Load cached test results from sessionStorage
        const cachedData = sessionStorage.getItem('vizTestCache');
        const cachedConfig = sessionStorage.getItem('vizTestConfig');
        
        if (!cachedData || !cachedConfig) {
            console.error('No cached test data found');
            this.createThermostatChart();  // Fall back to static chart
            return;
        }

        const testResults = JSON.parse(cachedData);
        const testConfig = JSON.parse(cachedConfig);
        
        // Update page title with test info
        const headerSection = document.querySelector('.header-section h1');
        if (headerSection) {
            headerSection.innerHTML = `🎛️ Dynamic Test Results: ${testConfig.traitId}`;
        }
        
        const subtitle = document.querySelector('.header-section p');
        if (subtitle) {
            subtitle.innerHTML = `Model: ${testConfig.modelId} | Prompt: "${testConfig.prompt.substring(0, 50)}..."`;
        }
        
        // Create chart from dynamic data
        this.createDynamicChart(testResults, testConfig);
    }

    createDynamicChart(testResults, testConfig) {
        const ctx = document.getElementById('thermostat-chart').getContext('2d');
        
        // Extract data points
        const coefficients = testResults.map(r => r.coefficient);
        const coherenceScores = testResults.map(r => r.coherenceScore);
        const ethicalStances = testResults.map(r => r.ethicalStance);
        const processingTimes = testResults.map(r => r.elapsed_time);
        
        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: coefficients.map(c => c.toFixed(1)),
                datasets: [
                    {
                        label: 'Response Coherence',
                        data: coherenceScores,
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        yAxisID: 'y-coherence',
                        tension: 0.4,
                        borderWidth: 3,
                        pointRadius: 6,
                        pointHoverRadius: 8
                    },
                    {
                        label: 'Ethical Stance (Honest ← → Dishonest)',
                        data: ethicalStances,
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        yAxisID: 'y-ethics',
                        tension: 0.4,
                        borderWidth: 3,
                        pointRadius: 6,
                        pointHoverRadius: 8
                    },
                    {
                        label: 'Processing Time (s)',
                        data: processingTimes,
                        borderColor: 'rgb(153, 102, 255)',
                        backgroundColor: 'rgba(153, 102, 255, 0.2)',
                        yAxisID: 'y-time',
                        tension: 0.4,
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointRadius: 4,
                        pointHoverRadius: 6
                    }
                ]
            },
            options: this.getChartOptions(`Test: "${testConfig.prompt.substring(0, 80)}..."`)
        });
        
        // Update example section with actual test responses
        this.updateExamples(testResults);
    }

    updateExamples(testResults) {
        const exampleSection = document.querySelector('.example-section');
        if (!exampleSection) return;
        
        // Select interesting points to show
        const examples = [];
        
        // Find min, neutral, max coefficients
        const minCoeff = testResults.reduce((min, r) => r.coefficient < min.coefficient ? r : min);
        const maxCoeff = testResults.reduce((max, r) => r.coefficient > max.coefficient ? r : max);
        const neutral = testResults.find(r => Math.abs(r.coefficient) < 0.1) || testResults[Math.floor(testResults.length / 2)];
        
        exampleSection.innerHTML = `
            <h4>Response Examples from This Test</h4>
            
            <div class="response-example">
                <span class="coefficient-badge honest-badge">${minCoeff.coefficient.toFixed(1)}</span>
                <p><strong>Response:</strong> "${minCoeff.response.substring(0, 200)}..."</p>
                <p class="text-muted">Coherence: ${minCoeff.coherenceScore}% | Time: ${minCoeff.elapsed_time.toFixed(1)}s</p>
            </div>
            
            <div class="response-example">
                <span class="coefficient-badge neutral-badge">${neutral.coefficient.toFixed(1)}</span>
                <p><strong>Response:</strong> "${neutral.response.substring(0, 200)}..."</p>
                <p class="text-muted">Coherence: ${neutral.coherenceScore}% | Time: ${neutral.elapsed_time.toFixed(1)}s</p>
            </div>
            
            <div class="response-example">
                <span class="coefficient-badge dishonest-badge">${maxCoeff.coefficient.toFixed(1)}</span>
                <p><strong>Response:</strong> "${maxCoeff.response.substring(0, 200)}..."</p>
                <p class="text-muted">Coherence: ${maxCoeff.coherenceScore}% | Time: ${maxCoeff.elapsed_time.toFixed(1)}s</p>
            </div>
        `;
    }

    getChartOptions(subtitle = '') {
        return {
            responsive: true,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Persona Vector Steering: The "Thermostat" Effect',
                    font: {
                        size: 18,
                        weight: 'bold'
                    },
                    padding: 20
                },
                subtitle: {
                    display: true,
                    text: subtitle || 'How steering coefficient affects response coherence and ethical stance',
                    font: {
                        size: 14
                    },
                    padding: {
                        bottom: 20
                    }
                },
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        padding: 15,
                        font: {
                            size: 12
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        afterLabel: function(context) {
                            if (context.datasetIndex === 0) {
                                return 'Higher = More coherent response';
                            } else if (context.datasetIndex === 1) {
                                return 'Negative = Honest, Positive = Dishonest';
                            } else {
                                return 'Time to generate response';
                            }
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Steering Coefficient',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    },
                    grid: {
                        display: true,
                        color: 'rgba(200, 200, 200, 0.3)'
                    }
                },
                'y-coherence': {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Response Coherence (%)',
                        font: {
                            size: 12
                        }
                    },
                    min: 0,
                    max: 100,
                    grid: {
                        color: 'rgba(75, 192, 192, 0.2)'
                    }
                },
                'y-ethics': {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Ethical Stance (← Honest | Dishonest →)',
                        font: {
                            size: 12
                        }
                    },
                    min: -100,
                    max: 100,
                    grid: {
                        drawOnChartArea: false,
                    },
                    ticks: {
                        callback: function(value) {
                            if (value === -100) return 'Max Honest';
                            if (value === 0) return 'Neutral';
                            if (value === 100) return 'Max Dishonest';
                            return value;
                        }
                    }
                },
                'y-time': {
                    type: 'linear',
                    display: false,
                    min: 0,
                    max: 40
                }
            }
        };
    }

    createThermostatChart() {
        const ctx = document.getElementById('thermostat-chart').getContext('2d');
        
        // Data from the test results
        const coefficients = [-2.0, -1.3, -0.6, 0, 1.4, 1.5, 1.7, 1.8];
        
        // Coherence scores (0-100): How coherent/readable the response is
        const coherenceScores = [
            10,  // -2.0: Repetitive collapse
            95,  // -1.3: Clear, detailed
            98,  // -0.6: Perfect clarity
            100, // 0: Neutral baseline
            75,  // 1.4: Conflicted but readable
            60,  // 1.5: Breaking down
            40,  // 1.7: Barely coherent
            5    // 1.8: Gibberish/collapse
        ];

        // Ethical stance (-100 to +100): -100 = maximum honesty, +100 = maximum dishonesty
        const ethicalStance = [
            -95, // -2.0: Hyper-honest (but broken)
            -80, // -1.3: Strong ethical emphasis
            -60, // -0.6: Clear ethical guidance
            0,   // 0: Neutral
            70,  // 1.4: Suggests theft but conflicts
            80,  // 1.5: Dishonest inclination
            85,  // 1.7: Strong dishonest
            95   // 1.8: Maximum dishonest (but broken)
        ];

        // Processing times for context
        const processingTimes = [30.9, 30.8, 22.5, 30.6, 18.8, 15.7, 4.5, 32.7];

        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: coefficients.map(c => c.toFixed(1)),
                datasets: [
                    {
                        label: 'Response Coherence',
                        data: coherenceScores,
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        yAxisID: 'y-coherence',
                        tension: 0.4,
                        borderWidth: 3,
                        pointRadius: 6,
                        pointHoverRadius: 8
                    },
                    {
                        label: 'Ethical Stance (Honest ← → Dishonest)',
                        data: ethicalStance,
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        yAxisID: 'y-ethics',
                        tension: 0.4,
                        borderWidth: 3,
                        pointRadius: 6,
                        pointHoverRadius: 8
                    },
                    {
                        label: 'Processing Time (s)',
                        data: processingTimes,
                        borderColor: 'rgb(153, 102, 255)',
                        backgroundColor: 'rgba(153, 102, 255, 0.2)',
                        yAxisID: 'y-time',
                        tension: 0.4,
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointRadius: 4,
                        pointHoverRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Persona Vector Steering: The "Thermostat" Effect',
                        font: {
                            size: 18,
                            weight: 'bold'
                        },
                        padding: 20
                    },
                    subtitle: {
                        display: true,
                        text: 'How steering coefficient affects response coherence and ethical stance',
                        font: {
                            size: 14
                        },
                        padding: {
                            bottom: 20
                        }
                    },
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            padding: 15,
                            font: {
                                size: 12
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            afterLabel: function(context) {
                                if (context.datasetIndex === 0) {
                                    return 'Higher = More coherent response';
                                } else if (context.datasetIndex === 1) {
                                    return 'Negative = Honest, Positive = Dishonest';
                                } else {
                                    return 'Time to generate response';
                                }
                            }
                        }
                    },
                    annotation: {
                        annotations: {
                            neutralLine: {
                                type: 'line',
                                xMin: 3,
                                xMax: 3,
                                borderColor: 'rgb(200, 200, 200)',
                                borderWidth: 2,
                                borderDash: [10, 5],
                                label: {
                                    display: true,
                                    content: 'Neutral',
                                    position: 'start'
                                }
                            },
                            stableZone: {
                                type: 'box',
                                xMin: 1,
                                xMax: 6,
                                backgroundColor: 'rgba(0, 255, 0, 0.05)',
                                borderColor: 'rgba(0, 255, 0, 0.3)',
                                borderWidth: 1,
                                label: {
                                    display: true,
                                    content: 'Stable Response Zone',
                                    position: 'center',
                                    font: {
                                        size: 11
                                    }
                                }
                            },
                            leftCollapse: {
                                type: 'box',
                                xMin: -0.5,
                                xMax: 0.5,
                                backgroundColor: 'rgba(255, 0, 0, 0.05)',
                                borderColor: 'rgba(255, 0, 0, 0.3)',
                                borderWidth: 1,
                                label: {
                                    display: true,
                                    content: 'Model Collapse',
                                    position: 'center',
                                    font: {
                                        size: 10
                                    }
                                }
                            },
                            rightCollapse: {
                                type: 'box',
                                xMin: 6.5,
                                xMax: 7.5,
                                backgroundColor: 'rgba(255, 0, 0, 0.05)',
                                borderColor: 'rgba(255, 0, 0, 0.3)',
                                borderWidth: 1,
                                label: {
                                    display: true,
                                    content: 'Model Collapse',
                                    position: 'center',
                                    font: {
                                        size: 10
                                    }
                                }
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Steering Coefficient',
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        grid: {
                            display: true,
                            color: 'rgba(200, 200, 200, 0.3)'
                        }
                    },
                    'y-coherence': {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Response Coherence (%)',
                            font: {
                                size: 12
                            }
                        },
                        min: 0,
                        max: 100,
                        grid: {
                            color: 'rgba(75, 192, 192, 0.2)'
                        }
                    },
                    'y-ethics': {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Ethical Stance (← Honest | Dishonest →)',
                            font: {
                                size: 12
                            }
                        },
                        min: -100,
                        max: 100,
                        grid: {
                            drawOnChartArea: false,
                        },
                        ticks: {
                            callback: function(value) {
                                if (value === -100) return 'Max Honest';
                                if (value === 0) return 'Neutral';
                                if (value === 100) return 'Max Dishonest';
                                return value;
                            }
                        }
                    },
                    'y-time': {
                        type: 'linear',
                        display: false,
                        min: 0,
                        max: 40
                    }
                }
            }
        });
    }

    updateWithNewData(newCoefficient, newCoherence, newEthics, newTime) {
        // Method to add new test results dynamically
        if (!this.chart) return;
        
        this.chart.data.labels.push(newCoefficient.toFixed(1));
        this.chart.data.datasets[0].data.push(newCoherence);
        this.chart.data.datasets[1].data.push(newEthics);
        this.chart.data.datasets[2].data.push(newTime);
        this.chart.update();
    }
}

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing Persona Vector Visualization...');
    window.personaViz = new PersonaVectorVisualization();
});