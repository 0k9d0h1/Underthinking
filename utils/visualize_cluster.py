import json
import re
import os
import random
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from collections import defaultdict

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['SECRET_KEY'] = 'thought_cluster_visualizer_key'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Data storage
all_questions = []  # List to store data for each question

def parse_clusters(cluster_text):
    """Extract clusters from text"""
    clusters = {}
    
    # Regular expression to find clusters
    cluster_pattern = r"Cluster (\d+): ([^\n]+)\s+Thoughts: ([0-9, ]+)"
    matches = re.finditer(cluster_pattern, cluster_text)
    
    for match in matches:
        cluster_id = int(match.group(1))
        cluster_description = match.group(2).strip()
        thought_ids = [int(tid.strip()) for tid in match.group(3).split(',')]
        
        clusters[cluster_id] = {
            'description': cluster_description,
            'thoughts': thought_ids
        }
    
    return clusters

def load_thoughts_from_data(thought_data):
    """Load thoughts from a single JSON object"""
    thoughts = {}
    
    if isinstance(thought_data, dict) and 'phrases' in thought_data:
        for i, phrase in enumerate(thought_data['phrases']):
            thoughts[i] = phrase
    
    return thoughts

def generate_stats(clusters, thoughts):
    """Generate statistics about the data"""
    
    num_clusters = len(clusters)
    num_thoughts = len(thoughts)
    
    # Count how many times each thought appears in clusters
    thoughts_per_cluster = defaultdict(int)
    for cluster in clusters.values():
        for thought_id in cluster['thoughts']:
            thoughts_per_cluster[thought_id] += 1
    
    shared_thoughts = [tid for tid, count in thoughts_per_cluster.items() if count > 1]
    
    return {
        'num_clusters': num_clusters,
        'num_thoughts': num_thoughts,
        'shared_thoughts': len(shared_thoughts)
    }

def generate_cluster_colors(num_clusters):
    """Generate a list of distinct colors for clusters"""
    # Predefined colors for up to 15 clusters
    predefined_colors = [
        "#FF5252", "#FF4081", "#E040FB", "#7C4DFF", "#536DFE",  # Red to Blue
        "#448AFF", "#40C4FF", "#18FFFF", "#64FFDA", "#69F0AE",  # Blue to Green
        "#B2FF59", "#EEFF41", "#FFFF00", "#FFD740", "#FFAB40",  # Green to Orange
    ]
    
    if num_clusters <= len(predefined_colors):
        return predefined_colors[:num_clusters]
    
    # If we need more colors, generate them randomly
    colors = predefined_colors.copy()
    
    for _ in range(num_clusters - len(predefined_colors)):
        # Generate a random color that's not too light
        r = random.randint(0, 200)  # Avoid too light colors
        g = random.randint(0, 200)
        b = random.randint(0, 200)
        color = f"#{r:02x}{g:02x}{b:02x}"
        colors.append(color)
    
    return colors

def prepare_thought_data(question_data):
    """Prepare thought data for visualization including cluster assignments and colors"""
    thoughts = question_data['thoughts']
    clusters = question_data['clusters']
    correctness_label = correctness_label = all_questions[question_data['question_id']].get('correctness', None)
    
    # Generate colors for each cluster
    colors = generate_cluster_colors(len(clusters))
    cluster_colors = {cluster_id: colors[i % len(colors)] for i, cluster_id in enumerate(clusters.keys())}
    
    # Create thought display data with cluster assignments
    thought_display_data = []
    
    # Find which cluster(s) each thought belongs to
    thought_clusters = defaultdict(list)
    for cluster_id, cluster in clusters.items():
        for thought_id in cluster['thoughts']:
            thought_clusters[thought_id].append({
                'id': cluster_id,
                'description': cluster['description'],
                'color': cluster_colors[cluster_id]
            })
    
    # Create display data for each thought
    for thought_id, text in thoughts.items():
        # Get clusters this thought belongs to
        thought_cluster_data = thought_clusters.get(thought_id, [])
        
        # If thought is not in any cluster, assign it to a default "Unclustered" category
        if not thought_cluster_data:
            thought_cluster_data = [{
                'id': 'unclustered',
                'description': 'Unclustered',
                'color': '#CCCCCC'  # Gray color for unclustered thoughts
            }]
        
        thought_display_data.append({
            'id': thought_id,
            'text': text,
            'clusters': thought_cluster_data,
            # Use the first cluster as the primary color for the thought
            'primary_color': thought_cluster_data[0]['color'],
            # For thoughts in multiple clusters, provide all colors
            'all_colors': [c['color'] for c in thought_cluster_data],
            'correctness': correctness_label  # Apply per-question label to each thought
        })
    
    # Sort thoughts by ID for consistent display
    thought_display_data.sort(key=lambda x: x['id'])
    
    # Prepare cluster data for legend
    cluster_data = []
    for cluster_id, cluster in clusters.items():
        cluster_data.append({
            'id': cluster_id,
            'description': cluster['description'],
            'color': cluster_colors[cluster_id],
            'thought_count': len(cluster['thoughts'])
        })
    
    # Sort clusters by ID
    cluster_data.sort(key=lambda x: x['id'])
    
    return {
        'thoughts': thought_display_data,
        'clusters': cluster_data
    }

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads"""
    global all_questions
    
    if 'cluster_file' not in request.files or 'thoughts_file' not in request.files:
        return jsonify({'error': 'Both files are required'}), 400
    
    cluster_file = request.files['cluster_file']
    thoughts_file = request.files['thoughts_file']
    
    if cluster_file.filename == '' or thoughts_file.filename == '':
        return jsonify({'error': 'Both files are required'}), 400
    
    # Save files
    cluster_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(cluster_file.filename))
    thoughts_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(thoughts_file.filename))
    
    cluster_file.save(cluster_path)
    thoughts_file.save(thoughts_path)
    
    try:
        # Reset questions data
        all_questions = []
        
        # Process cluster file - load all questions
        with open(cluster_path, 'r', encoding='utf-8') as f:
            cluster_lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # Process thoughts file - load all thoughts for each question
        with open(thoughts_path, 'r', encoding='utf-8') as f:
            thought_lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # Make sure we have equal number of entries in both files
        if len(cluster_lines) != len(thought_lines):
            return jsonify({
                'error': f"Mismatch in number of questions: {len(cluster_lines)} cluster entries vs {len(thought_lines)} thought entries"
            }), 400
        
        # Process each question
        for q_idx, (cluster_line, thought_line) in enumerate(zip(cluster_lines, thought_lines)):
            try:
                # Parse cluster data
                cluster_data = json.loads(cluster_line)
                cluster_text = cluster_data.get('gpt4o_answer', '')
                clusters = parse_clusters(cluster_text)
                
                # Parse thoughts data
                thought_data = json.loads(thought_line)
                thoughts = load_thoughts_from_data(thought_data)
                
                # Generate stats
                stats = generate_stats(clusters, thoughts)
                
                # Store question data
                all_questions.append({
                    'question_id': q_idx,
                    'clusters': clusters,
                    'thoughts': thoughts,
                    'stats': stats
                })
                
            except json.JSONDecodeError as e:
                return jsonify({
                    'error': f"JSON parsing error for question {q_idx+1}",
                    'context': f"Cluster line: {cluster_line[:50]}..." if len(cluster_line) > 50 else cluster_line
                }), 400
        
        # Prepare the first question's data for visualization
        first_question_data = prepare_thought_data(all_questions[0]) if all_questions else None
        
        # Return success with the number of questions found
        return jsonify({
            'success': True,
            'num_questions': len(all_questions),
            'first_question': first_question_data,
            'stats': all_questions[0]['stats'] if all_questions else None
        })
    
    except Exception as e:
        return jsonify({'error': f"Unexpected error: {str(e)}"}), 500

@app.route('/get_question/<int:question_id>')
def get_question(question_id):
    """Return data for a specific question"""
    global all_questions
    
    if question_id < 0 or question_id >= len(all_questions):
        return jsonify({'error': 'Question not found'}), 404
    
    question_data = all_questions[question_id]
    visualization_data = prepare_thought_data(question_data)
    
    return jsonify({
        'visualization_data': visualization_data,
        'stats': question_data['stats']
    })

# Create templates directory and necessary HTML files
os.makedirs('templates', exist_ok=True)

# Create index.html
index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Thought Cluster Visualizer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
            background-color: #f8f9fa;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .upload-section {
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .file-input {
            margin-bottom: 15px;
        }
        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: background-color 0.2s;
        }
        button:hover {
            background-color: #2980b9;
        }
        button.disabled {
            background-color: #bdc3c7;
            cursor: not-allowed;
        }
        button.disabled:hover {
            background-color: #bdc3c7;
        }
        .loading {
            display: none;
            margin-left: 10px;
        }
        .error {
            color: #e74c3c;
            margin-top: 10px;
        }
        .question-nav {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
            background-color: #fff;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .nav-buttons {
            display: flex;
            gap: 10px;
            margin-right: 20px;
        }
        .question-indicator {
            font-weight: bold;
            font-size: 16px;
        }
        .thought-grid {
            display: grid;
            grid-template-columns: 1fr;  /* One full-width column */
            gap: 10px;
            margin-top: 20px;
        }
        .thought-card {
            position: relative;
            padding: 10px 15px;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
            overflow: hidden;
            background-color: #fff;
        }
        .thought-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        .thought-id {
            font-weight: bold;
            margin-bottom: 8px;
            color: #2c3e50;
        }
        .thought-text {
            white-space: pre-wrap;
            overflow-wrap: break-word;
        }
        .color-stripe {
            height: 7px;
            width: 100%;
            position: absolute;
            top: 0;
            left: 0;
        }
        .multi-color-stripe {
            display: flex;
            height: 7px;
            width: 100%;
            position: absolute;
            top: 0;
            left: 0;
        }
        .stripe-segment {
            height: 100%;
            flex-grow: 1;
        }
        .cluster-tooltip {
            position: absolute;
            background-color: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 12px;
            z-index: 100;
            pointer-events: none;
            visibility: hidden;
            opacity: 0;
            transition: opacity 0.2s;
            white-space: nowrap;
        }
        .thought-card:hover .cluster-tooltip {
            visibility: visible;
            opacity: 1;
        }
        .stats-section {
            background-color: #fff;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .cluster-legend {
            background-color: #fff;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .legend-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin-bottom: 5px;
        }
        .legend-color {
            width: 15px;
            height: 15px;
            border-radius: 3px;
            margin-right: 8px;
        }
        .controls {
            display: flex;
            gap: 15px;
            align-items: center;
            margin-top: 15px;
        }
        .filter-dropdown {
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
        .search-box {
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #ddd;
            flex-grow: 1;
            max-width: 300px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Thought Cluster Visualizer</h1>
        
        <div class="upload-section">
            <h2>Load Data Files</h2>
            <form id="upload-form">
                <div class="file-input">
                    <label for="cluster-file">Cluster File (JSONL):</label>
                    <input type="file" id="cluster-file" name="cluster_file" accept=".jsonl,.json">
                </div>
                
                <div class="file-input">
                    <label for="thoughts-file">Thoughts File (JSONL):</label>
                    <input type="file" id="thoughts-file" name="thoughts_file" accept=".jsonl,.json">
                </div>
                
                <button type="submit">Load Data</button>
                <span class="loading" id="loading">Loading...</span>
                <div class="error" id="error-message"></div>
            </form>
        </div>
        
        <div id="visualization-container" style="display:none;">
            <div class="question-nav" id="question-nav">
                <div class="nav-buttons">
                    <button id="prev-question" title="Previous Question">← Previous</button>
                    <button id="next-question" title="Next Question">Next →</button>
                    <input type="number" id="jump-input" placeholder="Go to question #" style="width: 150px; margin-left: 15px;">
                    <button id="jump-button">Go</button>
                </div>
                <div class="question-indicator" id="question-indicator">Question 1 of 1</div>
            </div>
            
            <div class="stats-section" id="stats-section">
                <h2>Statistics</h2>
                <div id="stats-content">No data loaded</div>
            </div>
            
            <div class="cluster-legend" id="cluster-legend">
                <h2>Cluster Legend</h2>
                <div class="controls">
                    <select id="cluster-filter" class="filter-dropdown">
                        <option value="all">Show All Clusters</option>
                    </select>
                    <input type="text" id="search-box" class="search-box" placeholder="Search thoughts...">
                    <button id="clear-filters">Clear Filters</button>
                </div>
                <div id="legend-grid" class="legend-grid"></div>
            </div>
            
            <div id="thought-grid" class="thought-grid"></div>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const uploadForm = document.getElementById('upload-form');
            const errorMessage = document.getElementById('error-message');
            const loading = document.getElementById('loading');
            const visualizationContainer = document.getElementById('visualization-container');
            const thoughtGrid = document.getElementById('thought-grid');
            const legendGrid = document.getElementById('legend-grid');
            const statsContent = document.getElementById('stats-content');
            const questionIndicator = document.getElementById('question-indicator');
            const prevButton = document.getElementById('prev-question');
            const nextButton = document.getElementById('next-question');
            const clusterFilter = document.getElementById('cluster-filter');
            const searchBox = document.getElementById('search-box');
            const clearFiltersButton = document.getElementById('clear-filters');
            
            // Add file format descriptions
            document.getElementById('cluster-file').insertAdjacentHTML('afterend', 
                '<div><small>Format: One JSON object per line with {"gpt4o_answer": "Cluster 1... Cluster 2..."}</small></div>');
            document.getElementById('thoughts-file').insertAdjacentHTML('afterend', 
                '<div><small>Format: One JSON object per line with {"phrases": ["Thought 0 text", "Thought 1 text", ...]}</small></div>');
            
            let currentQuestionId = 0;
            let totalQuestions = 0;
            let currentVisualizationData = null;
            let allQuestions = [];
            
            // Question navigation listeners
            prevButton.addEventListener('click', function() {
                if (currentQuestionId > 0) {
                    currentQuestionId--;
                    loadQuestion(currentQuestionId);
                }
            });
            
            nextButton.addEventListener('click', function() {
                if (currentQuestionId < totalQuestions - 1) {
                    currentQuestionId++;
                    loadQuestion(currentQuestionId);
                }
            });

            const jumpInput = document.getElementById('jump-input');
            const jumpButton = document.getElementById('jump-button');

            jumpButton.addEventListener('click', function() {
                const qNum = parseInt(jumpInput.value);
                if (!isNaN(qNum) && qNum >= 1 && qNum <= totalQuestions) {
                    currentQuestionId = qNum - 1;
                    loadQuestion(currentQuestionId);
                }

                // Add correctness label to indicator
                const correctness = (allQuestions[currentQuestionId]?.thoughts[0]?.correctness);
                const correctnessLabel = correctness === true ? '✔️ Correct' : correctness === false ? '❌ Incorrect' : '❓ Unknown';
                questionIndicator.textContent = `Question ${currentQuestionId + 1} of ${totalQuestions} (${correctnessLabel})`;
            });

            // Filter listeners
            clusterFilter.addEventListener('change', filterThoughts);
            searchBox.addEventListener('input', filterThoughts);
            clearFiltersButton.addEventListener('click', clearFilters);
            
            function clearFilters() {
                clusterFilter.value = 'all';
                searchBox.value = '';
                filterThoughts();
            }
            
            // Form submission handler
            uploadForm.addEventListener('submit', function(e) {
                e.preventDefault();
                
                const formData = new FormData(uploadForm);
                const clusterFile = document.getElementById('cluster-file').files[0];
                const thoughtsFile = document.getElementById('thoughts-file').files[0];
                
                if (!clusterFile || !thoughtsFile) {
                    errorMessage.textContent = 'Please select both files';
                    return;
                }
                
                errorMessage.textContent = '';
                loading.style.display = 'inline';
                
                fetch('/upload', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    loading.style.display = 'none';
                    
                    if (data.error) {
                        let errorText = data.error;
                        if (data.context) {
                            errorText += '<br><pre>' + data.context + '</pre>';
                        }
                        errorMessage.innerHTML = errorText;
                        return;
                    }
                    
                    // Update with the number of questions
                    totalQuestions = data.num_questions;
                    currentQuestionId = 0;
                    
                    // Show visualization container
                    visualizationContainer.style.display = 'block';
                    
                    // Initialize with first question
                    if (data.first_question) {
                        allQuestions = [data.first_question];
                        
                        // Display first question
                        displayVisualization(data.first_question, data.stats);
                        updateQuestionNavigator();
                    }
                })
                .catch(error => {
                    loading.style.display = 'none';
                    errorMessage.innerHTML = 'Error: ' + error.message;
                });
            });
            
            function loadQuestion(questionId) {
                // Check if we already have this question in cache
                if (allQuestions[questionId]) {
                    displayVisualization(allQuestions[questionId], null);
                    updateQuestionNavigator();
                    return;
                }
                
                // Otherwise fetch it from server
                loading.style.display = 'inline';
                errorMessage.textContent = '';
                
                fetch(`/get_question/${questionId}`)
                    .then(response => response.json())
                    .then(data => {
                        loading.style.display = 'none';
                        
                        if (data.error) {
                            errorMessage.textContent = data.error;
                            return;
                        }
                        
                        // Store in cache
                        allQuestions[questionId] = data.visualization_data;
                        
                        // Display question
                        displayVisualization(data.visualization_data, data.stats);
                        updateQuestionNavigator();
                    })
                    .catch(error => {
                        loading.style.display = 'none';
                        errorMessage.textContent = 'Error loading question: ' + error.message;
                    });
            }
            
            function displayVisualization(visualizationData, stats) {
                // Store current data
                currentVisualizationData = visualizationData;
                
                // Update cluster legend
                updateLegend(visualizationData.clusters);
                
                // Display thoughts
                displayThoughts(visualizationData.thoughts);
                
                // Update stats if provided
                if (stats) {
                    updateStats(stats);
                }
                
                // Update filter dropdown
                updateFilterDropdown(visualizationData.clusters);
            }
            
            function updateLegend(clusters) {
                legendGrid.innerHTML = '';
                
                clusters.forEach(cluster => {
                    const legendItem = document.createElement('div');
                    legendItem.className = 'legend-item';
                    
                    const colorBox = document.createElement('div');
                    colorBox.className = 'legend-color';
                    colorBox.style.backgroundColor = cluster.color;
                    
                    const description = document.createElement('div');
                    description.textContent = `Cluster ${cluster.id}: ${cluster.description} (${cluster.thought_count})`;
                    
                    legendItem.appendChild(colorBox);
                    legendItem.appendChild(description);
                    legendGrid.appendChild(legendItem);
                });
            }
            
            function updateFilterDropdown(clusters) {
                // Clear previous options except "Show All"
                while (clusterFilter.options.length > 1) {
                    clusterFilter.remove(1);
                }
                
                // Add cluster options
                clusters.forEach(cluster => {
                    const option = document.createElement('option');
                    option.value = cluster.id;
                    option.textContent = `Cluster ${cluster.id}: ${cluster.description}`;
                    clusterFilter.appendChild(option);
                });
            }
            
            function displayThoughts(thoughts) {
                thoughtGrid.innerHTML = '';
                
                thoughts.forEach(thought => {
                    const thoughtCard = document.createElement('div');
                    thoughtCard.className = 'thought-card';
                    thoughtCard.dataset.thoughtId = thought.id;
                    thoughtCard.dataset.clusters = thought.clusters.map(c => c.id).join(',');
                    
                    // Create color stripe
                    if (thought.all_colors.length > 1) {
                        // Multiple clusters - create segments
                        const multiStripe = document.createElement('div');
                        multiStripe.className = 'multi-color-stripe';
                        
                        thought.all_colors.forEach(color => {
                            const segment = document.createElement('div');
                            segment.className = 'stripe-segment';
                            segment.style.backgroundColor = color;
                            multiStripe.appendChild(segment);
                        });
                        
                        thoughtCard.appendChild(multiStripe);
                    } else {
                        // Single cluster - solid color
                        const stripe = document.createElement('div');
                        stripe.className = 'color-stripe';
                        stripe.style.backgroundColor = thought.primary_color;
                        thoughtCard.appendChild(stripe);
                    }
                    
                    // Create tooltip
                    const tooltip = document.createElement('div');
                    tooltip.className = 'cluster-tooltip';
                    tooltip.innerHTML = thought.clusters.map(c => 
                        `Cluster ${c.id}: ${c.description}`
                    ).join('<br>');
                    thoughtCard.appendChild(tooltip);
                    
                    // Add thought content
                    const thoughtId = document.createElement('div');
                    thoughtId.className = 'thought-id';
                    thoughtId.textContent = `Thought ${thought.id}`;
                    
                    const thoughtText = document.createElement('div');
                    thoughtText.className = 'thought-text';
                    thoughtText.textContent = thought.text;
                    
                    thoughtCard.appendChild(thoughtId);
                    thoughtCard.appendChild(thoughtText);
                    thoughtGrid.appendChild(thoughtCard);
                });
                
                // Apply initial filters
                filterThoughts();
            }
            
            function filterThoughts() {
                const selectedCluster = clusterFilter.value;
                const searchTerm = searchBox.value.trim().toLowerCase();
                
                // Get all thought cards
                const thoughtCards = document.querySelectorAll('.thought-card');
                
                thoughtCards.forEach(card => {
                    const clusters = card.dataset.clusters.split(',');
                    const thoughtText = card.querySelector('.thought-text').textContent.toLowerCase();
                    
                    // Apply cluster filter
                    const passesClusterFilter = selectedCluster === 'all' || clusters.includes(selectedCluster);
                    
                    // Apply search filter
                    const passesSearchFilter = !searchTerm || thoughtText.includes(searchTerm);
                    
                    // Show/hide based on filters
                    card.style.display = (passesClusterFilter && passesSearchFilter) ? 'block' : 'none';
                });
            }
            
            function updateQuestionNavigator() {
                // Update question indicator
                const correctness = (currentVisualizationData && currentVisualizationData.thoughts[0]?.correctness);
                const correctnessLabel = correctness === true ? '✔️ Correct' : correctness === false ? '❌ Incorrect' : '❓ Unknown';
                questionIndicator.textContent = `Question ${currentQuestionId + 1} of ${totalQuestions} (${correctnessLabel})`;
                
                // Update button states
                prevButton.disabled = currentQuestionId === 0;
                prevButton.classList.toggle('disabled', currentQuestionId === 0);
                
                nextButton.disabled = currentQuestionId === totalQuestions - 1;
                nextButton.classList.toggle('disabled', currentQuestionId === totalQuestions - 1);
            }
            
            function updateStats(stats) {
                let statsText = `
                    <p>
                        <strong>Total Clusters:</strong> ${stats.num_clusters}<br>
                        <strong>Total Thoughts:</strong> ${stats.num_thoughts}
                    </p>
                `;
                
                if (stats.shared_thoughts > 0) {
                    statsText += `<p><strong>Thoughts in Multiple Clusters:</strong> ${stats.shared_thoughts}</p>`;
                }
                
                statsContent.innerHTML = statsText;
            }
        });
    </script>
</body>
</html>
"""

with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write(index_html)

if __name__ == "__main__":
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)