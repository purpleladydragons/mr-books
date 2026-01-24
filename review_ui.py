"""
Flask web UI for reviewing and correcting LLM comparisons.
"""

from flask import Flask, render_template_string, request, redirect, url_for, jsonify


# HTML template for the review UI
REVIEW_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Comparison Review - {{ comparison.id }}</title>
    <style>
        * {
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .position {
            font-size: 18px;
            font-weight: bold;
        }
        .nav-buttons {
            display: flex;
            gap: 10px;
        }
        .nav-btn {
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            border: 1px solid #ddd;
            background: white;
            border-radius: 4px;
        }
        .nav-btn:hover {
            background: #f0f0f0;
        }
        .nav-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .keyboard-hint {
            font-size: 12px;
            color: #666;
        }
        .comparison-container {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        .review-panel {
            flex: 1;
            background: white;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 20px;
            position: relative;
        }
        .review-panel.winner {
            border: 3px solid #4CAF50;
        }
        .review-panel.loser {
            border: 3px solid #f44336;
        }
        .review-label {
            position: absolute;
            top: -12px;
            left: 20px;
            background: white;
            padding: 0 10px;
            font-weight: bold;
            font-size: 18px;
        }
        .review-panel.winner .review-label {
            color: #4CAF50;
        }
        .review-panel.loser .review-label {
            color: #f44336;
        }
        .book-title {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }
        .context-text {
            max-height: 400px;
            overflow-y: auto;
            white-space: pre-wrap;
            font-size: 14px;
            line-height: 1.6;
            color: #333;
        }
        .winner-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 10px;
        }
        .winner-badge.winner {
            background: #e8f5e9;
            color: #4CAF50;
        }
        .winner-badge.loser {
            background: #ffebee;
            color: #f44336;
        }
        .actions {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 20px;
        }
        .correct-btn {
            padding: 15px 30px;
            font-size: 16px;
            cursor: pointer;
            border: 2px solid #2196F3;
            background: #2196F3;
            color: white;
            border-radius: 4px;
        }
        .correct-btn:hover {
            background: #1976D2;
        }
        .corrected-badge {
            display: inline-block;
            padding: 5px 15px;
            background: #fff3e0;
            color: #ef6c00;
            border-radius: 20px;
            font-size: 14px;
            font-weight: bold;
        }
        .stats-panel {
            background: white;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 20px;
        }
        .stats-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
        }
        .stat-item {
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .stat-label {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="position">
            Comparison {{ current_index + 1 }} of {{ total }}
            {% if comparison.manually_corrected %}
            <span class="corrected-badge">Manually Corrected</span>
            {% endif %}
        </div>
        <div class="nav-buttons">
            <button class="nav-btn" onclick="navigate('prev')" {% if current_index == 0 %}disabled{% endif %}>
                Previous
            </button>
            <button class="nav-btn" onclick="navigate('next')" {% if current_index == total - 1 %}disabled{% endif %}>
                Next
            </button>
        </div>
        <div class="keyboard-hint">
            Use arrow keys: &larr; Previous | Next &rarr;
        </div>
    </div>

    <div class="comparison-container">
        <div class="review-panel {% if comparison.winner_id == comparison.mention_a_id %}winner{% elif comparison.winner_id == comparison.mention_b_id %}loser{% endif %}">
            <div class="review-label">Review A</div>
            <div class="book-title">
                {{ comparison.book_a_title }}
                {% if comparison.winner_id == comparison.mention_a_id %}
                <span class="winner-badge winner">WINNER</span>
                {% elif comparison.winner_id == comparison.mention_b_id %}
                <span class="winner-badge loser">LOSER</span>
                {% endif %}
            </div>
            <div class="context-text">{{ comparison.context_a }}</div>
        </div>

        <div class="review-panel {% if comparison.winner_id == comparison.mention_b_id %}winner{% elif comparison.winner_id == comparison.mention_a_id %}loser{% endif %}">
            <div class="review-label">Review B</div>
            <div class="book-title">
                {{ comparison.book_b_title }}
                {% if comparison.winner_id == comparison.mention_b_id %}
                <span class="winner-badge winner">WINNER</span>
                {% elif comparison.winner_id == comparison.mention_a_id %}
                <span class="winner-badge loser">LOSER</span>
                {% endif %}
            </div>
            <div class="context-text">{{ comparison.context_b }}</div>
        </div>
    </div>

    <div class="actions">
        <form action="{{ url_for('correct_comparison_route', comparison_id=comparison.id) }}" method="POST">
            <button type="submit" class="correct-btn">
                {% if comparison.winner_id == comparison.mention_a_id %}
                    Correct: Set B as Winner
                {% elif comparison.winner_id == comparison.mention_b_id %}
                    Correct: Set A as Winner
                {% else %}
                    No winner recorded
                {% endif %}
            </button>
        </form>
    </div>

    <div class="stats-panel">
        <div class="stats-title">Analytics</div>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-value">{{ stats.total_comparisons }}</div>
                <div class="stat-label">Total Comparisons</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{{ stats.manually_corrected }}</div>
                <div class="stat-label">Manually Corrected</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{{ "%.1f"|format(stats.correction_rate) }}%</div>
                <div class="stat-label">Correction Rate</div>
            </div>
        </div>
    </div>

    <script>
        // Store comparison IDs for navigation
        const comparisonIds = {{ comparison_ids|tojson }};
        const currentIndex = {{ current_index }};

        function navigate(direction) {
            let newIndex;
            if (direction === 'prev' && currentIndex > 0) {
                newIndex = currentIndex - 1;
            } else if (direction === 'next' && currentIndex < comparisonIds.length - 1) {
                newIndex = currentIndex + 1;
            } else {
                return;
            }
            window.location.href = '/comparison/' + comparisonIds[newIndex];
        }

        // Keyboard navigation
        document.addEventListener('keydown', function(e) {
            if (e.key === 'ArrowLeft') {
                navigate('prev');
            } else if (e.key === 'ArrowRight') {
                navigate('next');
            }
        });
    </script>
</body>
</html>
'''

# Simple index page that redirects to first comparison
INDEX_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Comparison Review UI</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background: #f5f5f5;
        }
        .container {
            text-align: center;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        h1 {
            margin-bottom: 20px;
        }
        .stats {
            margin-bottom: 30px;
            color: #666;
        }
        .start-btn {
            padding: 15px 30px;
            font-size: 18px;
            cursor: pointer;
            border: none;
            background: #2196F3;
            color: white;
            border-radius: 4px;
            text-decoration: none;
        }
        .start-btn:hover {
            background: #1976D2;
        }
        .no-comparisons {
            color: #f44336;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Comparison Review UI</h1>
        {% if total > 0 %}
        <div class="stats">
            <p>{{ total }} comparisons to review</p>
            <p>{{ corrected }} already corrected ({{ "%.1f"|format(correction_rate) }}%)</p>
        </div>
        <a href="{{ url_for('view_comparison', comparison_id=first_id) }}" class="start-btn">
            Start Reviewing
        </a>
        {% else %}
        <p class="no-comparisons">No comparisons found. Run 'python main.py rank' first to generate comparisons.</p>
        {% endif %}
    </div>
</body>
</html>
'''


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)

    @app.route('/')
    def index():
        """Show the index page with stats and start button."""
        from db import get_review_stats, get_comparison_ids

        stats = get_review_stats()
        comparison_ids = get_comparison_ids()

        return render_template_string(
            INDEX_TEMPLATE,
            total=stats['total_comparisons'],
            corrected=stats['manually_corrected'],
            correction_rate=stats['correction_rate'],
            first_id=comparison_ids[0] if comparison_ids else None
        )

    @app.route('/comparison/<int:comparison_id>')
    def view_comparison(comparison_id):
        """View a single comparison."""
        from db import get_comparison_for_review, get_comparison_ids, get_review_stats

        comparison = get_comparison_for_review(comparison_id)
        if not comparison:
            return "Comparison not found", 404

        comparison_ids = get_comparison_ids()
        current_index = comparison_ids.index(comparison_id) if comparison_id in comparison_ids else 0
        stats = get_review_stats()

        return render_template_string(
            REVIEW_TEMPLATE,
            comparison=comparison,
            comparison_ids=comparison_ids,
            current_index=current_index,
            total=len(comparison_ids),
            stats=stats
        )

    @app.route('/comparison/<int:comparison_id>/correct', methods=['POST'])
    def correct_comparison_route(comparison_id):
        """Correct a comparison by flipping the winner."""
        from db import get_comparison_for_review, correct_comparison

        comparison = get_comparison_for_review(comparison_id)
        if not comparison:
            return "Comparison not found", 404

        # Flip the winner
        if comparison['winner_id'] == comparison['mention_a_id']:
            new_winner = comparison['mention_b_id']
        else:
            new_winner = comparison['mention_a_id']

        correct_comparison(comparison_id, new_winner)

        return redirect(url_for('view_comparison', comparison_id=comparison_id))

    return app


def run_review_ui(port=5000):
    """Run the Flask review UI server."""
    from db import init_db

    init_db()
    app = create_app()
    print(f"Starting review UI at http://localhost:{port}")
    print("Press Ctrl+C to stop")
    app.run(host='0.0.0.0', port=port, debug=False)
