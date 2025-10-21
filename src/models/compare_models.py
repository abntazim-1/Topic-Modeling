import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.models.evaluate import TopicModelEvaluator, evaluate_topics_csv, compare_models_csv

def evaluate_model_comprehensive(model_type, model_path, vectorizer_path, data_path, output_dir):
    """Evaluate a model with comprehensive metrics including coherence, perplexity, and silhouette score."""
    try:
        # Load the model and vectorizer
        evaluator = TopicModelEvaluator(
            model_path=model_path,
            vectorizer_path=vectorizer_path,
            data_path=data_path,
            model_type=model_type,
            output_dir=output_dir
        )
        
        # Run comprehensive evaluation
        evaluator.evaluate()
        
        # Extract metrics
        metrics = evaluator.metrics
        return {
            'topic_diversity': metrics.get('topic_diversity', 0.0),
            'coherence_score': metrics.get('coherence_score', 0.0),
            'perplexity': metrics.get('perplexity', 0.0),
            'silhouette_score': metrics.get('silhouette_score', 0.0),
            'num_topics': metrics.get('num_topics', 0),
            'evaluation_time': metrics.get('evaluation_time', 0.0)
        }
    except Exception as e:
        print(f'Error evaluating {model_type}: {e}')
        return {
            'topic_diversity': 0.0,
            'coherence_score': 0.0,
            'perplexity': 0.0,
            'silhouette_score': 0.0,
            'num_topics': 0,
            'evaluation_time': 0.0
        }

def create_comprehensive_comparison(lda_metrics, nmf_metrics, output_dir):
    """Create a comprehensive comparison visualization with multiple metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for visualization
    metrics = ['Topic Diversity', 'Coherence Score', 'Perplexity', 'Silhouette Score']
    lda_scores = [
        lda_metrics.get('topic_diversity', 0.0),
        lda_metrics.get('coherence_score', 0.0),
        lda_metrics.get('perplexity', 0.0),
        lda_metrics.get('silhouette_score', 0.0) if lda_metrics.get('silhouette_score') is not None else 0.0
    ]
    nmf_scores = [
        nmf_metrics.get('topic_diversity', 0.0),
        nmf_metrics.get('coherence_score', 0.0),
        nmf_metrics.get('perplexity', 0.0),
        nmf_metrics.get('silhouette_score', 0.0) if nmf_metrics.get('silhouette_score') is not None else 0.0
    ]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comprehensive Model Comparison: LDA vs NMF', fontsize=16, fontweight='bold')
    
    # Plot 1: Topic Diversity
    axes[0, 0].bar(['LDA', 'NMF'], [lda_scores[0], nmf_scores[0]], color=['skyblue', 'lightcoral'])
    axes[0, 0].set_title('Topic Diversity')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate([lda_scores[0], nmf_scores[0]]):
        axes[0, 0].text(i, v + 0.02, f"{v:.4f}", ha='center')
    
    # Plot 2: Coherence Score
    axes[0, 1].bar(['LDA', 'NMF'], [lda_scores[1], nmf_scores[1]], color=['skyblue', 'lightcoral'])
    axes[0, 1].set_title('Coherence Score')
    axes[0, 1].set_ylabel('Score')
    for i, v in enumerate([lda_scores[1], nmf_scores[1]]):
        axes[0, 1].text(i, v + 0.02, f"{v:.4f}", ha='center')
    
    # Plot 3: Perplexity (lower is better)
    axes[1, 0].bar(['LDA', 'NMF'], [lda_scores[2], nmf_scores[2]], color=['skyblue', 'lightcoral'])
    axes[1, 0].set_title('Perplexity (Lower is Better)')
    axes[1, 0].set_ylabel('Perplexity')
    for i, v in enumerate([lda_scores[2], nmf_scores[2]]):
        axes[1, 0].text(i, v + 0.02, f"{v:.2f}", ha='center')
    
    # Plot 4: Silhouette Score
    axes[1, 1].bar(['LDA', 'NMF'], [lda_scores[3], nmf_scores[3]], color=['skyblue', 'lightcoral'])
    axes[1, 1].set_title('Silhouette Score')
    axes[1, 1].set_ylabel('Score')
    for i, v in enumerate([lda_scores[3], nmf_scores[3]]):
        axes[1, 1].text(i, v + 0.02, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "comprehensive_model_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return comparison_path

# Evaluate both models with comprehensive metrics
print('Evaluating both LDA and NMF models with comprehensive metrics...')
print('=' * 80)

# Check if model files exist
lda_model_path = 'artifacts/lda_model.pkl'
nmf_model_path = 'artifacts/nmf_model.pkl'
vectorizer_path = 'artifacts/tfidf_vectorizer.pkl'
data_path = 'artifacts/preprocessed_bbc_news.csv'

# Evaluate LDA with comprehensive metrics
print('Evaluating LDA model...')
if os.path.exists(lda_model_path) and os.path.exists(vectorizer_path) and os.path.exists(data_path):
    lda_metrics = evaluate_model_comprehensive(
        'lda', lda_model_path, vectorizer_path, data_path, 'artifacts/evaluation'
    )
    print(f'LDA - Topic Diversity: {lda_metrics["topic_diversity"]:.4f}')
    print(f'LDA - Coherence Score: {lda_metrics["coherence_score"]:.4f}')
    print(f'LDA - Perplexity: {lda_metrics["perplexity"]:.2f}')
    silhouette_score = lda_metrics["silhouette_score"]
    print(f'LDA - Silhouette Score: {silhouette_score:.4f}' if silhouette_score is not None else 'LDA - Silhouette Score: N/A')
    print(f'LDA - Number of Topics: {lda_metrics["num_topics"]}')
else:
    print('LDA model files not found, using CSV-based evaluation...')
    try:
        lda_result = evaluate_topics_csv('lda', 'artifacts/lda_topics.csv', 'artifacts/evaluation')
        lda_metrics = {
            'topic_diversity': lda_result['metrics']['topic_diversity'],
            'coherence_score': 0.0,
            'perplexity': 0.0,
            'silhouette_score': 0.0,
            'num_topics': lda_result['metrics']['num_topics']
        }
        print(f'LDA - Topic Diversity: {lda_metrics["topic_diversity"]:.4f}')
    except Exception as e:
        print(f'LDA evaluation error: {e}')
        lda_metrics = {
            'topic_diversity': 0.0,
            'coherence_score': 0.0,
            'perplexity': 0.0,
            'silhouette_score': 0.0,
            'num_topics': 0
        }

# Evaluate NMF with comprehensive metrics
print('\nEvaluating NMF model...')
if os.path.exists(nmf_model_path) and os.path.exists(vectorizer_path) and os.path.exists(data_path):
    nmf_metrics = evaluate_model_comprehensive(
        'nmf', nmf_model_path, vectorizer_path, data_path, 'artifacts/evaluation'
    )
    print(f'NMF - Topic Diversity: {nmf_metrics["topic_diversity"]:.4f}')
    print(f'NMF - Coherence Score: {nmf_metrics["coherence_score"]:.4f}')
    print(f'NMF - Perplexity: {nmf_metrics["perplexity"]:.2f}')
    silhouette_score = nmf_metrics["silhouette_score"]
    print(f'NMF - Silhouette Score: {silhouette_score:.4f}' if silhouette_score is not None else 'NMF - Silhouette Score: N/A')
    print(f'NMF - Number of Topics: {nmf_metrics["num_topics"]}')
else:
    print('NMF model files not found, using CSV-based evaluation...')
    try:
        nmf_result = evaluate_topics_csv('nmf', 'artifacts/evaluation/nmf_topics.csv', 'artifacts/evaluation')
        nmf_metrics = {
            'topic_diversity': nmf_result['metrics']['topic_diversity'],
            'coherence_score': 0.0,
            'perplexity': 0.0,
            'silhouette_score': 0.0,
            'num_topics': nmf_result['metrics']['num_topics']
        }
        print(f'NMF - Topic Diversity: {nmf_metrics["topic_diversity"]:.4f}')
    except Exception as e:
        print(f'NMF evaluation error: {e}')
        nmf_metrics = {
            'topic_diversity': 0.0,
            'coherence_score': 0.0,
            'perplexity': 0.0,
            'silhouette_score': 0.0,
            'num_topics': 0
        }

# Create comprehensive comparison visualization
print('\nCreating comprehensive comparison visualization...')
comparison_path = create_comprehensive_comparison(lda_metrics, nmf_metrics, 'artifacts/evaluation')
print(f'Comparison visualization saved to: {comparison_path}')

# Save detailed comparison results
comparison_results = {
    'lda_metrics': lda_metrics,
    'nmf_metrics': nmf_metrics,
    'comparison_summary': {
        'topic_diversity_winner': 'LDA' if lda_metrics['topic_diversity'] > nmf_metrics['topic_diversity'] else 'NMF',
        'coherence_winner': 'LDA' if lda_metrics['coherence_score'] > nmf_metrics['coherence_score'] else 'NMF',
        'perplexity_winner': 'LDA' if lda_metrics['perplexity'] < nmf_metrics['perplexity'] else 'NMF',
        'silhouette_winner': 'LDA' if (lda_metrics['silhouette_score'] is not None and nmf_metrics['silhouette_score'] is not None and lda_metrics['silhouette_score'] > nmf_metrics['silhouette_score']) or (lda_metrics['silhouette_score'] is not None and nmf_metrics['silhouette_score'] is None) else 'NMF' if (nmf_metrics['silhouette_score'] is not None and lda_metrics['silhouette_score'] is None) or (lda_metrics['silhouette_score'] is not None and nmf_metrics['silhouette_score'] is not None and nmf_metrics['silhouette_score'] > lda_metrics['silhouette_score']) else 'Tie'
    }
}

results_path = 'artifacts/evaluation/comprehensive_comparison_results.json'
with open(results_path, 'w') as f:
    json.dump(comparison_results, f, indent=4)

print('=' * 80)
print('COMPREHENSIVE EVALUATION SUMMARY:')
print('=' * 80)
print(f'LDA Metrics:')
print(f'  - Topic Diversity: {lda_metrics["topic_diversity"]:.4f}')
print(f'  - Coherence Score: {lda_metrics["coherence_score"]:.4f}')
print(f'  - Perplexity: {lda_metrics["perplexity"]:.2f}')
lda_silhouette = lda_metrics["silhouette_score"]
print(f'  - Silhouette Score: {lda_silhouette:.4f}' if lda_silhouette is not None else '  - Silhouette Score: N/A')
print(f'  - Number of Topics: {lda_metrics["num_topics"]}')

print(f'\nNMF Metrics:')
print(f'  - Topic Diversity: {nmf_metrics["topic_diversity"]:.4f}')
print(f'  - Coherence Score: {nmf_metrics["coherence_score"]:.4f}')
print(f'  - Perplexity: {nmf_metrics["perplexity"]:.2f}')
nmf_silhouette = nmf_metrics["silhouette_score"]
print(f'  - Silhouette Score: {nmf_silhouette:.4f}' if nmf_silhouette is not None else '  - Silhouette Score: N/A')
print(f'  - Number of Topics: {nmf_metrics["num_topics"]}')

print('\nWINNER BY METRIC:')
print(f'  - Topic Diversity: {comparison_results["comparison_summary"]["topic_diversity_winner"]}')
print(f'  - Coherence Score: {comparison_results["comparison_summary"]["coherence_winner"]}')
print(f'  - Perplexity (lower is better): {comparison_results["comparison_summary"]["perplexity_winner"]}')
print(f'  - Silhouette Score: {comparison_results["comparison_summary"]["silhouette_winner"]}')

print('=' * 80)
print('Evaluation completed!')
print(f'Detailed results saved to: {results_path}')
print(f'Visualization saved to: {comparison_path}')
