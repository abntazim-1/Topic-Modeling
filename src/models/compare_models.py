"""
DEPRECATED: This script is now replaced by the unified evaluation system.
Use src/models/evaluate.py instead for all evaluation needs.

This script now redirects to the unified evaluation system.
"""

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

# Import the unified evaluation system
from src.models.evaluate import auto_discover_and_evaluate, evaluate_batch_models

def redirect_to_unified_evaluation():
    """
    Redirect to the unified evaluation system.
    This function demonstrates how to use the new system.
    """
    print("=" * 80)
    print("DEPRECATION NOTICE")
    print("=" * 80)
    print("This script (compare_models.py) is now DEPRECATED.")
    print("Please use the unified evaluation system instead:")
    print()
    print("NEW WAY (Recommended):")
    print("  python src/models/evaluate.py --mode auto")
    print()
    print("Or programmatically:")
    print("  from src.models.evaluate import auto_discover_and_evaluate")
    print("  results = auto_discover_and_evaluate()")
    print()
    print("The unified system provides:")
    print("  ✓ Single script for all evaluation needs")
    print("  ✓ Automatic model discovery")
    print("  ✓ Comprehensive comparison and visualization")
    print("  ✓ Consistent metrics across all models")
    print("  ✓ Better error handling and logging")
    print()
    print("=" * 80)
    print("Running unified evaluation system instead...")
    print("=" * 80)
    
    # Run the unified evaluation system
    try:
        results = auto_discover_and_evaluate(
            data_path="artifacts/preprocessed_bbc_news.csv",
            artifacts_dir="artifacts",
            output_dir="artifacts/evaluation"
        )
        
        if 'error' in results:
            print(f"Evaluation failed: {results['error']}")
        else:
            print("Unified evaluation completed successfully!")
            print(f"Results saved to: artifacts/evaluation/")
            
            if 'individual_results' in results:
                print("\nEvaluation Summary:")
                for model_name, metrics in results['individual_results'].items():
                    if 'error' not in metrics:
                        print(f"\n{model_name}:")
                        print(f"  - Topic Diversity: {metrics.get('topic_diversity', 0.0):.4f}")
                        print(f"  - Coherence Score: {metrics.get('coherence_score', 0.0):.4f}")
                        print(f"  - Perplexity: {metrics.get('perplexity', 0.0):.2f}")
                        print(f"  - Silhouette Score: {metrics.get('silhouette_score', 0.0):.4f}")
                        print(f"  - Number of Topics: {metrics.get('num_topics', 0)}")
                    else:
                        print(f"\n{model_name}: Error - {metrics['error']}")
                        
    except Exception as e:
        print(f"Error running unified evaluation: {e}")
        print("Please ensure models are trained first by running:")
        print("  python src/models/train.py --model_type lda")
        print("  python src/models/train.py --model_type nmf")

if __name__ == "__main__":
    # Redirect to the unified evaluation system
    redirect_to_unified_evaluation()
