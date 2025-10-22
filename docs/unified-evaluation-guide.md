# Unified Evaluation System

## Overview

The topic modeling project now uses a **unified evaluation system** that consolidates all model evaluation needs into a single, comprehensive script: `src/models/evaluate.py`. This eliminates the need for multiple evaluation scripts and provides consistent, comprehensive evaluation across all models.

## Why Unified Evaluation?

### Problems with Multiple Scripts
- **Code Duplication**: Similar evaluation logic scattered across multiple files
- **Inconsistency**: Different metrics and evaluation approaches
- **Maintenance Overhead**: Updates needed in multiple places
- **Confusion**: Users unsure which script to use for what purpose

### Benefits of Unified System
- **Single Source of Truth**: All evaluation logic in one place
- **Consistent Metrics**: Same evaluation approach for all models
- **Comprehensive Features**: Single model, batch, and auto-discovery modes
- **Better Visualization**: Multiple chart types and comparison views
- **Easy Maintenance**: One script to update and maintain

## Usage

### Command Line Interface

#### Auto-Discovery Mode (Recommended)
```bash
# Automatically discover and evaluate all available models
python src/models/evaluate.py --mode auto
```

#### Single Model Evaluation
```bash
# Evaluate a specific model
python src/models/evaluate.py --mode single \
    --model_type lda \
    --model_path artifacts/lda_model.pkl \
    --vectorizer_path artifacts/tfidf_vectorizer.pkl \
    --data_path artifacts/preprocessed_bbc_news.csv
```

#### Batch Evaluation
```bash
# Evaluate multiple models (requires configuration)
python src/models/evaluate.py --mode batch
```

### Programmatic Usage

#### Auto-Discovery
```python
from src.models.evaluate import auto_discover_and_evaluate

# Automatically find and evaluate all models
results = auto_discover_and_evaluate(
    data_path="artifacts/preprocessed_bbc_news.csv",
    artifacts_dir="artifacts",
    output_dir="artifacts/evaluation"
)
```

#### Single Model Evaluation
```python
from src.models.evaluate import evaluate_model

# Evaluate a specific model
results = evaluate_model(
    model_type="lda",
    model_path="artifacts/lda_model.pkl",
    vectorizer_path="artifacts/tfidf_vectorizer.pkl",
    data_path="artifacts/preprocessed_bbc_news.csv",
    output_dir="artifacts/evaluation"
)
```

#### Batch Model Evaluation
```python
from src.models.evaluate import evaluate_batch_models

# Define model configurations
model_configs = [
    {
        'model_type': 'lda',
        'model_path': 'artifacts/lda_model.pkl',
        'vectorizer_path': 'artifacts/tfidf_vectorizer.pkl',
        'name': 'LDA Model'
    },
    {
        'model_type': 'nmf',
        'model_path': 'artifacts/nmf_model.pkl',
        'vectorizer_path': 'artifacts/tfidf_vectorizer.pkl',
        'name': 'NMF Model'
    }
]

# Evaluate multiple models
results = evaluate_batch_models(
    model_configs=model_configs,
    data_path="artifacts/preprocessed_bbc_news.csv",
    output_dir="artifacts/evaluation"
)
```

#### Direct Class Usage
```python
from src.models.evaluate import TopicModelEvaluator, ModelComparison

# Single model evaluation
evaluator = TopicModelEvaluator(
    model_path="artifacts/lda_model.pkl",
    vectorizer_path="artifacts/tfidf_vectorizer.pkl",
    data_path="artifacts/preprocessed_bbc_news.csv",
    model_type="lda",
    output_dir="artifacts/evaluation"
)
metrics = evaluator.evaluate()

# Model comparison
comparison = ModelComparison("artifacts/evaluation")
results = comparison.compare_models(model_configs, data_path)
```

## Features

### Comprehensive Metrics
- **Coherence Score**: Topic interpretability
- **Topic Diversity**: Uniqueness of topics
- **Silhouette Score**: Clustering quality
- **Perplexity**: Model fit (LDA only)
- **Reconstruction Error**: Model fit (NMF only)
- **Topic Intrusion**: Topic quality assessment

### Visualization Types
- **Bar Charts**: Metric comparison across models
- **Radar Charts**: Multi-dimensional model comparison
- **Heatmaps**: Detailed metrics matrix
- **Topic Heatmaps**: Topic-word relationships

### Evaluation Modes
1. **Single Model**: Evaluate one specific model
2. **Batch Models**: Evaluate multiple configured models
3. **Auto-Discovery**: Automatically find and evaluate all models

## Output Files

The unified evaluation system generates:

### JSON Reports
- `{model_type}_evaluation.json`: Individual model results
- `comprehensive_comparison_results.json`: Batch comparison results

### Visualizations
- `comprehensive_model_comparison.png`: Main comparison chart
- `model_comparison_radar.png`: Multi-dimensional radar chart
- `model_metrics_heatmap.png`: Metrics heatmap
- `{model_type}_topic_heatmap.png`: Topic-specific heatmaps

## Migration from Old System

### Deprecated Scripts
- `src/models/compare_models.py` - Now redirects to unified system
- Individual evaluation scripts - Replaced by unified system

### Migration Steps
1. **Replace script calls**:
   ```bash
   # OLD
   python src/models/compare_models.py
   
   # NEW
   python src/models/evaluate.py --mode auto
   ```

2. **Update imports**:
   ```python
   # OLD
   from src.models.compare_models import evaluate_model_comprehensive
   
   # NEW
   from src.models.evaluate import auto_discover_and_evaluate
   ```

3. **Use new functions**:
   ```python
   # OLD
   results = evaluate_model_comprehensive(...)
   
   # NEW
   results = auto_discover_and_evaluate()
   ```

## Examples

### Quick Start
```bash
# Train models first
python src/models/train.py --model_type lda
python src/models/train.py --model_type nmf

# Evaluate all models
python src/models/evaluate.py --mode auto
```

### Custom Evaluation
```python
from src.models.evaluate import TopicModelEvaluator

# Create custom evaluator
evaluator = TopicModelEvaluator(
    model_path="my_model.pkl",
    vectorizer_path="my_vectorizer.pkl",
    data_path="my_data.csv",
    model_type="lda",
    output_dir="custom_evaluation"
)

# Run evaluation
metrics = evaluator.evaluate()
print(f"Coherence: {metrics['coherence_score']:.4f}")
```

## Troubleshooting

### Common Issues

1. **No models found**:
   - Ensure models are trained first
   - Check file paths in artifacts directory

2. **Import errors**:
   - Verify all dependencies are installed
   - Check Python path includes project root

3. **Evaluation failures**:
   - Check model files are not corrupted
   - Verify data format compatibility

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run evaluation with debug output
results = auto_discover_and_evaluate()
```

## Best Practices

1. **Use Auto-Discovery**: Let the system find models automatically
2. **Check Output Directory**: Ensure write permissions for results
3. **Review Visualizations**: Use generated charts for model comparison
4. **Save Results**: Keep evaluation reports for model comparison
5. **Regular Evaluation**: Run evaluation after model training

## Future Enhancements

- **Custom Metrics**: Add user-defined evaluation metrics
- **Model Selection**: Automatic best model recommendation
- **Interactive Visualizations**: Web-based comparison tools
- **Export Options**: Multiple output formats (CSV, Excel, etc.)
- **Performance Optimization**: Parallel evaluation for large models
