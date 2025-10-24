# Topic Modeling Pipeline

A comprehensive, production-ready pipeline for topic modeling using LDA and NMF algorithms with MLOps best practices.

## Overview

This pipeline orchestrates the complete topic modeling workflow from data ingestion through model training, evaluation, and artifact saving. It follows MLOps best practices with structured logging, error handling, and modular design.

## Features

- **Complete Workflow**: End-to-end pipeline from raw data to trained models
- **Dual Model Support**: Both LDA and NMF topic modeling algorithms
- **Comprehensive Evaluation**: Multiple metrics including coherence, diversity, and quality measures
- **Production Ready**: Structured logging, error handling, and artifact management
- **Modular Design**: Clean separation of concerns with reusable components
- **MLOps Best Practices**: Configuration management, logging, and monitoring

## Pipeline Stages

### 1. Load Configuration
- Loads YAML configuration from `config/config.yaml`
- Sets up structured logging with console and file output
- Creates necessary output directories
- Validates configuration parameters

### 2. Load and Preprocess Data
- **Data Ingestion**: Loads raw data with validation
- **Text Preprocessing**: Cleaning, tokenization, and n-gram generation
- **Quality Checks**: Data validation and quality metrics
- **Artifact Saving**: Saves preprocessed data for reuse

### 3. Train Models
- **LDA Training**: Latent Dirichlet Allocation with optimized hyperparameters
- **NMF Training**: Non-negative Matrix Factorization with appropriate settings
- **Model Validation**: Quality checks and validation metrics
- **Artifact Persistence**: Saves trained models and vectorizers

### 4. Evaluate Models
- **Comprehensive Metrics**: Coherence, diversity, silhouette scores
- **Model Comparison**: Side-by-side evaluation and ranking
- **Visualization**: Charts and plots for model comparison
- **Report Generation**: Detailed evaluation reports

### 5. Save Artifacts
- **Model Artifacts**: Trained models, vectorizers, and metadata
- **Evaluation Results**: Metrics, reports, and visualizations
- **Pipeline Metadata**: Execution logs and configuration snapshots

## Usage

### Command Line Interface

```bash
# Basic usage with default settings
python src/pipelines/run_pipeline.py

# Custom configuration
python src/pipelines/run_pipeline.py --config config/my_config.yaml

# Custom output directory
python src/pipelines/run_pipeline.py --output artifacts/experiment_1

# Debug logging
python src/pipelines/run_pipeline.py --log-level DEBUG
```

### Programmatic Usage

```python
from src.pipelines.run_pipeline import TopicModelingPipeline

# Initialize pipeline
pipeline = TopicModelingPipeline(
    config_path="config/config.yaml",
    output_dir="artifacts/topic_models",
    log_level="INFO"
)

# Run complete pipeline
pipeline.run_pipeline()

# Or run individual stages
pipeline.load_configuration()
pipeline.load_and_preprocess_data()
pipeline.train_models()
pipeline.evaluate_models()
pipeline.save_artifacts()
```

## Configuration

The pipeline uses YAML configuration files. Example configuration:

```yaml
data:
  csv_path: artifacts/bbc-news-data.csv
  expected_columns: [category, text]
  delimiter: ','
  encoding: 'utf-8'
  allow_duplicates: false

preprocessing:
  text_column: "content"
  stopwords: ['said', 'mr', 'say', 'also', 'would', 'one', 'two', 'us']
  spacy_model: "en_core_web_lg"
  min_token_length: 2
  gen_bigrams: true
  gen_trigrams: true

training:
  num_topics: 10
  random_state: 42

logging:
  level: 'INFO'
```

## Output Structure

```
artifacts/topic_models/
├── models/
│   ├── lda_model.pkl              # Trained LDA model
│   ├── nmf_model.pkl              # Trained NMF model
│   └── tfidf_vectorizer.pkl       # TF-IDF vectorizer
├── evaluation/
│   ├── lda_evaluation.json        # LDA evaluation metrics
│   ├── nmf_evaluation.json        # NMF evaluation metrics
│   ├── comprehensive_comparison_results.json
│   ├── comprehensive_model_comparison.png
│   ├── model_comparison_radar.png
│   └── model_metrics_heatmap.png
├── preprocessed_data/
│   └── preprocessed_bbc_news.csv  # Preprocessed dataset
└── pipeline_metadata.json         # Pipeline execution metadata
```

## Dependencies

### Core Dependencies
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical operations
- `scikit-learn`: Machine learning algorithms
- `gensim`: Topic modeling (LDA)
- `spacy`: Natural language processing
- `matplotlib`: Data visualization
- `seaborn`: Statistical data visualization
- `pyyaml`: Configuration file parsing

### Optional Dependencies
- `wordcloud`: Word cloud generation
- `tqdm`: Progress bars
- `plotly`: Interactive visualizations

## Error Handling

The pipeline includes comprehensive error handling:

- **AppException**: Base exception for application errors
- **DataValidationError**: Data quality and validation issues
- **Graceful Failures**: Detailed error logging with tracebacks
- **Recovery**: Checkpoint-based recovery for long-running processes

## Logging

Structured logging with multiple levels:

- **Console Output**: Real-time progress monitoring
- **File Logging**: Rotating log files with detailed information
- **Structured Format**: Timestamps, log levels, and source locations
- **Configurable Levels**: DEBUG, INFO, WARNING, ERROR

## Model Evaluation Metrics

### LDA Metrics
- **Coherence Score**: Topic interpretability (c_v measure)
- **Perplexity**: Model fit quality
- **Topic Diversity**: Word uniqueness across topics
- **Silhouette Score**: Clustering quality

### NMF Metrics
- **Coherence Score**: Topic interpretability
- **Reconstruction Error**: Matrix factorization quality
- **Topic Diversity**: Word uniqueness across topics
- **Silhouette Score**: Clustering quality

### Comparative Analysis
- **Side-by-side Comparison**: Direct metric comparison
- **Visualization**: Charts and plots for model comparison
- **Ranking**: Best model identification per metric
- **Comprehensive Reports**: Detailed analysis and recommendations

## Best Practices

### MLOps Principles
- **Reproducibility**: Fixed random seeds and version control
- **Modularity**: Clean separation of concerns
- **Monitoring**: Comprehensive logging and metrics
- **Artifact Management**: Organized output structure
- **Error Handling**: Graceful failure management

### Code Quality
- **Type Hints**: Full type annotation support
- **Documentation**: Comprehensive docstrings and comments
- **Testing**: Unit and integration test support
- **Linting**: Code quality enforcement
- **Configuration**: External configuration management

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **spaCy Model Not Found**
   ```bash
   python -m spacy download en_core_web_lg
   ```

3. **Memory Issues**
   - Reduce batch size in preprocessing
   - Use smaller vocabulary size
   - Process data in chunks

4. **Configuration Errors**
   - Check YAML syntax
   - Validate file paths
   - Ensure required fields are present

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
python src/pipelines/run_pipeline.py --log-level DEBUG
```

## Contributing

1. Follow the existing code structure
2. Add comprehensive tests for new features
3. Update documentation for changes
4. Ensure all linting checks pass
5. Follow MLOps best practices

## License

This project is part of the Topic Modeling Project. See LICENSE for details.

## Support

For issues and questions:
1. Check the logs for detailed error information
2. Review the configuration files
3. Ensure all dependencies are installed
4. Check the troubleshooting section above
