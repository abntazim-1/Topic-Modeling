# 🎯 Topic Modeling CLI

A professional command-line interface for the Topic Modeling Project that provides easy access to training, evaluation, and prediction functionalities.

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements

# Make CLI executable (optional)
chmod +x cli.py
```

### Basic Usage

```bash
# Show available commands
python cli.py --help

# Train a model
python cli.py train --model lda --topics 10

# Evaluate a model
python cli.py evaluate --model lda

# Predict topics for text
python cli.py predict --model lda --text "Your news article text here"

# Show available models
python cli.py show-models
```

## 📋 Commands

### 🏋️ Train

Train topic models (LDA or NMF) with configurable parameters.

```bash
python cli.py train --model lda --topics 10
python cli.py train --model nmf --topics 15 --data-path data/news.csv
python cli.py train --model lda --force  # Retrain existing model
```

**Options:**
- `--model, -m`: Model type (lda, nmf) **[required]**
- `--topics, -t`: Number of topics (3-50) **[optional, uses model default]**
- `--data-path, -d`: Path to training data CSV **[optional, uses config default]**
- `--output-dir, -o`: Output directory for models **[default: artifacts]**
- `--vectorizer, -v`: Vectorizer type (tfidf, bow) **[default: tfidf]**
- `--force, -f`: Force retraining even if model exists **[flag]**

### 📊 Evaluate

Evaluate trained models and generate comprehensive metrics.

```bash
python cli.py evaluate --model lda
python cli.py evaluate --model nmf --metrics coherence_score --metrics topic_diversity
python cli.py evaluate --model lda --model-path artifacts/custom_model.pkl
```

**Options:**
- `--model, -m`: Model to evaluate (lda, nmf) **[required]**
- `--model-path, -p`: Path to trained model **[optional, uses config default]**
- `--data-path, -d`: Path to evaluation data **[optional, uses config default]**
- `--output-dir, -o`: Output directory for results **[default: artifacts/evaluation]**
- `--metrics, -me`: Specific metrics to compute **[can be used multiple times]**
- `--save-plots, -s`: Save evaluation plots **[default: true]**

### 🔮 Predict

Predict topics for input text using trained models.

```bash
python cli.py predict --model lda --text "Your news article text here"
python cli.py predict --model nmf --text "Technology news" --topics 3 --format json
python cli.py predict --model lda --text "Sports update" --model-path artifacts/custom_model.pkl
```

**Options:**
- `--model, -m`: Model to use (lda, nmf) **[required]**
- `--text, -t`: Text to analyze **[required]**
- `--topics, -k`: Number of top topics (1-20) **[default: 5]**
- `--model-path, -p`: Path to trained model **[optional, uses config default]**
- `--vectorizer-path, -v`: Path to vectorizer **[optional, uses config default]**
- `--format, -f`: Output format (table, json, simple) **[default: table]**

### 📋 Show Models

Display information about available trained models.

```bash
python cli.py show-models
python cli.py show-models --model lda
python cli.py show-models --format json
```

**Options:**
- `--model, -m`: Specific model to show (lda, nmf, all) **[default: all]**
- `--format, -f`: Output format (table, json, simple) **[default: table]**

### 🔧 Preprocess

Preprocess raw text data using the same pipeline as training.

```bash
python cli.py preprocess --data-path data/raw_news.csv
python cli.py preprocess --data-path data/news.csv --output-path processed/news.csv
python cli.py preprocess --data-path data/news.csv --no-bigrams --no-trigrams
```

**Options:**
- `--data-path, -d`: Path to raw data CSV **[optional, uses config default]**
- `--output-path, -o`: Output path for processed data **[default: artifacts/preprocessed_data.csv]**
- `--text-column, -c`: Name of text column **[default: content]**
- `--spacy-model, -s`: SpaCy model to use **[default: en_core_web_lg]**
- `--min-token-length, -l`: Minimum token length **[default: 2]**
- `--gen-bigrams/--no-bigrams`: Generate bigrams **[default: true]**
- `--gen-trigrams/--no-trigrams`: Generate trigrams **[default: true]**

## ⚙️ Configuration

The CLI uses configuration files to manage settings:

### Main Configuration (`config/config.yaml`)

```yaml
data:
  csv_path: artifacts/bbc-news-data.csv
  expected_columns: [category, text]
  delimiter: ','
  encoding: 'utf-8'
  allow_duplicates: false

api:
  host: '0.0.0.0'
  port: 5000
  debug: false
  models_path: 'artifacts'
  max_topics: 20
  default_topics: 10

logging:
  level: 'INFO'
```

### Model Registry (`config/model_registry.yaml`)

Defines available models, their parameters, and evaluation metrics.

### Logging Configuration (`config/logging.yaml`)

Configures logging levels, formats, and output destinations.

## 🎨 Output Formats

### Table Format (Default)
Clean, human-readable tables with proper alignment and formatting.

### JSON Format
Machine-readable JSON output for integration with other tools.

### Simple Format
Minimal output with essential information only.

## 📊 Example Workflows

### Complete Training and Evaluation Pipeline

```bash
# 1. Preprocess data
python cli.py preprocess --data-path data/raw_news.csv

# 2. Train LDA model
python cli.py train --model lda --topics 10

# 3. Train NMF model
python cli.py train --model nmf --topics 10

# 4. Evaluate both models
python cli.py evaluate --model lda
python cli.py evaluate --model nmf

# 5. Show model status
python cli.py show-models
```

### Batch Prediction

```bash
# Predict topics for multiple texts
python cli.py predict --model lda --text "Technology news about AI" --format json
python cli.py predict --model lda --text "Sports news about football" --format json
python cli.py predict --model lda --text "Political news about elections" --format json
```

## 🔧 Advanced Usage

### Custom Configuration

```bash
# Use custom config file
python cli.py --config custom_config.yaml train --model lda

# Set custom log level
python cli.py --log-level DEBUG train --model lda --topics 10
```

### Integration with Scripts

```python
# Use CLI programmatically
import subprocess

# Train model
result = subprocess.run([
    'python', 'cli.py', 'train', 
    '--model', 'lda', 
    '--topics', '10'
], capture_output=True, text=True)

# Evaluate model
result = subprocess.run([
    'python', 'cli.py', 'evaluate', 
    '--model', 'lda'
], capture_output=True, text=True)
```

## 🐛 Troubleshooting

### Common Issues

1. **Model not found**: Ensure the model has been trained first
2. **Data file not found**: Check the data path in configuration
3. **Memory issues**: Reduce the number of topics or use smaller datasets
4. **SpaCy model not found**: Install the required SpaCy model

### Debug Mode

```bash
# Enable debug logging
python cli.py --log-level DEBUG train --model lda --topics 10
```

### Validation

```bash
# Check model status
python cli.py show-models

# Verify configuration
python cli.py --config config/config.yaml show-models
```

## 📈 Performance Tips

1. **Use appropriate number of topics**: Start with 10-15 topics
2. **Preprocess data once**: Use the preprocess command before training
3. **Monitor memory usage**: Large datasets may require more RAM
4. **Use force flag carefully**: Only retrain when necessary

## 🤝 Contributing

The CLI is designed to be easily extensible. To add new commands:

1. Add the command function to `src/cli/cli.py`
2. Update the configuration files if needed
3. Add tests for the new functionality
4. Update this documentation

## 📚 Related Documentation

- [Data Preprocessing Guide](docs/user-guide/data-preprocessing.md)
- [Model Evaluation Guide](docs/user-guide/model-evaluation.md)
- [Topic Modeling Guide](docs/user-guide/topic-modeling.md)
- [API Documentation](docs/api/)
