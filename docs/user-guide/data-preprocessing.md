# Data Preprocessing

The data preprocessing module provides tools for cleaning and preparing text data for topic modeling.

## TextPreprocessor Class

The `TextPreprocessor` class handles the cleaning and preprocessing of raw text data.

### Key Features

- **Text Cleaning**: Removes special characters, numbers, and extra whitespace
- **Tokenization**: Splits text into individual words using spaCy
- **Stopword Removal**: Removes common words that don't carry meaning, including news-specific stopwords
- **Lemmatization**: Reduces words to their base form using spaCy
- **N-gram Generation**: Creates bigrams and trigrams for better topic modeling
- **Customizable**: Configurable preprocessing steps and stopwords

### Default Stopwords

The preprocessing pipeline uses a comprehensive set of stopwords that combines:

1. **NLTK English stopwords** - Standard English stopwords (the, a, an, and, or, but, etc.)
2. **News-specific stopwords** - Words commonly found in news articles that don't add semantic value

```python
# News-specific words added to NLTK English stopwords
news_stopwords = {
    'said', 'mr', 'mrs', 'one', 'two', 'year', 'new', 'us', 'like', 
    'time', 'people', 'say', 'month', 'day', 'bn'
}

# Final comprehensive stopwords = NLTK English stopwords + news-specific words
comprehensive_stopwords = nltk_stopwords.union(news_stopwords)
```

This approach ensures that both common English stopwords and news-specific noisy words are filtered out, resulting in cleaner, more meaningful topics.

### Usage Example

```python
from src.data.data_preprocessing import PreprocessingPipeline, PreprocessingConfig, get_comprehensive_stopwords

# Initialize preprocessing configuration with comprehensive stopwords
config = PreprocessingConfig(
    text_column="content",
    stopwords=get_comprehensive_stopwords(),  # Uses NLTK + news-specific stopwords
    spacy_model="en_core_web_lg",
    min_token_length=2,
    gen_bigrams=True,
    gen_trigrams=True
)

# Create preprocessing pipeline
pipeline = PreprocessingPipeline(config)

# Process texts
raw_texts = ["Your raw text data here..."]
result_df = pipeline.run(raw_texts)

# Access processed data
cleaned_texts = result_df['cleaned_text'].tolist()
tokenized_texts = result_df['tokens'].tolist()
```

## Data Ingestion

The `DataIngestion` class handles loading data from various sources.

### Supported Formats

- CSV files
- JSON files
- Text files
- Database connections

### Example

```python
from src.data.data_ingestion import DataIngestion

# Load data from CSV
ingestion = DataIngestion()
data = ingestion.load_csv("path/to/data.csv", text_column="content")
```
