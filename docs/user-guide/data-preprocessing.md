# Data Preprocessing

The data preprocessing module provides tools for cleaning and preparing text data for topic modeling.

## TextPreprocessor Class

The `TextPreprocessor` class handles the cleaning and preprocessing of raw text data.

### Key Features

- **Text Cleaning**: Removes special characters, numbers, and extra whitespace
- **Tokenization**: Splits text into individual words
- **Stopword Removal**: Removes common words that don't carry meaning
- **Lemmatization**: Reduces words to their base form
- **Customizable**: Configurable preprocessing steps

### Usage Example

```python
from src.data.data_preprocessing import TextPreprocessor

# Initialize preprocessor
preprocessor = TextPreprocessor(
    remove_stopwords=True,
    lemmatize=True,
    min_word_length=3
)

# Process texts
raw_texts = ["Your raw text data here..."]
processed_texts = preprocessor.preprocess_texts(raw_texts)
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
