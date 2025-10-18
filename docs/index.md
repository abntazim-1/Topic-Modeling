# Topic Modeling Project

Welcome to the Topic Modeling Project! This project provides a comprehensive toolkit for automated topic modeling using Latent Dirichlet Allocation (LDA) and Non-negative Matrix Factorization (NMF) algorithms.

## Features

- **Multiple Topic Modeling Algorithms**: Support for LDA and NMF models
- **Automated Preprocessing**: Text cleaning, tokenization, and vectorization
- **Model Evaluation**: Comprehensive evaluation metrics including coherence and perplexity
- **Easy-to-use API**: Simple Python interface for all operations
- **Production Ready**: Robust error handling and logging
- **Extensible**: Modular design for easy customization

## Quick Start

```python
from src.topics.lda_model import LDAModeler
from src.data.data_preprocessing import TextPreprocessor

# Initialize preprocessor
preprocessor = TextPreprocessor()
texts = preprocessor.preprocess_texts(raw_texts)

# Initialize LDA model
lda = LDAModeler(num_topics=10)
lda.train(corpus, dictionary)

# Get topics
topics = lda.get_topics()
```

## Installation

See the [Installation Guide](getting-started/installation.md) for detailed setup instructions.

## Documentation

- [Getting Started](getting-started/installation.md) - Installation and setup
- [User Guide](user-guide/data-preprocessing.md) - How to use the project
- [API Reference](api/index.md) - Complete API documentation
- [Examples](examples.md) - Code examples and tutorials

## Contributing

We welcome contributions! Please see our [Contributing Guide](contributing.md) for details.
