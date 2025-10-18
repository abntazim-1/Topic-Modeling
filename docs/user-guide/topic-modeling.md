# Topic Modeling

This guide covers the topic modeling capabilities of the project, including LDA and NMF implementations.

## LDA (Latent Dirichlet Allocation)

The `LDAModeler` class provides a comprehensive interface for LDA topic modeling.

### Key Features

- **Flexible Configuration**: Customizable hyperparameters
- **Model Evaluation**: Coherence and perplexity metrics
- **Topic Extraction**: Get top words for each topic
- **Document Classification**: Assign topics to documents
- **Model Persistence**: Save and load trained models

### Basic Usage

```python
from src.topics.lda_model import LDAModeler

# Initialize model
lda = LDAModeler(
    num_topics=10,
    passes=10,
    chunksize=100,
    alpha='auto',
    random_state=42
)

# Train model
lda.train(corpus, dictionary)

# Get topics
topics = lda.get_topics(num_words=10)
```

### Model Evaluation

```python
# Compute coherence score
coherence = lda.compute_coherence_score(texts, dictionary)

# Compute perplexity
perplexity = lda.compute_perplexity(corpus)

# Get dominant topics per document
dominant_topics = lda.get_dominant_topic_per_doc(corpus)
```

## NMF (Non-negative Matrix Factorization)

The `NMFModeler` class provides NMF-based topic modeling.

### Usage

```python
from src.topics.nmf_model import NMFModeler

# Initialize model
nmf = NMFModeler(
    num_topics=10,
    max_iter=1000,
    random_state=42
)

# Train model
nmf.train(corpus, dictionary)

# Get topics
topics = nmf.get_topics()
```

## Choosing Between LDA and NMF

- **LDA**: Better for longer documents, probabilistic approach
- **NMF**: Better for shorter documents, deterministic approach
- **Evaluation**: Use coherence scores to compare models
