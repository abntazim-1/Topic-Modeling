# Quick Start

This guide will help you get started with topic modeling in just a few minutes.

## Basic Usage

### 1. Prepare Your Data

```python
from src.data.data_preprocessing import TextPreprocessor

# Your raw texts
raw_texts = [
    "This is a sample document about machine learning.",
    "Another document discussing artificial intelligence.",
    "A third document about data science and analytics."
]

# Initialize preprocessor
preprocessor = TextPreprocessor()

# Preprocess texts
processed_texts = preprocessor.preprocess_texts(raw_texts)
```

### 2. Create Topic Model

```python
from src.topics.lda_model import LDAModeler
from src.features.tfidf import TFIDFVectorizer

# Vectorize texts
vectorizer = TFIDFVectorizer()
corpus, dictionary = vectorizer.fit_transform(processed_texts)

# Train LDA model
lda = LDAModeler(num_topics=5)
lda.train(corpus, dictionary)

# Get topics
topics = lda.get_topics(num_words=10)
for topic in topics:
    print(f"Topic {topic['topic_id']}: {topic['words']}")
```

### 3. Evaluate Model

```python
# Compute coherence score
coherence_score = lda.compute_coherence_score(processed_texts, dictionary)
print(f"Coherence Score: {coherence_score:.4f}")

# Compute perplexity
perplexity = lda.compute_perplexity(corpus)
print(f"Perplexity: {perplexity:.4f}")
```

## Next Steps

- Learn about [Data Preprocessing](../user-guide/data-preprocessing.md)
- Explore [Topic Modeling](../user-guide/topic-modeling.md) in detail
- Check out [Model Evaluation](../user-guide/model-evaluation.md) techniques
- Browse the [API Reference](../api/index.md) for complete documentation
