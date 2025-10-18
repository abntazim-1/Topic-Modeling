# Examples

This page contains practical examples of using the Topic Modeling Project.

## Basic Topic Modeling

```python
from src.topics.lda_model import LDAModeler
from src.data.data_preprocessing import TextPreprocessor
from src.features.tfidf import TFIDFVectorizer

# Sample data
texts = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing helps computers understand text.",
    "Computer vision enables machines to interpret visual information.",
    "Data science combines statistics, programming, and domain expertise."
]

# Preprocess texts
preprocessor = TextPreprocessor()
processed_texts = preprocessor.preprocess_texts(texts)

# Vectorize
vectorizer = TFIDFVectorizer()
corpus, dictionary = vectorizer.fit_transform(processed_texts)

# Train model
lda = LDAModeler(num_topics=3)
lda.train(corpus, dictionary)

# Get topics
topics = lda.get_topics()
for topic in topics:
    print(f"Topic {topic['topic_id']}: {topic['words']}")
```

## Model Comparison

```python
from src.topics.lda_model import LDAModeler
from src.topics.nmf_model import NMFModeler

# Train both models
lda = LDAModeler(num_topics=5)
nmf = NMFModeler(num_topics=5)

lda.train(corpus, dictionary)
nmf.train(corpus, dictionary)

# Compare coherence scores
lda_coherence = lda.compute_coherence_score(processed_texts, dictionary)
nmf_coherence = nmf.compute_coherence_score(processed_texts, dictionary)

print(f"LDA Coherence: {lda_coherence:.4f}")
print(f"NMF Coherence: {nmf_coherence:.4f}")
```

## Document Classification

```python
# Get dominant topic for each document
dominant_topics = lda.get_dominant_topic_per_doc(corpus)

for i, topic_id in enumerate(dominant_topics):
    print(f"Document {i}: Topic {topic_id}")
```
