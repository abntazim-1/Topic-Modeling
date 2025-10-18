# Model Evaluation

This guide covers the evaluation metrics and techniques available for assessing topic model quality.

## Evaluation Metrics

### Coherence Score

The coherence score measures the semantic similarity between words in a topic.

```python
# Compute coherence score
coherence_score = lda.compute_coherence_score(
    texts=processed_texts,
    id2word=dictionary,
    coherence='c_v'  # Options: 'c_v', 'c_uci', 'c_npmi'
)
```

**Interpretation:**
- Higher scores indicate better topic quality
- Values typically range from -1 to 1
- c_v is generally preferred for most use cases

### Perplexity

Perplexity measures how well the model predicts unseen data.

```python
# Compute perplexity
perplexity = lda.compute_perplexity(corpus)
```

**Interpretation:**
- Lower perplexity indicates better model fit
- Should be evaluated on held-out test data
- Can be used to compare different models

## Model Comparison

### Comparing LDA and NMF

```python
from src.topics.lda_model import LDAModeler
from src.topics.nmf_model import NMFModeler

# Train both models
lda = LDAModeler(num_topics=10)
nmf = NMFModeler(num_topics=10)

lda.train(corpus, dictionary)
nmf.train(corpus, dictionary)

# Compare metrics
lda_coherence = lda.compute_coherence_score(texts, dictionary)
nmf_coherence = nmf.compute_coherence_score(texts, dictionary)

print(f"LDA Coherence: {lda_coherence:.4f}")
print(f"NMF Coherence: {nmf_coherence:.4f}")
```

### Hyperparameter Tuning

```python
# Test different numbers of topics
topic_counts = [5, 10, 15, 20]
coherence_scores = []

for num_topics in topic_counts:
    lda = LDAModeler(num_topics=num_topics)
    lda.train(corpus, dictionary)
    coherence = lda.compute_coherence_score(texts, dictionary)
    coherence_scores.append(coherence)
    print(f"Topics: {num_topics}, Coherence: {coherence:.4f}")
```

## Best Practices

1. **Use Multiple Metrics**: Combine coherence and perplexity
2. **Cross-Validation**: Evaluate on held-out data
3. **Parameter Tuning**: Test different hyperparameters
4. **Domain Knowledge**: Validate topics manually
5. **Reproducibility**: Set random seeds for consistent results
