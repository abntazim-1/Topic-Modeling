# LDA Topic Number Optimization

## Overview

I've created a focused topic optimization system that **only optimizes the number of topics (5-15)** while keeping all other LDA parameters exactly as they were in your original setup.

## What It Does

✅ **Tests topics 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15**  
✅ **Keeps all other LDA parameters fixed** (alpha='auto', eta='auto', passes=20, etc.)  
✅ **Finds the topic count with best coherence score**  
✅ **Saves the best model and results**  

## How to Use

### Option 1: Enhanced Training Script (Recommended)
```bash
python src/models/train.py --model_type lda --optimize_topics_only
```

### Option 2: Direct Optimization Script
```bash
python scripts/optimize_topics.py
```

### Option 3: Programmatic Usage
```python
from src.models.train import TopicModelTrainer

trainer = TopicModelTrainer(
    model_type="lda",
    data_path="artifacts/preprocessed_bbc_news.csv",
    output_dir="artifacts/",
    optimize_topics_only=True  # This is the key parameter!
)

trainer.load_data()
trainer.prepare_features()
trainer.train_model()  # This will optimize topics 5-15
trainer.save_artifacts()
```

## Fixed LDA Parameters

The system keeps these parameters exactly as they were:
- **Alpha**: 'auto'
- **Eta**: 'auto'  
- **Passes**: 20
- **Iterations**: 800
- **Chunksize**: 2000
- **Update Every**: 1
- **Decay**: 0.5
- **Offset**: 10.0

## Output Files

After optimization, you'll get:
- `artifacts/topic_optimization/topic_optimization_results.json` - Detailed results
- `artifacts/topic_optimization/topic_optimization_plots.png` - Visualization plots
- `artifacts/topic_optimization/optimization_summary.txt` - Human-readable summary
- `artifacts/best_lda_model.pkl` - Best trained model
- `artifacts/best_model_topics.csv` - Topics from best model

## Expected Results

The system will:
1. Test each topic count (5-15)
2. Calculate coherence score for each
3. Select the topic count with highest coherence
4. Train the final model with optimal topic count
5. Save all results and visualizations

## Example Output

```
LDA Topic Number Optimization Results
====================================

Fixed Parameters:
- Alpha: auto
- Eta: auto
- Passes: 20
- Iterations: 800
- Chunksize: 2000
- Update Every: 1
- Decay: 0.5
- Offset: 10.0

Optimization Results:
- Best Number of Topics: 8
- Best Coherence Score: -6.4523
- Total Models Tested: 11

Detailed Results:
- 5 topics: Coherence=-7.1234, Perplexity=-8.1234, Diversity=0.78
- 6 topics: Coherence=-6.9876, Perplexity=-8.2345, Diversity=0.79
- 7 topics: Coherence=-6.6543, Perplexity=-8.3456, Diversity=0.80
- 8 topics: Coherence=-6.4523, Perplexity=-8.4567, Diversity=0.81
- 9 topics: Coherence=-6.5432, Perplexity=-8.5678, Diversity=0.82
- ...
```

## Quick Start

To run the optimization right now:

```bash
# Navigate to your project directory
cd "G:\Topic Modeling Project"

# Run topic optimization
python src/models/train.py --model_type lda --optimize_topics_only
```

This will automatically:
- Load your BBC news data
- Test topics 5-15
- Find the best topic count
- Save the optimized model
- Generate visualizations and reports

The process typically takes 5-10 minutes depending on your data size.
