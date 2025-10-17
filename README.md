# Topic Modeling Project

End-to-end topic modeling toolkit for analyzing text corpora using LDA and NMF. Includes data preprocessing, model training, evaluation, visualization, logging, and an optional web API.

## Features

- LDA and NMF topic models with configurable hyperparameters
- Data ingestion and preprocessing (tokenization, stopword removal, TF–IDF)
- Evaluation from CSV or trained models (diversity, coherence, heatmaps)
- Visualizations: topic-word heatmaps and comparison plots
- Clean logging with persistent run logs and minimal warnings
- CLI utilities and a simple FastAPI server (optional)
- Docker support and reproducible runs via `Makefile`

## Project Structure

```
g:\Topic Modeling Project/
├── artifacts/                 # Datasets, models, topics, evaluation outputs
├── config/                    # App, logging, and model registry configs
├── docs/                      # Architecture and API docs
├── src/                       # Source code (CLI, models, API, pipelines, utils)
├── notebooks/                 # Exploration and experiments
├── tests/                     # Unit/integration/E2E tests
├── docker/                    # Dockerfile and compose
├── logs/                      # Persistent project logs
├── evaluate_models.py         # Convenience wrapper to evaluate LDA/NMF
├── Makefile                   # Common tasks (build, test, format)
└── README.md
```

## Requirements

- Python 3.10+ (tested on 3.11)
- OS: Windows, macOS, or Linux
- Recommended: virtual environment

## Installation

```
# Clone and enter the project
git clone <your-repo-url>
cd "Topic Modeling Project"

# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate           # Windows
# source .venv/bin/activate          # macOS/Linux

# Install dependencies
pip install -r requirements
```

## Quick Start

### Evaluate existing topics (CSV)

Evaluate LDA topics from `artifacts/lda_topics.csv` and save metrics/plots to `artifacts/evaluation/`:

```
python -m src.models.evaluate --model-type lda --topics-csv artifacts/lda_topics.csv --output-dir artifacts/evaluation
```

Evaluate NMF topics:

```
python -m src.models.evaluate --model-type nmf --topics-csv artifacts/nmf_topics.csv --output-dir artifacts/evaluation
```

Or use the convenience wrapper to evaluate both models:

```
python evaluate_models.py
```

Outputs include:

- `artifacts/evaluation/<model>_evaluation.json`
- `artifacts/evaluation/<model>_topic_heatmap.png`
- `artifacts/evaluation/model_comparison.png` (when comparing)
- `artifacts/evaluation/<model>_evaluation.log` (per-run logs)

### CLI Help

```
python -m src.models.evaluate --help
python -m src.cli.cli --help
```

## Training and Inference

- Training script: `python -m src.models.train --help`
- Inference script: `python -m src.models.infer --help`

These scripts integrate with `src/topics/` implementations (`lda_model.py`, `nmf_model.py`) and `src/features/` for TF–IDF/embeddings.

## Logging

- Project-level logs: `logs/project.log`
- Wrapper logs: `artifacts/evaluation/evaluate_models.log`
- Per-evaluation logs: `artifacts/evaluation/<model>_evaluation.log`

Customize log output path via environment:

```
set LOG_FILE_PATH=g:\Topic Modeling Project\logs\custom.log   # Windows
# export LOG_FILE_PATH="$(pwd)/logs/custom.log"                # macOS/Linux
```

Logging is configured by `src/utils/logger.py` and is set up automatically by CLI and wrappers. Noisy import-time messages (e.g., TensorFlow info) are minimized.

## Configuration

- Main config: `config/config.yaml`
- Logging config (optional): `config/logging.yaml`
- Model registry: `config/model_registry.yaml`

Override settings via environment variables or by editing config files. Most CLI commands accept `--output-dir` and other flags for paths.

## API (Optional)

Launch the FastAPI server locally:

```
pip install uvicorn fastapi
uvicorn src.api.app:app --reload
```

See `docs/api.md` for routes and usage.

## Docker

Build and run using Docker:

```
# Build the image
docker build -t topic-modeling -f docker/Dockerfile .

# Run with compose
docker compose -f docker/docker-compose.yml up --build
```

## Makefile (Common Tasks)

```
# Lint, test, and format
make lint
make test
make format

# Build docker image
make build-image
```

## Tests

Run the unit/integration test suites:

```
pip install pytest
pytest -q
```

## Notebooks

Exploratory notebooks are available under `notebooks/`. For reproducibility, prefer running CLI commands or pipelines instead of notebooks for production tasks.

## Notes

- Evaluation: CSV-based evaluation assumes files like `artifacts/lda_topics.csv` and `artifacts/nmf_topics.csv` with topic-word distributions.
- Warnings: The codebase filters out common noisy warnings to keep logs clean.
- Paths: Use forward slashes or escape backslashes on Windows.

## Contributing

Pull requests are welcome. Please add tests for new features and keep changes minimal and focused.

## License

Distributed under the terms of the MIT License. See `LICENSE` for details.