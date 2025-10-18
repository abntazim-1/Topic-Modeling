# Installation

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Install Dependencies

Install the required packages for MkDocs documentation:

```bash
pip install mkdocs
pip install mkdocs-material
pip install mkdocstrings[python]
pip install mkdocs-gen-files
pip install mkdocs-literate-nav
pip install mkdocs-git-revision-date-localized-plugin
```

## Install Project Dependencies

Install the project's core dependencies:

```bash
pip install gensim
pip install scikit-learn
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install wordcloud
pip install pyLDAvis
pip install bertopic
pip install transformers
pip install torch
```

## Development Setup

For development, install additional tools:

```bash
pip install pytest
pip install black
pip install flake8
pip install mypy
pip install pre-commit
```

## Verify Installation

Test that everything is working:

```bash
# Test MkDocs
mkdocs --version

# Test project imports
python -c "from src.topics.lda_model import LDAModeler; print('Installation successful!')"
```
