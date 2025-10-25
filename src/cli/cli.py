"""
Professional CLI Interface for Topic Modeling Project
===================================================

This module provides a comprehensive command-line interface for the topic modeling project.
It uses Click for clean command definitions and integrates with the project's logging system.

Commands:
- train: Train topic models (LDA/NMF) with configurable parameters
- evaluate: Evaluate trained models and generate metrics
- predict: Predict topics for input text using trained models
- show-models: List available models and their metadata

Usage:
    python -m src.cli.cli train --model lda --topics 10
    python -m src.cli.cli evaluate --model lda
    python -m src.cli.cli predict --model lda --text "Your text here"
    python -m src.cli.cli show-models
"""

import os
import sys
import yaml
import click
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import project modules
from src.utils.logger import get_logger, setup_logging
from src.utils.exceptions import AppException, DataValidationError

# Import modules with error handling
try:
    from src.pipelines.run_pipeline import TopicModelingPipeline
except ImportError as e:
    TopicModelingPipeline = None
    print(f"Warning: Could not import TopicModelingPipeline: {e}")

try:
    from src.models.train import TopicModelTrainer
except ImportError as e:
    TopicModelTrainer = None
    print(f"Warning: Could not import TopicModelTrainer: {e}")

try:
    from src.models.evaluate import TopicModelEvaluator, ModelComparison
except ImportError as e:
    TopicModelEvaluator = None
    ModelComparison = None
    print(f"Warning: Could not import evaluation modules: {e}")

try:
    from src.models.infer import TopicModelInference
except ImportError as e:
    TopicModelInference = None
    print(f"Warning: Could not import TopicModelInference: {e}")


class CLIConfig:
    """Configuration manager for CLI operations."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize CLI configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        self.model_registry = self._load_model_registry()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load main configuration file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            click.echo(f"Configuration file not found: {self.config_path}", err=True)
            sys.exit(1)
        except yaml.YAMLError as e:
            click.echo(f"Error parsing configuration file: {e}", err=True)
            sys.exit(1)
    
    def _load_model_registry(self) -> Dict[str, Any]:
        """Load model registry configuration."""
        registry_path = "config/model_registry.yaml"
        try:
            with open(registry_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            click.echo(f"Model registry not found: {registry_path}", err=True)
            sys.exit(1)
        except yaml.YAMLError as e:
            click.echo(f"Error parsing model registry: {e}", err=True)
            sys.exit(1)
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        if model_name not in self.model_registry.get('models', {}):
            available_models = list(self.model_registry.get('models', {}).keys())
            raise click.BadParameter(f"Model '{model_name}' not found. Available models: {available_models}")
        
        return self.model_registry['models'][model_name]
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.config.get('data', {})
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self.config.get('api', {})


def setup_cli_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging for CLI operations."""
    try:
        # Setup logging using the project's logging configuration
        setup_logging("config/logging.yaml")
        logger = get_logger("cli")
        
        # Set log level
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        logger.setLevel(numeric_level)
        
        return logger
    except Exception as e:
        # Fallback to basic logging
        logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
        return logging.getLogger("cli")


@click.group()
@click.option('--log-level', '-l', 
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR'], case_sensitive=False),
              default='INFO',
              help='Set the logging level')
@click.option('--config', '-c',
              type=click.Path(exists=True, path_type=Path),
              default='config/config.yaml',
              help='Path to configuration file')
@click.pass_context
def cli(ctx: click.Context, log_level: str, config: str):
    """
    Topic Modeling CLI
    
    A professional command-line interface for training, evaluating, and using topic models.
    
    This CLI provides access to LDA and NMF topic modeling capabilities with comprehensive
    evaluation metrics and prediction functionality.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Setup logging
    logger = setup_cli_logging(log_level)
    
    # Initialize configuration
    try:
        cli_config = CLIConfig(str(config))
        ctx.obj['config'] = cli_config
        ctx.obj['logger'] = logger
        
        logger.info("Topic Modeling CLI initialized")
        
    except Exception as e:
        click.echo(f"Failed to initialize CLI: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--model', '-m',
              type=click.Choice(['lda', 'nmf'], case_sensitive=False),
              required=True,
              help='Model type to train (LDA or NMF)')
@click.option('--topics', '-t',
              type=click.IntRange(min=3, max=50),
              default=None,
              help='Number of topics (3-50). Uses model default if not specified')
@click.option('--data-path', '-d',
              type=click.Path(exists=True, path_type=Path),
              default=None,
              help='Path to training data CSV file')
@click.option('--output-dir', '-o',
              type=click.Path(path_type=Path),
              default='artifacts',
              help='Output directory for trained models')
@click.option('--vectorizer', '-v',
              type=click.Choice(['tfidf', 'bow'], case_sensitive=False),
              default='tfidf',
              help='Vectorizer type to use')
@click.option('--force', '-f',
              is_flag=True,
              help='Force retraining even if model already exists')
@click.pass_context
def train(ctx: click.Context, model: str, topics: Optional[int], data_path: Optional[str], 
          output_dir: str, vectorizer: str, force: bool):
    """
    Train a topic model (LDA or NMF)
    
    This command trains a topic model using the specified parameters and saves
    the trained model along with evaluation metrics.
    
    Examples:
        cli train --model lda --topics 10
        cli train --model nmf --topics 15 --data-path data/news.csv
        cli train --model lda --force  # Retrain existing model
    """
    logger = ctx.obj['logger']
    config = ctx.obj['config']
    
    try:
        logger.info(f"Starting {model.upper()} model training")
        
        # Check if required modules are available
        if TopicModelingPipeline is None:
            raise click.BadParameter("TopicModelingPipeline module not available. Please install required dependencies.")
        
        # Get model configuration
        model_config = config.get_model_config(model)
        
        # Determine number of topics
        if topics is None:
            topics = model_config.get('default_topics', 10)
            logger.info(f"Using default topics: {topics}")
        
        # Validate topics range
        min_topics = model_config.get('min_topics', 3)
        max_topics = model_config.get('max_topics', 20)
        if not (min_topics <= topics <= max_topics):
            raise click.BadParameter(f"Topics must be between {min_topics} and {max_topics}")
        
        # Get data path
        if data_path is None:
            data_config = config.get_data_config()
            data_path = data_config.get('csv_path', 'artifacts/bbc-news-data.csv')
        
        # Check if model already exists
        model_path = Path(output_dir) / f"{model}_model.pkl"
        if model_path.exists() and not force:
            click.echo(f"Model already exists: {model_path}")
            click.echo("Use --force to retrain or choose a different output directory")
            return
        
        # Initialize pipeline
        pipeline = TopicModelingPipeline(
            config_path=str(ctx.params.get('config', 'config/config.yaml')),
            output_dir=output_dir,
            log_level=ctx.params.get('log_level', 'INFO')
        )
        
        # Train the model
        click.echo(f"Training {model.upper()} model with {topics} topics...")
        click.echo(f"Data: {data_path}")
        click.echo(f"Output: {output_dir}")
        
        start_time = datetime.now()
        
        # Run training pipeline
        results = pipeline.run_training_pipeline(
            model_type=model,
            num_topics=topics,
            vectorizer_type=vectorizer,
            data_path=str(data_path)
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Display results
        click.echo("\nTraining completed successfully!")
        click.echo(f"Duration: {duration:.2f} seconds")
        click.echo(f"Model saved: {results['model_path']}")
        click.echo(f"Topics saved: {results['topics_path']}")
        
        if 'evaluation_results' in results:
            eval_results = results['evaluation_results']
            click.echo("\nEvaluation Results:")
            for metric, value in eval_results.items():
                click.echo(f"  {metric}: {value:.4f}")
        
        logger.info(f"{model.upper()} training completed in {duration:.2f}s")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        click.echo(f"Training failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--model', '-m',
              type=click.Choice(['lda', 'nmf'], case_sensitive=False),
              required=True,
              help='Model to evaluate')
@click.option('--model-path', '-p',
              type=click.Path(exists=True, path_type=Path),
              default=None,
              help='Path to trained model file')
@click.option('--data-path', '-d',
              type=click.Path(exists=True, path_type=Path),
              default=None,
              help='Path to evaluation data CSV file')
@click.option('--output-dir', '-o',
              type=click.Path(path_type=Path),
              default='artifacts/evaluation',
              help='Output directory for evaluation results')
@click.option('--metrics', '-me',
              multiple=True,
              help='Specific metrics to compute (can be used multiple times)')
@click.option('--save-plots', '-s',
              is_flag=True,
              default=True,
              help='Save evaluation plots')
@click.pass_context
def evaluate(ctx: click.Context, model: str, model_path: Optional[str], data_path: Optional[str],
             output_dir: str, metrics: tuple, save_plots: bool):
    """
    Evaluate a trained topic model
    
    This command evaluates a trained model using various metrics and generates
    comprehensive evaluation reports and visualizations.
    
    Examples:
        cli evaluate --model lda
        cli evaluate --model nmf --metrics coherence_score --metrics topic_diversity
        cli evaluate --model lda --model-path artifacts/custom_model.pkl
    """
    logger = ctx.obj['logger']
    config = ctx.obj['config']
    
    try:
        logger.info(f"Starting {model.upper()} model evaluation")
        
        # Check if required modules are available
        if TopicModelEvaluator is None:
            raise click.BadParameter("TopicModelEvaluator module not available. Please install required dependencies.")
        
        # Get model configuration
        model_config = config.get_model_config(model)
        
        # Determine model path
        if model_path is None:
            model_path = model_config.get('model_path', f'artifacts/{model}_model.pkl')
        
        # Check if model exists
        if not Path(model_path).exists():
            raise click.BadParameter(f"Model file not found: {model_path}")
        
        # Get data path
        if data_path is None:
            data_config = config.get_data_config()
            data_path = data_config.get('csv_path', 'artifacts/bbc-news-data.csv')
        
        # Determine metrics to compute
        if metrics:
            metrics_to_compute = list(metrics)
        else:
            metrics_to_compute = model_config.get('evaluation_metrics', ['coherence_score', 'topic_diversity'])
        
        # Initialize evaluator
        evaluator = TopicModelEvaluator(
            model_path=str(model_path),
            model_type=model,
            logger=logger
        )
        
        click.echo(f"Evaluating {model.upper()} model...")
        click.echo(f"Model: {model_path}")
        click.echo(f"Data: {data_path}")
        click.echo(f"Output: {output_dir}")
        click.echo(f"Metrics: {', '.join(metrics_to_compute)}")
        
        # Run evaluation
        start_time = datetime.now()
        
        evaluation_results = evaluator.evaluate_model(
            data_path=str(data_path),
            metrics=metrics_to_compute,
            output_dir=str(output_dir),
            save_plots=save_plots
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Display results
        click.echo("\nEvaluation completed successfully!")
        click.echo(f"Duration: {duration:.2f} seconds")
        
        click.echo("\nEvaluation Results:")
        for metric, value in evaluation_results.items():
            if isinstance(value, (int, float)):
                click.echo(f"  {metric}: {value:.4f}")
            else:
                click.echo(f"  {metric}: {value}")
        
        if save_plots:
            click.echo(f"\nPlots saved to: {output_dir}")
        
        logger.info(f"{model.upper()} evaluation completed in {duration:.2f}s")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        click.echo(f"Evaluation failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--model', '-m',
              type=click.Choice(['lda', 'nmf'], case_sensitive=False),
              required=True,
              help='Model to use for prediction')
@click.option('--text', '-t',
              type=str,
              required=True,
              help='Text to analyze')
@click.option('--topics', '-k',
              type=click.IntRange(min=1, max=20),
              default=5,
              help='Number of top topics to return')
@click.option('--model-path', '-p',
              type=click.Path(exists=True, path_type=Path),
              default=None,
              help='Path to trained model file')
@click.option('--vectorizer-path', '-v',
              type=click.Path(exists=True, path_type=Path),
              default=None,
              help='Path to vectorizer file')
@click.option('--format', '-f',
              type=click.Choice(['table', 'json', 'simple'], case_sensitive=False),
              default='table',
              help='Output format')
@click.pass_context
def predict(ctx: click.Context, model: str, text: str, topics: int, model_path: Optional[str],
            vectorizer_path: Optional[str], format: str):
    """
    Predict topics for input text
    
    This command uses a trained model to predict the most relevant topics
    for the given input text.
    
    Examples:
        cli predict --model lda --text "Your news article text here"
        cli predict --model nmf --text "Technology news" --topics 3 --format json
        cli predict --model lda --text "Sports update" --model-path artifacts/custom_model.pkl
    """
    logger = ctx.obj['logger']
    config = ctx.obj['config']
    
    try:
        logger.info(f"Starting topic prediction with {model.upper()} model")
        
        # Check if required modules are available
        if TopicModelInference is None:
            raise click.BadParameter("TopicModelInference module not available. Please install required dependencies.")
        
        # Get model configuration
        model_config = config.get_model_config(model)
        
        # Determine model path
        if model_path is None:
            model_path = model_config.get('model_path', f'artifacts/{model}_model.pkl')
        
        # Determine vectorizer path
        if vectorizer_path is None:
            vectorizer_path = model_config.get('vectorizer_path', 'artifacts/tfidf_vectorizer.pkl')
        
        # Check if files exist
        if not Path(model_path).exists():
            raise click.BadParameter(f"Model file not found: {model_path}")
        if not Path(vectorizer_path).exists():
            raise click.BadParameter(f"Vectorizer file not found: {vectorizer_path}")
        
        # Initialize inference engine
        inference = TopicModelInference(
            model_path=str(model_path),
            vectorizer_path=str(vectorizer_path),
            model_type=model,
            logger=logger
        )
        
        click.echo(f"Predicting topics with {model.upper()} model...")
        click.echo(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        click.echo(f"Top {topics} topics requested")
        
        # Run prediction
        start_time = datetime.now()
        
        prediction_results = inference.predict_topics(
            text=text,
            num_topics=topics
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Display results based on format
        click.echo(f"\nPrediction completed in {duration:.2f} seconds")
        
        if format == 'json':
            import json
            click.echo(json.dumps(prediction_results, indent=2))
        
        elif format == 'simple':
            click.echo(f"\nDominant Topic: {prediction_results['dominant_topic']}")
            click.echo(f"Confidence: {prediction_results['confidence']:.4f}")
            click.echo("\nTop Topics:")
            for i, topic in enumerate(prediction_results['topics'], 1):
                click.echo(f"  {i}. Topic {topic['topic_id']} (prob: {topic['probability']:.4f})")
                click.echo(f"     Words: {', '.join(topic['topic_words'][:5])}")
        
        else:  # table format
            click.echo(f"\nDominant Topic: {prediction_results['dominant_topic']}")
            click.echo(f"Confidence: {prediction_results['confidence']:.4f}")
            
            click.echo("\nTopic Distribution:")
            click.echo("+---------+-------------+------------------------------------------+")
            click.echo("| Topic   | Probability | Top Words                                |")
            click.echo("+---------+-------------+------------------------------------------+")
            
            for topic in prediction_results['topics']:
                words_str = ', '.join(topic['topic_words'][:5])
                if len(words_str) > 40:
                    words_str = words_str[:37] + "..."
                click.echo(f"| {topic['topic_id']:7d} | {topic['probability']:11.4f} | {words_str:<40} |")
            
            click.echo("+---------+-------------+------------------------------------------+")
        
        logger.info(f"Topic prediction completed in {duration:.2f}s")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        click.echo(f"Prediction failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--model', '-m',
              type=click.Choice(['lda', 'nmf', 'all'], case_sensitive=False),
              default='all',
              help='Specific model to show or all models')
@click.option('--format', '-f',
              type=click.Choice(['table', 'json', 'simple'], case_sensitive=False),
              default='table',
              help='Output format')
@click.pass_context
def show_models(ctx: click.Context, model: str, format: str):
    """
    Show available models and their metadata
    
    This command displays information about available trained models,
    including their configuration, parameters, and status.
    
    Examples:
        cli show-models
        cli show-models --model lda
        cli show-models --format json
    """
    logger = ctx.obj['logger']
    config = ctx.obj['config']
    
    try:
        logger.info("Displaying available models")
        
        models_registry = config.model_registry.get('models', {})
        
        if model != 'all' and model not in models_registry:
            available_models = list(models_registry.keys())
            raise click.BadParameter(f"Model '{model}' not found. Available models: {available_models}")
        
        # Filter models to show
        if model == 'all':
            models_to_show = models_registry
        else:
            models_to_show = {model: models_registry[model]}
        
        if format == 'json':
            import json
            click.echo(json.dumps(models_to_show, indent=2))
            return
        
        # Display models in table format
        click.echo("Available Models")
        click.echo("=" * 50)
        
        for model_name, model_info in models_to_show.items():
            # Check if model files exist
            model_path = Path(model_info.get('model_path', f'artifacts/{model_name}_model.pkl'))
            vectorizer_path = Path(model_info.get('vectorizer_path', 'artifacts/tfidf_vectorizer.pkl'))
            topics_path = Path(model_info.get('topics_path', f'artifacts/{model_name}_topics.csv'))
            
            model_exists = model_path.exists()
            vectorizer_exists = vectorizer_path.exists()
            topics_exist = topics_path.exists()
            
            status = "Ready" if all([model_exists, vectorizer_exists]) else "Incomplete"
            
            if format == 'simple':
                click.echo(f"\n{model_name.upper()}: {model_info.get('name', 'Unknown')}")
                click.echo(f"  Status: {status}")
                click.echo(f"  Description: {model_info.get('description', 'No description')}")
                click.echo(f"  Default Topics: {model_info.get('default_topics', 'N/A')}")
                click.echo(f"  Model File: {'EXISTS' if model_exists else 'MISSING'} {model_path}")
                click.echo(f"  Vectorizer: {'EXISTS' if vectorizer_exists else 'MISSING'} {vectorizer_path}")
                click.echo(f"  Topics: {'EXISTS' if topics_exist else 'MISSING'} {topics_path}")
            else:
                click.echo(f"\n{model_name.upper()}: {model_info.get('name', 'Unknown')}")
                click.echo(f"   Status: {status}")
                click.echo(f"   Description: {model_info.get('description', 'No description')}")
                click.echo(f"   Topics Range: {model_info.get('min_topics', 3)}-{model_info.get('max_topics', 20)}")
                click.echo(f"   Default Topics: {model_info.get('default_topics', 'N/A')}")
                
                click.echo(f"\n   Files:")
                click.echo(f"     Model: {'EXISTS' if model_exists else 'MISSING'} {model_path}")
                click.echo(f"     Vectorizer: {'EXISTS' if vectorizer_exists else 'MISSING'} {vectorizer_path}")
                click.echo(f"     Topics: {'EXISTS' if topics_exist else 'MISSING'} {topics_path}")
                
                # Show parameters
                parameters = model_info.get('parameters', {})
                if parameters:
                    click.echo(f"\n   Parameters:")
                    for param, value in parameters.items():
                        click.echo(f"     {param}: {value}")
                
                # Show evaluation metrics
                metrics = model_info.get('evaluation_metrics', [])
                if metrics:
                    click.echo(f"\n   Evaluation Metrics:")
                    click.echo(f"     {', '.join(metrics)}")
        
        click.echo(f"\nTotal Models: {len(models_to_show)}")
        
        logger.info(f"Displayed {len(models_to_show)} model(s)")
        
    except Exception as e:
        logger.error(f"Failed to show models: {e}")
        click.echo(f"Failed to show models: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()