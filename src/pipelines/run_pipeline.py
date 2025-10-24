"""
Main Pipeline Script for Topic Modeling Project
==============================================

This script orchestrates the complete topic modeling pipeline:
1. Data Loading and Preprocessing
2. Model Training (LDA and NMF)
3. Model Evaluation
4. Artifact Saving

The pipeline follows MLOps best practices with structured logging,
error handling, and modular design.
"""

import os
import sys
import yaml
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import project modules
from src.utils.logger import get_logger, setup_logging
from src.utils.exceptions import AppException, DataValidationError
from src.data.data_ingestion import DataIngestion, DataIngestionConfig
from src.data.data_preprocessing import PreprocessingPipeline, PreprocessingConfig
from src.models.train import TopicModelTrainer
from src.models.evaluate import TopicModelEvaluator, ModelComparison


class TopicModelingPipeline:
    """
    Main pipeline orchestrator for topic modeling workflow.
    
    This class coordinates the entire pipeline from data ingestion
    through model training, evaluation, and artifact saving.
    """
    
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        output_dir: str = "artifacts/topic_models",
        log_level: str = "INFO"
    ):
        """
        Initialize the topic modeling pipeline.
        
        Args:
            config_path: Path to YAML configuration file
            output_dir: Directory to save all artifacts
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.config_path = config_path
        self.output_dir = output_dir
        self.log_level = log_level
        
        # Setup logging
        setup_logging(log_level=log_level)
        self.logger = get_logger(__name__)
        
        # Pipeline state
        self.config = None
        self.raw_data = None
        self.preprocessed_data = None
        self.lda_model = None
        self.nmf_model = None
        self.lda_evaluation = None
        self.nmf_evaluation = None
        
        # Create output directories
        self._create_output_directories()
        
        self.logger.info(f"Topic Modeling Pipeline initialized")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Configuration: {self.config_path}")

    def _create_output_directories(self) -> None:
        """Create necessary output directories."""
        try:
            directories = [
                self.output_dir,
                os.path.join(self.output_dir, "models"),
                os.path.join(self.output_dir, "evaluation"),
                os.path.join(self.output_dir, "preprocessed_data")
            ]
            
            for directory in directories:
                os.makedirs(directory, exist_ok=True)
                self.logger.debug(f"Created directory: {directory}")
                
        except Exception as e:
            self.logger.error(f"Failed to create output directories: {e}")
            raise AppException(f"Failed to create output directories: {e}")

    def load_configuration(self) -> None:
        """Load configuration from YAML file."""
        try:
            self.logger.info("Loading configuration...")
            
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            self.logger.info(f"Configuration loaded from {self.config_path}")
            self.logger.debug(f"Configuration keys: {list(self.config.keys())}")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise AppException(f"Configuration loading failed: {e}")

    def load_and_preprocess_data(self) -> None:
        """
        Stage 1: Load and preprocess the dataset.
        
        This stage handles:
        - Data ingestion with validation
        - Text preprocessing (cleaning, tokenization, n-grams)
        - Data quality checks
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("STAGE 1: DATA LOADING AND PREPROCESSING")
            self.logger.info("=" * 60)
            
            # Step 1: Data Ingestion
            self.logger.info("Loading raw data...")
            ingestion_config = DataIngestionConfig.from_yaml(self.config_path)
            data_ingestion = DataIngestion(ingestion_config, logger=self.logger)
            
            self.raw_data = data_ingestion.ingest()
            self.logger.info(f"Raw data loaded: {self.raw_data.shape[0]} documents")
            
            # Step 2: Text Preprocessing
            self.logger.info("Preprocessing text data...")
            preprocessing_config = PreprocessingConfig.from_dict(
                self.config.get('preprocessing', {})
            )
            preprocessing_config.output_dir = os.path.join(self.output_dir, "preprocessed_data")
            
            preprocessing_pipeline = PreprocessingPipeline(
                preprocessing_config, 
                logger=self.logger
            )
            
            self.preprocessed_data = preprocessing_pipeline.run(self.raw_data)
            self.logger.info(f"Text preprocessing completed: {self.preprocessed_data.shape[0]} documents")
            
            # Step 3: Save preprocessed data
            preprocessed_path = preprocessing_pipeline.save(self.preprocessed_data)
            self.logger.info(f"Preprocessed data saved to: {preprocessed_path}")
            
            # Step 4: Data quality summary
            self._log_data_quality_summary()
            
            self.logger.info("Data loading and preprocessing completed successfully")
            
        except Exception as e:
            self.logger.error(f"Data loading and preprocessing failed: {e}")
            raise AppException(f"Data preprocessing stage failed: {e}")

    def train_models(self) -> None:
        """
        Stage 2: Train both LDA and NMF models.
        
        This stage handles:
        - LDA model training with optimized hyperparameters
        - NMF model training with appropriate settings
        - Model validation and quality checks
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("STAGE 2: MODEL TRAINING")
            self.logger.info("=" * 60)
            
            if self.preprocessed_data is None:
                raise DataValidationError("No preprocessed data available for training")
            
            # Prepare data path for training
            preprocessed_path = os.path.join(
                self.output_dir, 
                "preprocessed_data", 
                "preprocessed_bbc_news.csv"
            )
            
            # Train LDA Model
            self.logger.info("Training LDA model...")
            lda_trainer = TopicModelTrainer(
                model_type="lda",
                num_topics=self.config.get('training', {}).get('num_topics', 10),
                data_path=preprocessed_path,
                vectorizer_type="tfidf",
                output_dir=os.path.join(self.output_dir, "models"),
                random_state=42
            )
            
            lda_trainer.load_data()
            lda_trainer.prepare_features()
            lda_trainer.train_model()
            lda_trainer.save_artifacts()
            lda_trainer.log_training_summary()
            
            self.lda_model = lda_trainer
            self.logger.info("LDA model training completed successfully")
            
            # Train NMF Model
            self.logger.info("Training NMF model...")
            nmf_trainer = TopicModelTrainer(
                model_type="nmf",
                num_topics=self.config.get('training', {}).get('num_topics', 10),
                data_path=preprocessed_path,
                vectorizer_type="tfidf",
                output_dir=os.path.join(self.output_dir, "models"),
                random_state=42
            )
            
            nmf_trainer.load_data()
            nmf_trainer.prepare_features()
            nmf_trainer.train_model()
            nmf_trainer.save_artifacts()
            nmf_trainer.log_training_summary()
            
            self.nmf_model = nmf_trainer
            self.logger.info("NMF model training completed successfully")
            
            self.logger.info("Model training stage completed successfully")
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            raise AppException(f"Model training stage failed: {e}")

    def evaluate_models(self) -> None:
        """
        Stage 3: Evaluate both trained models.
        
        This stage handles:
        - Comprehensive model evaluation (coherence, diversity, etc.)
        - Model comparison and ranking
        - Evaluation report generation
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("STAGE 3: MODEL EVALUATION")
            self.logger.info("=" * 60)
            
            if self.lda_model is None or self.nmf_model is None:
                raise DataValidationError("Models not trained yet - cannot evaluate")
            
            preprocessed_path = os.path.join(
                self.output_dir, 
                "preprocessed_data", 
                "preprocessed_bbc_news.csv"
            )
            
            evaluation_dir = os.path.join(self.output_dir, "evaluation")
            
            # Evaluate LDA Model
            self.logger.info("Evaluating LDA model...")
            lda_evaluator = TopicModelEvaluator(
                model_path=os.path.join(self.output_dir, "models", "lda_model.pkl"),
                vectorizer_path=os.path.join(self.output_dir, "models", "tfidf_vectorizer.pkl"),
                data_path=preprocessed_path,
                model_type="lda",
                output_dir=evaluation_dir
            )
            
            self.lda_evaluation = lda_evaluator.evaluate()
            self.logger.info(f"LDA evaluation completed: {self.lda_evaluation}")
            
            # Evaluate NMF Model
            self.logger.info("Evaluating NMF model...")
            nmf_evaluator = TopicModelEvaluator(
                model_path=os.path.join(self.output_dir, "models", "nmf_model.pkl"),
                vectorizer_path=os.path.join(self.output_dir, "models", "tfidf_vectorizer.pkl"),
                data_path=preprocessed_path,
                model_type="nmf",
                output_dir=evaluation_dir
            )
            
            self.nmf_evaluation = nmf_evaluator.evaluate()
            self.logger.info(f"NMF evaluation completed: {self.nmf_evaluation}")
            
            # Generate comprehensive comparison
            self.logger.info("Generating model comparison...")
            model_comparison = ModelComparison(evaluation_dir)
            
            model_configs = [
                {
                    'model_type': 'lda',
                    'model_path': os.path.join(self.output_dir, "models", "lda_model.pkl"),
                    'vectorizer_path': os.path.join(self.output_dir, "models", "tfidf_vectorizer.pkl"),
                    'name': 'LDA Model'
                },
                {
                    'model_type': 'nmf',
                    'model_path': os.path.join(self.output_dir, "models", "nmf_model.pkl"),
                    'vectorizer_path': os.path.join(self.output_dir, "models", "tfidf_vectorizer.pkl"),
                    'name': 'NMF Model'
                }
            ]
            
            comparison_results = model_comparison.compare_models(model_configs, preprocessed_path)
            self.logger.info("Model comparison completed")
            
            self.logger.info("Model evaluation stage completed successfully")
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            raise AppException(f"Model evaluation stage failed: {e}")

    def save_artifacts(self) -> None:
        """
        Stage 4: Save all pipeline artifacts.
        
        This stage handles:
        - Model artifacts (trained models, vectorizers)
        - Evaluation results and reports
        - Pipeline metadata and logs
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("STAGE 4: ARTIFACT SAVING")
            self.logger.info("=" * 60)
            
            # Save pipeline metadata
            pipeline_metadata = {
                "pipeline_version": "1.0.0",
                "execution_time": datetime.now().isoformat(),
                "config_path": self.config_path,
                "output_directory": self.output_dir,
                "data_shape": self.raw_data.shape if self.raw_data is not None else None,
                "preprocessed_shape": self.preprocessed_data.shape if self.preprocessed_data is not None else None,
                "models_trained": {
                    "lda": self.lda_model is not None,
                    "nmf": self.nmf_model is not None
                },
                "evaluations_completed": {
                    "lda": self.lda_evaluation is not None,
                    "nmf": self.nmf_evaluation is not None
                }
            }
            
            metadata_path = os.path.join(self.output_dir, "pipeline_metadata.json")
            import json
            with open(metadata_path, 'w') as f:
                json.dump(pipeline_metadata, f, indent=4)
            
            self.logger.info(f"Pipeline metadata saved to: {metadata_path}")
            
            # Log artifact summary
            self._log_artifact_summary()
            
            self.logger.info("Artifact saving stage completed successfully")
            
        except Exception as e:
            self.logger.error(f"Artifact saving failed: {e}")
            raise AppException(f"Artifact saving stage failed: {e}")

    def _log_data_quality_summary(self) -> None:
        """Log summary of data quality metrics."""
        try:
            if self.raw_data is not None and self.preprocessed_data is not None:
                self.logger.info("Data Quality Summary:")
                self.logger.info(f"  Raw data: {self.raw_data.shape[0]} documents, {self.raw_data.shape[1]} columns")
                self.logger.info(f"  Preprocessed data: {self.preprocessed_data.shape[0]} documents, {self.preprocessed_data.shape[1]} columns")
                
                # Check for missing values
                raw_missing = self.raw_data.isnull().sum().sum()
                preprocessed_missing = self.preprocessed_data.isnull().sum().sum()
                
                self.logger.info(f"  Raw data missing values: {raw_missing}")
                self.logger.info(f"  Preprocessed data missing values: {preprocessed_missing}")
                
        except Exception as e:
            self.logger.warning(f"Could not generate data quality summary: {e}")

    def _log_artifact_summary(self) -> None:
        """Log summary of saved artifacts."""
        try:
            self.logger.info("Artifact Summary:")
            
            # List all files in output directory
            for root, dirs, files in os.walk(self.output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    self.logger.info(f"  {file_path} ({file_size} bytes)")
            
        except Exception as e:
            self.logger.warning(f"Could not generate artifact summary: {e}")

    def run_pipeline(self) -> None:
        """
        Execute the complete topic modeling pipeline.
        
        This method orchestrates all pipeline stages in sequence:
        1. Load Configuration
        2. Load and Preprocess Data
        3. Train Models
        4. Evaluate Models
        5. Save Artifacts
        """
        try:
            start_time = datetime.now()
            self.logger.info("=" * 80)
            self.logger.info("STARTING TOPIC MODELING PIPELINE")
            self.logger.info("=" * 80)
            self.logger.info(f"Pipeline started at: {start_time}")
            self.logger.info(f"Output directory: {self.output_dir}")
            
            # Stage 1: Load Configuration
            self.load_configuration()
            
            # Stage 2: Load and Preprocess Data
            self.load_and_preprocess_data()
            
            # Stage 3: Train Models
            self.train_models()
            
            # Stage 4: Evaluate Models
            self.evaluate_models()
            
            # Stage 5: Save Artifacts
            self.save_artifacts()
            
            # Pipeline completion
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.info("=" * 80)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            self.logger.info(f"Total execution time: {duration}")
            self.logger.info(f"Pipeline completed at: {end_time}")
            self.logger.info(f"All artifacts saved to: {self.output_dir}")
            
        except Exception as e:
            self.logger.error("=" * 80)
            self.logger.error("PIPELINE FAILED")
            self.logger.error("=" * 80)
            self.logger.error(f"Pipeline failed with error: {e}")
            raise AppException(f"Pipeline execution failed: {e}")


def main():
    """
    Main entry point for the topic modeling pipeline.
    
    This function handles command-line arguments and orchestrates
    the complete pipeline execution.
    """
    parser = argparse.ArgumentParser(
        description="Topic Modeling Pipeline - Complete MLOps Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py
  python run_pipeline.py --config config/config.yaml --output artifacts/topic_models
  python run_pipeline.py --log-level DEBUG
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration YAML file (default: config/config.yaml)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/topic_models",
        help="Output directory for all artifacts (default: artifacts/topic_models)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize and run pipeline
        pipeline = TopicModelingPipeline(
            config_path=args.config,
            output_dir=args.output,
            log_level=args.log_level
        )
        
        pipeline.run_pipeline()
        
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Check the logs for detailed information")
        print(f"All artifacts saved to: {args.output}")
        print("=" * 80)
        
    except AppException as e:
        print(f"\nPipeline failed with application error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nPipeline failed with unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
