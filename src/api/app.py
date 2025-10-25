"""
Topic Modeling API Application
============================

A Flask-based REST API for topic modeling operations including:
- Model inference
- Topic visualization
- Model management
- Health monitoring

This API provides endpoints for interacting with trained LDA and NMF models.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.exceptions import BadRequest, NotFound, InternalServerError
import pandas as pd
import numpy as np
import joblib
from src.utils.logger import get_logger, setup_logging
from src.utils.exceptions import AppException, DataValidationError
from src.topics.lda_model import LDA
from src.topics.nmf_model import NMF


class TopicModelingAPI:
    """
    Flask API application for topic modeling operations.
    
    This class provides REST endpoints for:
    - Model inference and prediction
    - Topic visualization and analysis
    - Model management and health checks
    - Batch processing capabilities
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the Topic Modeling API.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.app = Flask(__name__)
        self.logger = get_logger(__name__)
        
        # Enable CORS for frontend integration
        CORS(self.app)
        
        # API configuration
        self.app.config['JSON_SORT_KEYS'] = False
        self.app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
        
        # Static files configuration
        self.app.static_folder = os.path.join(os.path.dirname(__file__), 'static')
        self.app.template_folder = os.path.join(os.path.dirname(__file__), 'templates')
        
        # Model storage
        self.models = {
            'lda': None,
            'nmf': None
        }
        self.vectorizers = {}
        self.model_metadata = {}
        
        # Load configuration and setup routes
        self._load_configuration()
        self._setup_routes()
        self._setup_error_handlers()
        
        self.logger.info("Topic Modeling API initialized successfully")

    def _load_configuration(self) -> None:
        """Load API configuration from YAML file."""
        try:
            import yaml
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                self.config = self._get_default_config()
                self.logger.warning(f"Config file not found, using defaults: {self.config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            self.config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default API configuration."""
        return {
            'api': {
                'host': '0.0.0.0',
                'port': 5000,
                'debug': False,
                'models_path': 'artifacts/topic_models/models',
                'max_topics': 20,
                'default_topics': 10
            },
            'logging': {
                'level': 'INFO'
            }
        }

    def _setup_routes(self) -> None:
        """Setup API routes and endpoints."""
        
        @self.app.route('/', methods=['GET'])
        def index():
            """Serve the main web interface."""
            try:
                template_path = os.path.join(self.app.template_folder, 'index.html')
                if os.path.exists(template_path):
                    with open(template_path, 'r', encoding='utf-8') as f:
                        return f.read()
                else:
                    return jsonify({'error': 'Web interface not found'}), 404
            except Exception as e:
                self.logger.error(f"Failed to serve index page: {e}")
                return jsonify({'error': 'Failed to load web interface'}), 500
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            try:
                return jsonify({
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0.0',
                    'models_loaded': {
                        'lda': self.models['lda'] is not None,
                        'nmf': self.models['nmf'] is not None
                    }
                }), 200
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
                return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

        @self.app.route('/models', methods=['GET'])
        def list_models():
            """List available models."""
            try:
                models_info = {}
                for model_type in ['lda', 'nmf']:
                    if self.models[model_type] is not None:
                        models_info[model_type] = {
                            'loaded': True,
                            'metadata': self.model_metadata.get(model_type, {})
                        }
                    else:
                        models_info[model_type] = {
                            'loaded': False,
                            'error': 'Model not loaded'
                        }
                
                return jsonify({
                    'models': models_info,
                    'total_models': len([m for m in self.models.values() if m is not None])
                }), 200
            except Exception as e:
                self.logger.error(f"Failed to list models: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/models/<model_type>/load', methods=['POST'])
        def load_model(model_type: str):
            """Load a specific model."""
            try:
                if model_type not in ['lda', 'nmf']:
                    return jsonify({'error': 'Invalid model type. Use "lda" or "nmf"'}), 400
                
                success = self._load_model(model_type)
                if success:
                    return jsonify({
                        'message': f'{model_type.upper()} model loaded successfully',
                        'model_type': model_type,
                        'metadata': self.model_metadata.get(model_type, {})
                    }), 200
                else:
                    return jsonify({'error': f'Failed to load {model_type} model'}), 500
                    
            except Exception as e:
                self.logger.error(f"Failed to load model {model_type}: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/models/<model_type>/topics', methods=['GET'])
        def get_topics(model_type: str):
            """Get topics from a specific model."""
            try:
                if model_type not in ['lda', 'nmf']:
                    return jsonify({'error': 'Invalid model type'}), 400
                
                if self.models[model_type] is None:
                    return jsonify({'error': f'{model_type.upper()} model not loaded'}), 404
                
                num_words = request.args.get('num_words', 10, type=int)
                topics = self.models[model_type].get_topics(num_words=num_words)
                
                return jsonify({
                    'model_type': model_type,
                    'topics': topics,
                    'num_topics': len(topics),
                    'num_words_per_topic': num_words
                }), 200
                
            except Exception as e:
                self.logger.error(f"Failed to get topics from {model_type}: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/predict', methods=['POST'])
        def predict_topics():
            """Predict topics for input text."""
            try:
                data = request.get_json()
                if not data or 'text' not in data:
                    return jsonify({'error': 'Text input required'}), 400
                
                text = data['text']
                model_type = data.get('model_type', 'lda')
                num_topics = data.get('num_topics', 5)
                
                if model_type not in ['lda', 'nmf']:
                    return jsonify({'error': 'Invalid model type'}), 400
                
                if self.models[model_type] is None:
                    return jsonify({'error': f'{model_type.upper()} model not loaded'}), 404
                
                # Predict topics for the input text
                prediction = self._predict_text_topics(text, model_type, num_topics)
                
                return jsonify({
                    'input_text': text,
                    'model_type': model_type,
                    'prediction': prediction,
                    'timestamp': datetime.now().isoformat()
                }), 200
                
            except Exception as e:
                self.logger.error(f"Prediction failed: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/predict/batch', methods=['POST'])
        def predict_batch():
            """Predict topics for multiple texts."""
            try:
                data = request.get_json()
                if not data or 'texts' not in data:
                    return jsonify({'error': 'Texts array required'}), 400
                
                texts = data['texts']
                model_type = data.get('model_type', 'lda')
                num_topics = data.get('num_topics', 5)
                
                if not isinstance(texts, list):
                    return jsonify({'error': 'Texts must be an array'}), 400
                
                if len(texts) > 100:  # Limit batch size
                    return jsonify({'error': 'Batch size too large (max 100)'}), 400
                
                if model_type not in ['lda', 'nmf']:
                    return jsonify({'error': 'Invalid model type'}), 400
                
                if self.models[model_type] is None:
                    return jsonify({'error': f'{model_type.upper()} model not loaded'}), 404
                
                # Predict topics for all texts
                predictions = []
                for i, text in enumerate(texts):
                    try:
                        prediction = self._predict_text_topics(text, model_type, num_topics)
                        predictions.append({
                            'index': i,
                            'text': text,
                            'prediction': prediction
                        })
                    except Exception as e:
                        predictions.append({
                            'index': i,
                            'text': text,
                            'error': str(e)
                        })
                
                return jsonify({
                    'model_type': model_type,
                    'predictions': predictions,
                    'total_texts': len(texts),
                    'successful_predictions': len([p for p in predictions if 'error' not in p]),
                    'timestamp': datetime.now().isoformat()
                }), 200
                
            except Exception as e:
                self.logger.error(f"Batch prediction failed: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/models/<model_type>/evaluate', methods=['GET'])
        def evaluate_model(model_type: str):
            """Get evaluation metrics for a model."""
            try:
                if model_type not in ['lda', 'nmf']:
                    return jsonify({'error': 'Invalid model type'}), 400
                
                if self.models[model_type] is None:
                    return jsonify({'error': f'{model_type.upper()} model not loaded'}), 404
                
                # Get model evaluation metrics
                metrics = self._get_model_metrics(model_type)
                
                return jsonify({
                    'model_type': model_type,
                    'metrics': metrics,
                    'timestamp': datetime.now().isoformat()
                }), 200
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate model {model_type}: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/models/<model_type>/visualize', methods=['GET'])
        def visualize_model(model_type: str):
            """Generate visualization for a model."""
            try:
                if model_type not in ['lda', 'nmf']:
                    return jsonify({'error': 'Invalid model type'}), 400
                
                if self.models[model_type] is None:
                    return jsonify({'error': f'{model_type.upper()} model not loaded'}), 404
                
                # Generate visualization
                viz_path = self._generate_visualization(model_type)
                
                if viz_path and os.path.exists(viz_path):
                    return send_file(viz_path, mimetype='image/png')
                else:
                    return jsonify({'error': 'Failed to generate visualization'}), 500
                    
            except Exception as e:
                self.logger.error(f"Failed to visualize model {model_type}: {e}")
                return jsonify({'error': str(e)}), 500

    def _setup_error_handlers(self) -> None:
        """Setup error handlers for the API."""
        
        @self.app.errorhandler(400)
        def bad_request(error):
            return jsonify({'error': 'Bad request', 'message': str(error)}), 400
        
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({'error': 'Not found', 'message': str(error)}), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({'error': 'Internal server error', 'message': str(error)}), 500
        
        @self.app.errorhandler(Exception)
        def handle_exception(e):
            self.logger.error(f"Unhandled exception: {e}")
            return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

    def _load_model(self, model_type: str) -> bool:
        """Load a specific model from disk."""
        try:
            models_path = self.config['api']['models_path']
            model_path = os.path.join(models_path, f"{model_type}_model.pkl")
            vectorizer_path = os.path.join(models_path, "tfidf_vectorizer.pkl")
            
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                return False
            
            if not os.path.exists(vectorizer_path):
                self.logger.error(f"Vectorizer file not found: {vectorizer_path}")
                return False
            
            # Load model
            if model_type == 'lda':
                self.models[model_type] = LDA()
                self.models[model_type].load_model(model_path)
            elif model_type == 'nmf':
                self.models[model_type] = NMF()
                self.models[model_type].load_model(model_path)
            
            # Load vectorizer
            self.vectorizers[model_type] = joblib.load(vectorizer_path)
            
            # Load metadata
            self.model_metadata[model_type] = {
                'model_path': model_path,
                'vectorizer_path': vectorizer_path,
                'loaded_at': datetime.now().isoformat(),
                'num_topics': getattr(self.models[model_type], 'num_topics', 'unknown')
            }
            
            self.logger.info(f"{model_type.upper()} model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load {model_type} model: {e}")
            return False

    def _predict_text_topics(self, text: str, model_type: str, num_topics: int) -> Dict[str, Any]:
        """Predict topics for a single text."""
        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Vectorize text
            vectorizer = self.vectorizers[model_type]
            text_vector = vectorizer.transform([processed_text])
            
            # Get topic predictions
            if model_type == 'lda':
                # For LDA, we need to use the original dictionary from training
                # Get the dictionary from the loaded model
                lda_model = self.models[model_type].get_model()
                
                # Convert text to bag of words using the model's dictionary
                tokens = processed_text.split()
                
                # Use the model's original dictionary (not create a new one)
                corpus = [lda_model.id2word.doc2bow(tokens)]
                
                # Get topic distribution
                topic_dist = lda_model.get_document_topics(
                    corpus[0], minimum_probability=0.0
                )
                
                # Sort by probability and get top topics
                topic_dist = sorted(topic_dist, key=lambda x: x[1], reverse=True)
                top_topics = topic_dist[:num_topics]
                
                prediction = {
                    'topics': [
                        {
                            'topic_id': int(topic_id),
                            'probability': float(prob),
                            'topic_words': self._get_topic_words(model_type, int(topic_id))
                        }
                        for topic_id, prob in top_topics
                    ],
                    'dominant_topic': int(top_topics[0][0]) if top_topics else None,
                    'confidence': float(top_topics[0][1]) if top_topics else 0.0
                }
                
            else:  # NMF
                # For NMF, use the transform method on new text
                nmf_model = self.models[model_type].get_model()
                
                # Transform the input text to get topic distribution
                doc_topics = nmf_model.transform(text_vector)
                doc_topics = doc_topics[0]  # Get first document
                
                # Convert to regular Python floats to avoid JSON serialization issues
                doc_topics = doc_topics.astype(np.float64)
                
                # Get top topics
                top_indices = np.argsort(doc_topics)[::-1][:num_topics]
                top_topics = [(int(i), float(doc_topics[i])) for i in top_indices]
                
                prediction = {
                    'topics': [
                        {
                            'topic_id': int(topic_id),
                            'probability': float(prob),
                            'topic_words': self._get_topic_words(model_type, int(topic_id))
                        }
                        for topic_id, prob in top_topics
                    ],
                    'dominant_topic': int(top_indices[0]) if len(top_indices) > 0 else None,
                    'confidence': float(doc_topics[top_indices[0]]) if len(top_indices) > 0 else 0.0
                }
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Failed to predict topics: {e}")
            raise AppException(f"Topic prediction failed: {e}")

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for prediction using the same pipeline as training."""
        try:
            from src.data.data_preprocessing import PreprocessingPipeline, PreprocessingConfig
            
            # Use the same preprocessing configuration as training
            config = PreprocessingConfig(
                text_column="content",
                spacy_model="en_core_web_lg",
                min_token_length=2,
                gen_bigrams=True,
                gen_trigrams=True
            )
            
            # Create preprocessing pipeline
            pipeline = PreprocessingPipeline(config, self.logger)
            
            # Process the text (returns DataFrame with cleaned_text column)
            result_df = pipeline.run([text])
            
            # Return the cleaned and tokenized text as a string
            return result_df['cleaned_text'].iloc[0]
            
        except Exception as e:
            self.logger.error(f"Text preprocessing failed: {e}")
            # Fallback to basic preprocessing if the full pipeline fails
            import re
            import html
            
            text = html.unescape(text)
            text = re.sub(r'<.*?>', ' ', text)
            text = text.lower()
            text = re.sub(r'[^a-z\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text

    def _get_topic_words(self, model_type: str, topic_id: int, num_words: int = 10) -> List[Dict[str, Any]]:
        """Get top words for a specific topic."""
        try:
            topics = self.models[model_type].get_topics(num_words=num_words)
            
            for topic in topics:
                if topic['topic_id'] == topic_id:
                    # Convert any numpy types to Python types for JSON serialization
                    words = []
                    for word, weight in topic['words']:
                        words.append({
                            'word': str(word),
                            'weight': float(weight)
                        })
                    return words
            
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to get topic words: {e}")
            return []

    def _get_model_metrics(self, model_type: str) -> Dict[str, Any]:
        """Get evaluation metrics for a model."""
        try:
            # This would typically load from evaluation results
            # For now, return basic model info
            return {
                'num_topics': getattr(self.models[model_type], 'num_topics', 'unknown'),
                'model_type': model_type,
                'loaded_at': self.model_metadata.get(model_type, {}).get('loaded_at', 'unknown')
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get model metrics: {e}")
            return {'error': str(e)}

    def _generate_visualization(self, model_type: str) -> Optional[str]:
        """Generate visualization for a model."""
        try:
            # This would generate topic visualizations
            # For now, return None (placeholder)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to generate visualization: {e}")
            return None

    def run(self, host: str = None, port: int = None, debug: bool = None) -> None:
        """Run the Flask application."""
        try:
            # Use config values or defaults
            host = host or self.config['api']['host']
            port = port or self.config['api']['port']
            debug = debug if debug is not None else self.config['api']['debug']
            
            self.logger.info(f"Starting Topic Modeling API on {host}:{port}")
            self.logger.info(f"Debug mode: {debug}")
            
            # Auto-load models if available
            self._auto_load_models()
            
            self.app.run(host=host, port=port, debug=debug)
            
        except Exception as e:
            self.logger.error(f"Failed to run API: {e}")
            raise AppException(f"API startup failed: {e}")

    def _auto_load_models(self) -> None:
        """Automatically load available models."""
        try:
            # Check if api config exists
            if 'api' not in self.config:
                self.logger.warning("API configuration not found, skipping auto-loading")
                return
                
            models_path = self.config['api'].get('models_path', 'artifacts')
            
            for model_type in ['lda', 'nmf']:
                model_path = os.path.join(models_path, f"{model_type}_model.pkl")
                if os.path.exists(model_path):
                    self.logger.info(f"Auto-loading {model_type} model...")
                    self._load_model(model_type)
                else:
                    self.logger.debug(f"No {model_type} model found at {model_path}")
                    
        except Exception as e:
            self.logger.warning(f"Auto-loading models failed: {e}")


def create_app(config_path: str = "config/config.yaml") -> Flask:
    """
    Create and configure the Flask application.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured Flask application
    """
    api = TopicModelingAPI(config_path)
    return api.app


def main():
    """Main entry point for the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Topic Modeling API Server")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Configuration file path")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    try:
        # Setup logging
        setup_logging(log_level="INFO")
        logger = get_logger(__name__)
        
        # Create and run API
        api = TopicModelingAPI(config_path=args.config)
        logger.info("Topic Modeling API starting...")
        
        api.run(host=args.host, port=args.port, debug=args.debug)
        
    except Exception as e:
        print(f"Failed to start API: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
