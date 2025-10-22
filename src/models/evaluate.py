"""
Unified Topic Model Evaluation
Comprehensive evaluation for both LDA and NMF models including:
- Coherence Score
- Topic Diversity
- Topic Intrusion Tests
- Model Comparison
- Batch Evaluation
- Visualization Generation
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
from pathlib import Path
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import joblib
import gensim
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from scipy.sparse import csr_matrix
import argparse

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.utils.logger import get_logger, setup_logging
from src.utils.exceptions import DataValidationError, AppException
from src.topics.lda_model import LDA
from src.topics.nmf_model import NMF


class TopicModelEvaluator:
    """
    Unified evaluation framework for trained topic models (LDA or NMF).
    Measures interpretability, diversity, and quality of topics using
    statistical and semantic metrics.
    """
    
    def __init__(
        self,
        model_path: str,
        vectorizer_path: str,
        data_path: str,
        model_type: str,
        output_dir: str = "artifacts/evaluation",
        random_state: int = 42
    ) -> None:
        """
        Initialize the topic model evaluator.
        
        Args:
            model_path: Path to the trained model file.
            vectorizer_path: Path to the trained vectorizer file.
            data_path: Path to the preprocessed data file.
            model_type: Type of model ('lda' or 'nmf').
            output_dir: Directory to save evaluation outputs.
            random_state: Random state for reproducibility.
        """
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.data_path = data_path
        self.model_type = model_type.lower()
        self.output_dir = output_dir
        self.random_state = random_state
        
        self.logger = get_logger(__name__)
        self.model = None
        self.vectorizer = None
        self.texts = None
        self.corpus = None
        self.id2word = None
        self.matrix = None
        self.feature_names = None
        
        # Metrics storage
        self.metrics = {
            "coherence_score": None,
            "topic_diversity": None,
            "silhouette_score": None,
            "perplexity": None,
            "reconstruction_error": None,
            "topic_intrusion_score": None,
            "num_topics": 0,
            "evaluation_time": 0.0
        }
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"TopicModelEvaluator initialized for {self.model_type} model at {model_path}")

    def load_artifacts(self) -> None:
        """Load model, vectorizer, and data artifacts."""
        try:
            # Load model
            if self.model_type == "lda":
                self.model = LDA()
                self.model.load_model(self.model_path)
            elif self.model_type == "nmf":
                self.model = NMF()
                self.model.load_model(self.model_path)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            self.logger.info(f"Loading model from {self.model_path}")
            
            # Load vectorizer
            self.vectorizer = joblib.load(self.vectorizer_path)
            self.logger.info(f"Loading vectorizer from {self.vectorizer_path}")
            
            # Load data
            df = pd.read_csv(self.data_path)
            if 'tokens' in df.columns:
                import ast
                self.texts = df['tokens'].apply(ast.literal_eval).tolist()
            elif 'cleaned_tokens' in df.columns:
                import ast
                self.texts = df['cleaned_tokens'].apply(ast.literal_eval).tolist()
            else:
                # Fallback to cleaned_text
                self.texts = df['cleaned_text'].astype(str).apply(lambda s: s.split()).tolist()
            
            self.logger.info(f"Loading data from {self.data_path}")
            
            # Prepare corpus and matrices for evaluation
            if self.model_type == "lda":
                # Create gensim dictionary and corpus
                self.id2word = Dictionary(self.texts)
                self.corpus = [self.id2word.doc2bow(tokens) for tokens in self.texts]
            else:
                # For NMF, prepare TF-IDF matrix
                joined_texts = [" ".join(tokens) for tokens in self.texts]
                self.matrix = self.vectorizer.transform(joined_texts)
                self.feature_names = self.vectorizer.get_feature_names()
            
            self.logger.info("Successfully loaded all artifacts")
            
        except Exception as e:
            self.logger.error(f"Error loading artifacts: {str(e)}")
            raise AppException(f"Failed to load evaluation artifacts: {str(e)}")

    def compute_coherence(self, texts: Optional[List[str]] = None) -> float:
        """
        Calculate topic coherence using gensim.models.CoherenceModel.
        
        Args:
            texts: Tokenized texts for coherence computation.
            
        Returns:
            Coherence score (c_v measure).
        """
        try:
            if not self.model.is_trained():
                raise AppException("Model is not trained yet.")
            
            # Use provided texts or fallback to loaded texts
            eval_texts = texts if texts is not None else self.texts
            
            if self.model_type == "lda":
                # For LDA, use the trained model directly
                coherence_model = CoherenceModel(
                    model=self.model.get_model(),
                    texts=eval_texts,
                    dictionary=self.id2word,
                    coherence='c_v'
                )
                coherence = coherence_model.get_coherence()
            else:
                # For NMF, extract topics and compute coherence
                topics = self.model.get_topics(num_words=10)
                topic_words = [[word for word, _ in topic['words']] for topic in topics]
                
                coherence_model = CoherenceModel(
                    topics=topic_words,
                    texts=eval_texts,
                    dictionary=Dictionary(eval_texts),
                    coherence='c_v'
                )
                coherence = coherence_model.get_coherence()
            
            self.metrics["coherence_score"] = coherence
            self.logger.info(f"Coherence score (c_v): {coherence:.4f}")
            return coherence
            
        except Exception as e:
            self.logger.error(f"Error computing coherence: {str(e)}")
            self.metrics["coherence_score"] = None
            return 0.0

    def compute_topic_diversity(self, top_n: int = 10) -> float:
        """
        Calculate topic diversity as the ratio of unique words across all topics.
        
        Args:
            top_n: Number of top words per topic to consider.
            
        Returns:
            Topic diversity score (0-1, higher is better).
        """
        try:
            if not self.model.is_trained():
                raise AppException("Model is not trained yet.")
            
            topics = self.model.get_topics(num_words=top_n)
            all_words = []
            
            for topic in topics:
                words = [word for word, _ in topic['words']]
                all_words.extend(words)
            
            unique_words = len(set(all_words))
            total_words = len(all_words)
            diversity = unique_words / total_words if total_words > 0 else 0.0
            
            self.metrics["topic_diversity"] = diversity
            self.logger.info(f"Topic diversity (top {top_n} words): {diversity:.4f}")
            return diversity
            
        except Exception as e:
            self.logger.error(f"Error computing topic diversity: {str(e)}")
            self.metrics["topic_diversity"] = None
            return 0.0

    def compute_silhouette_score(self) -> Optional[float]:
        """
        Calculate silhouette score for topic clustering quality.
        
        Returns:
            Silhouette score or None if not computable.
        """
        try:
            if not self.model.is_trained():
                raise AppException("Model is not trained yet.")
            
            if self.model_type == "lda":
                # Get document-topic distributions
                doc_topics = []
                for doc_bow in self.corpus:
                    topic_probs = self.model.get_model().get_document_topics(doc_bow, minimum_probability=0.0)
                    topic_dist = [0.0] * self.model.num_topics
                    for topic_id, prob in topic_probs:
                        topic_dist[topic_id] = prob
                    doc_topics.append(topic_dist)
                
                doc_topics = np.array(doc_topics)
                dominant_topics = np.argmax(doc_topics, axis=1)
                
            else:
                # For NMF, use document-topic matrix
                if self.model.document_topic_matrix is None:
                    raise AppException("Document-topic matrix not available for NMF.")
                
                doc_topics = self.model.document_topic_matrix
                dominant_topics = np.argmax(doc_topics, axis=1)
            
            # Check if we have enough topics for silhouette score
            unique_topics = len(set(dominant_topics))
            if unique_topics < 2:
                self.logger.warning("All documents assigned to same topic, silhouette score undefined")
                self.metrics["silhouette_score"] = None
                return None
            
            # Compute silhouette score
            silhouette = silhouette_score(doc_topics, dominant_topics)
            self.metrics["silhouette_score"] = silhouette
            self.logger.info(f"Silhouette score: {silhouette:.4f}")
            return silhouette
            
        except Exception as e:
            self.logger.error(f"Error computing silhouette score: {str(e)}")
            self.metrics["silhouette_score"] = None
            return None

    def compute_perplexity(self) -> Optional[float]:
        """
        Calculate perplexity for LDA models.
        
        Returns:
            Perplexity score or None for NMF.
        """
        try:
            if self.model_type != "lda":
                self.logger.info("Perplexity not applicable for NMF models")
                return None
            
            if not self.model.is_trained():
                raise AppException("Model is not trained yet.")
            
            perplexity = self.model.compute_perplexity(self.corpus)
            self.metrics["perplexity"] = perplexity
            self.logger.info(f"Perplexity: {perplexity:.4f}")
            return perplexity
            
        except Exception as e:
            self.logger.error(f"Error computing perplexity: {str(e)}")
            self.metrics["perplexity"] = None
            return None

    def compute_reconstruction_error(self) -> Optional[float]:
        """
        Calculate reconstruction error for NMF models.
        
        Returns:
            Reconstruction error or None for LDA.
        """
        try:
            if self.model_type != "nmf":
                self.logger.info("Reconstruction error not applicable for LDA models")
                return None
            
            if not self.model.is_trained():
                raise AppException("Model is not trained yet.")
            
            reconstruction_error = self.model.get_reconstruction_error()
            self.metrics["reconstruction_error"] = reconstruction_error
            self.logger.info(f"Reconstruction error: {reconstruction_error:.4f}")
            return reconstruction_error
            
        except Exception as e:
            self.logger.error(f"Error computing reconstruction error: {str(e)}")
            self.metrics["reconstruction_error"] = None
            return None

    def compute_topic_intrusion(self, num_intruders: int = 1) -> float:
        """
        Compute topic intrusion score to measure topic quality.
        
        Args:
            num_intruders: Number of intruder words per topic.
            
        Returns:
            Topic intrusion score (0-1, higher is better).
        """
        try:
            if not self.model.is_trained():
                raise AppException("Model is not trained yet.")
            
            topics = self.model.get_topics(num_words=10)
            intrusion_scores = []
            
            for topic in topics:
                topic_words = [word for word, _ in topic['words']]
                
                # Simple intrusion test: check if top words are semantically related
                # This is a simplified version - in practice, you'd use word embeddings
                # or human evaluation for more accurate intrusion detection
                
                # For now, we'll use a simple heuristic based on word frequency
                # in the original corpus
                word_freq = {}
                for doc in self.texts:
                    for word in doc:
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                # Calculate average frequency of topic words
                topic_word_freqs = [word_freq.get(word, 0) for word in topic_words]
                avg_freq = np.mean(topic_word_freqs) if topic_word_freqs else 0
                
                # Higher frequency words are less likely to be intruders
                intrusion_score = min(1.0, avg_freq / 100.0)  # Normalize
                intrusion_scores.append(intrusion_score)
            
            avg_intrusion = np.mean(intrusion_scores) if intrusion_scores else 0.0
            self.metrics["topic_intrusion_score"] = avg_intrusion
            self.logger.info(f"Topic intrusion score: {avg_intrusion:.4f}")
            return avg_intrusion
            
        except Exception as e:
            self.logger.error(f"Error computing topic intrusion: {str(e)}")
            self.metrics["topic_intrusion_score"] = None
            return 0.0

    def evaluate(self) -> Dict[str, Any]:
        """
        Run comprehensive evaluation of the topic model.
        
        Returns:
            Dictionary containing all evaluation metrics.
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting comprehensive evaluation for {self.model_type} model...")
            
            # Load artifacts
            self.load_artifacts()
            
            # Compute all metrics
            self.compute_coherence()
            self.compute_topic_diversity()
            self.compute_silhouette_score()
            
            if self.model_type == "lda":
                self.compute_perplexity()
            else:
                self.compute_reconstruction_error()
            
            self.compute_topic_intrusion()
            
            # Set number of topics
            self.metrics["num_topics"] = self.model.num_topics
            
            # Calculate evaluation time
            self.metrics["evaluation_time"] = (datetime.now() - start_time).total_seconds()
            
            # Save evaluation report
            self._save_evaluation_report()
            
            self.logger.info(f"Evaluation complete for {self.model_type} model")
            self.logger.info(f"Metrics: {self.metrics}")
            
            return self.metrics
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            raise AppException(f"Evaluation failed: {str(e)}")

    def _save_evaluation_report(self) -> None:
        """Save evaluation report to JSON file."""
        try:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            report = {
                "model_type": self.model_type,
                "model_path": self.model_path,
                "vectorizer_path": self.vectorizer_path,
                "data_path": self.data_path,
                "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "metrics": convert_numpy_types(self.metrics),
                "topics": convert_numpy_types(self.model.get_topics(num_words=10))
            }
            
            report_path = os.path.join(self.output_dir, f"{self.model_type}_evaluation.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
            
            self.logger.info(f"Evaluation report saved to {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving evaluation report: {str(e)}")


class ModelComparison:
    """
    Comprehensive model comparison and visualization framework.
    """
    
    def __init__(self, output_dir: str = "artifacts/evaluation"):
        self.output_dir = output_dir
        self.logger = get_logger(__name__)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def compare_models(self, model_configs: List[Dict[str, str]], data_path: str) -> Dict[str, Any]:
        """
        Compare multiple models and generate comprehensive comparison report.
        
        Args:
            model_configs: List of model configurations with keys:
                - model_type: 'lda' or 'nmf'
                - model_path: Path to model file
                - vectorizer_path: Path to vectorizer file
                - name: Display name for the model
            data_path: Path to preprocessed data
            
        Returns:
            Dictionary containing comparison results and metrics
        """
        self.logger.info(f"Starting comparison of {len(model_configs)} models...")
        
        results = {}
        all_metrics = []
        
        for config in model_configs:
            model_name = config.get('name', config['model_type'])
            self.logger.info(f"Evaluating {model_name}...")
            
            try:
                evaluator = TopicModelEvaluator(
                    model_path=config['model_path'],
                    vectorizer_path=config['vectorizer_path'],
                    data_path=data_path,
                    model_type=config['model_type'],
                    output_dir=self.output_dir
                )
                
                metrics = evaluator.evaluate()
                results[model_name] = metrics
                all_metrics.append({
                    'name': model_name,
                    'type': config['model_type'],
                    'metrics': metrics
                })
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {str(e)}")
                results[model_name] = {
                    'error': str(e),
                    'topic_diversity': 0.0,
                    'coherence_score': 0.0,
                    'perplexity': 0.0,
                    'silhouette_score': 0.0,
                    'num_topics': 0
                }
        
        # Generate comparison visualizations
        self._create_comparison_visualizations(all_metrics)
        
        # Generate comparison report
        comparison_report = self._generate_comparison_report(all_metrics)
        
        # Save comprehensive results
        self._save_comparison_results(results, comparison_report)
        
        return {
            'individual_results': results,
            'comparison_report': comparison_report,
            'visualizations': self._get_visualization_paths()
        }
    
    def _create_comparison_visualizations(self, all_metrics: List[Dict]) -> None:
        """Create comprehensive comparison visualizations."""
        try:
            # Set up the plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create main comparison figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Comprehensive Model Comparison', fontsize=16, fontweight='bold')
            
            # Extract metrics for plotting
            model_names = [m['name'] for m in all_metrics]
            diversity_scores = [m['metrics'].get('topic_diversity', 0.0) for m in all_metrics]
            coherence_scores = [m['metrics'].get('coherence_score', 0.0) for m in all_metrics]
            perplexity_scores = [m['metrics'].get('perplexity', 0.0) if m['metrics'].get('perplexity') is not None else 0.0 for m in all_metrics]
            silhouette_scores = [m['metrics'].get('silhouette_score', 0.0) if m['metrics'].get('silhouette_score') is not None else 0.0 for m in all_metrics]
            
            # Plot 1: Topic Diversity
            axes[0, 0].bar(model_names, diversity_scores, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('Topic Diversity (Higher is Better)')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].set_ylim(0, 1)
            axes[0, 0].tick_params(axis='x', rotation=45)
            for i, v in enumerate(diversity_scores):
                axes[0, 0].text(i, v + 0.02, f"{v:.4f}", ha='center', va='bottom')
            
            # Plot 2: Coherence Score
            axes[0, 1].bar(model_names, coherence_scores, color='lightcoral', alpha=0.7)
            axes[0, 1].set_title('Coherence Score (Higher is Better)')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
            for i, v in enumerate(coherence_scores):
                axes[0, 1].text(i, v + 0.02, f"{v:.4f}", ha='center', va='bottom')
            
            # Plot 3: Perplexity (Lower is Better)
            axes[1, 0].bar(model_names, perplexity_scores, color='lightgreen', alpha=0.7)
            axes[1, 0].set_title('Perplexity (Lower is Better)')
            axes[1, 0].set_ylabel('Perplexity')
            axes[1, 0].tick_params(axis='x', rotation=45)
            for i, v in enumerate(perplexity_scores):
                axes[1, 0].text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom')
            
            # Plot 4: Silhouette Score
            axes[1, 1].bar(model_names, silhouette_scores, color='gold', alpha=0.7)
            axes[1, 1].set_title('Silhouette Score (Higher is Better)')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
            for i, v in enumerate(silhouette_scores):
                axes[1, 1].text(i, v + 0.02, f"{v:.4f}", ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save the comparison plot
            comparison_path = os.path.join(self.output_dir, "comprehensive_model_comparison.png")
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create radar chart for multi-dimensional comparison
            self._create_radar_chart(all_metrics)
            
            # Create heatmap for detailed metrics
            self._create_metrics_heatmap(all_metrics)
            
            self.logger.info(f"Comparison visualizations saved to {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error creating comparison visualizations: {str(e)}")
    
    def _create_radar_chart(self, all_metrics: List[Dict]) -> None:
        """Create radar chart for multi-dimensional model comparison."""
        try:
            # Normalize metrics to 0-1 scale for radar chart
            metrics_data = []
            model_names = []
            
            for model in all_metrics:
                metrics = model['metrics']
                # Normalize perplexity (invert and scale) - handle None values
                perplexity = metrics.get('perplexity')
                if perplexity is not None and perplexity > 0:
                    normalized_perplexity = max(0, 1 - (perplexity / 1000))
                else:
                    normalized_perplexity = 0
                
                # Handle silhouette score None values
                silhouette = metrics.get('silhouette_score')
                silhouette_score = max(0, silhouette) if silhouette is not None else 0
                
                metrics_data.append([
                    metrics.get('topic_diversity', 0.0),
                    metrics.get('coherence_score', 0.0),
                    normalized_perplexity,
                    silhouette_score
                ])
                model_names.append(model['name'])
            
            # Create radar chart
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            # Define metrics labels
            metrics_labels = ['Topic Diversity', 'Coherence', 'Perplexity (inverted)', 'Silhouette']
            
            # Set up angles for radar chart
            angles = np.linspace(0, 2 * np.pi, len(metrics_labels), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            # Plot each model
            colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
            
            for i, (model_name, data) in enumerate(zip(model_names, metrics_data)):
                data += data[:1]  # Complete the circle
                ax.plot(angles, data, 'o-', linewidth=2, label=model_name, color=colors[i])
                ax.fill(angles, data, alpha=0.25, color=colors[i])
            
            # Customize the chart
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics_labels)
            ax.set_ylim(0, 1)
            ax.set_title('Multi-dimensional Model Comparison', size=16, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.grid(True)
            
            plt.tight_layout()
            radar_path = os.path.join(self.output_dir, "model_comparison_radar.png")
            plt.savefig(radar_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating radar chart: {str(e)}")
    
    def _create_metrics_heatmap(self, all_metrics: List[Dict]) -> None:
        """Create heatmap of all metrics for all models."""
        try:
            # Prepare data for heatmap
            metrics_names = ['Topic Diversity', 'Coherence', 'Perplexity', 'Silhouette', 'Num Topics']
            model_names = [m['name'] for m in all_metrics]
            
            # Create matrix for heatmap
            heatmap_data = []
            for model in all_metrics:
                metrics = model['metrics']
                
                # Handle None values properly
                perplexity = metrics.get('perplexity')
                perplexity_val = perplexity if perplexity is not None else 0.0
                
                silhouette = metrics.get('silhouette_score')
                silhouette_val = max(0, silhouette) if silhouette is not None else 0.0
                
                row = [
                    metrics.get('topic_diversity', 0.0),
                    metrics.get('coherence_score', 0.0),
                    perplexity_val,
                    silhouette_val,
                    metrics.get('num_topics', 0)
                ]
                heatmap_data.append(row)
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(heatmap_data, 
                       xticklabels=metrics_names, 
                       yticklabels=model_names,
                       annot=True, 
                       fmt='.4f', 
                       cmap='YlOrRd',
                       ax=ax)
            
            ax.set_title('Model Metrics Heatmap', fontsize=14, fontweight='bold')
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Models')
            
            plt.tight_layout()
            heatmap_path = os.path.join(self.output_dir, "model_metrics_heatmap.png")
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating metrics heatmap: {str(e)}")
    
    def _generate_comparison_report(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive comparison report."""
        report = {
            'summary': {},
            'winners': {},
            'detailed_analysis': {}
        }
        
        # Find winners for each metric
        metrics_to_compare = ['topic_diversity', 'coherence_score', 'silhouette_score']
        lower_is_better = ['perplexity']
        
        for metric in metrics_to_compare:
            # Filter out models with None values for this metric
            valid_models = [m for m in all_metrics if m['metrics'].get(metric) is not None]
            if valid_models:
                best_model = max(valid_models, key=lambda x: x['metrics'].get(metric, 0.0))
                report['winners'][metric] = {
                    'model': best_model['name'],
                    'score': best_model['metrics'].get(metric, 0.0)
                }
            else:
                report['winners'][metric] = {
                    'model': 'N/A',
                    'score': 0.0
                }
        
        for metric in lower_is_better:
            # Filter out models with None values for this metric
            valid_models = [m for m in all_metrics if m['metrics'].get(metric) is not None]
            if valid_models:
                best_model = min(valid_models, key=lambda x: x['metrics'].get(metric, float('inf')))
                report['winners'][metric] = {
                    'model': best_model['name'],
                    'score': best_model['metrics'].get(metric, 0.0)
                }
            else:
                report['winners'][metric] = {
                    'model': 'N/A',
                    'score': 0.0
                }
        
        # Generate summary statistics
        for metric in metrics_to_compare + lower_is_better:
            scores = [m['metrics'].get(metric, 0.0) for m in all_metrics]
            # Filter out None values for proper statistics
            valid_scores = [s for s in scores if s is not None]
            if valid_scores:
                report['summary'][metric] = {
                    'mean': np.mean(valid_scores),
                    'std': np.std(valid_scores),
                    'min': np.min(valid_scores),
                    'max': np.max(valid_scores)
                }
            else:
                report['summary'][metric] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0
                }
        
        return report
    
    def _save_comparison_results(self, results: Dict, comparison_report: Dict) -> None:
        """Save comprehensive comparison results."""
        try:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            comparison_data = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'individual_results': convert_numpy_types(results),
                'comparison_report': convert_numpy_types(comparison_report),
                'summary': {
                    'total_models': len(results),
                    'successful_evaluations': len([r for r in results.values() if 'error' not in r]),
                    'failed_evaluations': len([r for r in results.values() if 'error' in r])
                }
            }
            
            results_path = os.path.join(self.output_dir, "comprehensive_comparison_results.json")
            with open(results_path, 'w') as f:
                json.dump(comparison_data, f, indent=4)
            
            self.logger.info(f"Comparison results saved to {results_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving comparison results: {str(e)}")
    
    def _get_visualization_paths(self) -> List[str]:
        """Get paths to generated visualizations."""
        return [
            os.path.join(self.output_dir, "comprehensive_model_comparison.png"),
            os.path.join(self.output_dir, "model_comparison_radar.png"),
            os.path.join(self.output_dir, "model_metrics_heatmap.png")
        ]


def evaluate_model(model_type: str, model_path: str, vectorizer_path: str, 
                  data_path: str, output_dir: str = "artifacts/evaluation") -> Dict[str, Any]:
    """
    Convenience function to evaluate a single model.
    
    Args:
        model_type: Type of model ('lda' or 'nmf').
        model_path: Path to trained model.
        vectorizer_path: Path to trained vectorizer.
        data_path: Path to preprocessed data.
        output_dir: Output directory for results.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    evaluator = TopicModelEvaluator(
        model_path=model_path,
        vectorizer_path=vectorizer_path,
        data_path=data_path,
        model_type=model_type,
        output_dir=output_dir
    )
    
    return evaluator.evaluate()


def evaluate_batch_models(model_configs: List[Dict[str, str]], data_path: str, 
                          output_dir: str = "artifacts/evaluation") -> Dict[str, Any]:
    """
    Evaluate multiple models in batch and generate comparison.
    
    Args:
        model_configs: List of model configurations
        data_path: Path to preprocessed data
        output_dir: Output directory for results
        
    Returns:
        Dictionary containing batch evaluation results
    """
    comparison = ModelComparison(output_dir)
    return comparison.compare_models(model_configs, data_path)


def auto_discover_and_evaluate(data_path: str, artifacts_dir: str = "artifacts", 
                              output_dir: str = "artifacts/evaluation") -> Dict[str, Any]:
    """
    Automatically discover and evaluate all available models.
    
    Args:
        data_path: Path to preprocessed data
        artifacts_dir: Directory containing model artifacts
        output_dir: Output directory for results
        
    Returns:
        Dictionary containing evaluation results
    """
    logger = get_logger(__name__)
    logger.info("Auto-discovering models for evaluation...")
    
    model_configs = []
    
    # Look for LDA models
    lda_model_path = os.path.join(artifacts_dir, "lda_model.pkl")
    vectorizer_path = os.path.join(artifacts_dir, "tfidf_vectorizer.pkl")
    
    if os.path.exists(lda_model_path) and os.path.exists(vectorizer_path):
        model_configs.append({
            'model_type': 'lda',
            'model_path': lda_model_path,
            'vectorizer_path': vectorizer_path,
            'name': 'LDA Model'
        })
        logger.info("Found LDA model")
    
    # Look for NMF models
    nmf_model_path = os.path.join(artifacts_dir, "nmf_model.pkl")
    
    if os.path.exists(nmf_model_path) and os.path.exists(vectorizer_path):
        model_configs.append({
            'model_type': 'nmf',
            'model_path': nmf_model_path,
            'vectorizer_path': vectorizer_path,
            'name': 'NMF Model'
        })
        logger.info("Found NMF model")
    
    if not model_configs:
        logger.warning("No models found for evaluation")
        return {'error': 'No models found for evaluation'}
    
    logger.info(f"Found {len(model_configs)} models for evaluation")
    return evaluate_batch_models(model_configs, data_path, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive Topic Model Evaluation")
    parser.add_argument("--mode", type=str, choices=['single', 'batch', 'auto'], default='auto',
                       help="Evaluation mode: single model, batch models, or auto-discover")
    parser.add_argument("--model_type", type=str, choices=['lda', 'nmf'], help="Model type for single evaluation")
    parser.add_argument("--model_path", type=str, help="Path to trained model")
    parser.add_argument("--vectorizer_path", type=str, help="Path to vectorizer")
    parser.add_argument("--data_path", type=str, default="artifacts/preprocessed_bbc_news.csv", 
                       help="Path to preprocessed data")
    parser.add_argument("--output_dir", type=str, default="artifacts/evaluation", help="Output directory")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="Artifacts directory for auto-discovery")
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.model_type or not args.model_path or not args.vectorizer_path:
            print("Error: --model_type, --model_path, and --vectorizer_path are required for single mode")
            sys.exit(1)
        
        results = evaluate_model(
            model_type=args.model_type,
            model_path=args.model_path,
            vectorizer_path=args.vectorizer_path,
            data_path=args.data_path,
            output_dir=args.output_dir
        )
        print(f"Single model evaluation completed. Results: {results}")
        
    elif args.mode == 'batch':
        # For batch mode, you would need to provide model configurations
        # This is a placeholder - in practice, you'd load configs from a file
        print("Batch mode requires model configurations. Use auto mode for automatic discovery.")
        
    elif args.mode == 'auto':
        results = auto_discover_and_evaluate(
            data_path=args.data_path,
            artifacts_dir=args.artifacts_dir,
            output_dir=args.output_dir
        )
        print(f"Auto-discovery evaluation completed. Results saved to {args.output_dir}")
        
        # Print summary
        if 'individual_results' in results:
            print("\nEvaluation Summary:")
            for model_name, metrics in results['individual_results'].items():
                if 'error' not in metrics:
                    print(f"{model_name}:")
                    print(f"  - Topic Diversity: {metrics.get('topic_diversity', 0.0):.4f}")
                    print(f"  - Coherence Score: {metrics.get('coherence_score', 0.0):.4f}")
                    
                    # Handle None values for perplexity
                    perplexity = metrics.get('perplexity')
                    if perplexity is not None:
                        print(f"  - Perplexity: {perplexity:.2f}")
                    else:
                        print(f"  - Perplexity: N/A")
                    
                    # Handle None values for silhouette score
                    silhouette = metrics.get('silhouette_score')
                    if silhouette is not None:
                        print(f"  - Silhouette Score: {silhouette:.4f}")
                    else:
                        print(f"  - Silhouette Score: N/A")
                    
                    print(f"  - Number of Topics: {metrics.get('num_topics', 0)}")
                else:
                    print(f"{model_name}: Error - {metrics['error']}")