"""
Unified Topic Model Evaluation
Comprehensive evaluation for both LDA and NMF models including:
- Coherence Score
- Topic Diversity
- Topic Intrusion Tests
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

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.utils.logger import get_logger, setup_logging
from src.utils.exceptions import DataValidationError, AppException
from src.topics.LDA_model import LDA
from src.topics.NMF_model import NMF


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
                self.feature_names = self.vectorizer.get_feature_names_out().tolist()
            
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
            report = {
                "model_type": self.model_type,
                "model_path": self.model_path,
                "vectorizer_path": self.vectorizer_path,
                "data_path": self.data_path,
                "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "metrics": self.metrics,
                "topics": self.model.get_topics(num_words=10)
            }
            
            report_path = os.path.join(self.output_dir, f"{self.model_type}_evaluation.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
            
            self.logger.info(f"Evaluation report saved to {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving evaluation report: {str(e)}")


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


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Topic Model")
    parser.add_argument("--model_type", type=str, required=True, help="Model type: lda or nmf")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--vectorizer_path", type=str, required=True, help="Path to vectorizer")
    parser.add_argument("--data_path", type=str, required=True, help="Path to preprocessed data")
    parser.add_argument("--output_dir", type=str, default="artifacts/evaluation", help="Output directory")
    
    args = parser.parse_args()
    
    results = evaluate_model(
        model_type=args.model_type,
        model_path=args.model_path,
        vectorizer_path=args.vectorizer_path,
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    print(f"Evaluation completed. Results: {results}")