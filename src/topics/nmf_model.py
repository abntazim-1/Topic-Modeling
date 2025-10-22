"""
NMF Model Definition
Clean model structure for Non-Negative Matrix Factorization topic modeling.
"""

import os
import sys
import logging
import numpy as np
from typing import Any, List, Optional, Dict, Tuple, Union
from sklearn.decomposition import NMF as SklearnNMF
from scipy.sparse import issparse, csr_matrix

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.utils.logger import get_logger
from src.utils.exceptions import DataValidationError, AppException


class NMF:
    """
    NMF Topic Modeling model definition using sklearn.decomposition.NMF.
    Provides model structure, hyperparameters, and basic operations.
    """
    
    def __init__(
        self,
        num_topics: int = 10,
        init: str = 'nndsvda',
        max_iter: int = 600,
        random_state: int = 42,
        alpha: float = 0.1,
        l1_ratio: float = 0.5,
        beta_loss: str = 'frobenius',
        solver: str = 'mu',
        tol: float = 1e-4
    ) -> None:
        """
        Initialize NMF model with hyperparameters.
        
        Args:
            num_topics: Number of topics to extract.
            init: Initialization method ('random', 'nndsvd', 'nndsvda', 'nndsvdar').
            max_iter: Maximum number of iterations.
            random_state: Random state for reproducibility.
            alpha: Regularization parameter.
            l1_ratio: Ratio of L1 to L2 regularization.
            beta_loss: Beta divergence loss function.
            solver: Solver to use.
            tol: Tolerance for convergence.
        """
        self.num_topics = num_topics
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.beta_loss = beta_loss
        self.solver = solver
        self.tol = tol
        
        self.model: Optional[SklearnNMF] = None
        self.feature_names: Optional[List[str]] = None
        self.document_topic_matrix: Optional[np.ndarray] = None
        self.topic_term_matrix: Optional[np.ndarray] = None
        
        self.logger = get_logger(__name__)
        
        self.logger.info(
            f"NMF model initialized with num_topics={num_topics}, init={init}, "
            f"max_iter={max_iter}, alpha={alpha}, l1_ratio={l1_ratio}"
        )

    def get_model_params(self) -> Dict[str, Any]:
        """Get model hyperparameters as dictionary."""
        return {
            'num_topics': self.num_topics,
            'init': self.init,
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'alpha': self.alpha,
            'l1_ratio': self.l1_ratio,
            'beta_loss': self.beta_loss,
            'solver': self.solver,
            'tol': self.tol
        }

    def create_model(self) -> SklearnNMF:
        """
        Create sklearn NMF instance with configured parameters.
        
        Returns:
            Configured NMF instance.
        """
        return SklearnNMF(
            n_components=self.num_topics,
            init=self.init,
            max_iter=self.max_iter,
            random_state=self.random_state,
            alpha_W=self.alpha,
            l1_ratio=self.l1_ratio,
            beta_loss=self.beta_loss,
            solver=self.solver,
            tol=self.tol,
            verbose=1  # Enable verbose output for debugging
        )

    def set_model(self, model: SklearnNMF) -> None:
        """Set the trained model."""
        self.model = model
        self.logger.info("NMF model set successfully")

    def get_model(self) -> Optional[SklearnNMF]:
        """Get the current model."""
        return self.model

    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self.model is not None

    def set_feature_names(self, feature_names: List[str]) -> None:
        """Set feature names for topic extraction."""
        self.feature_names = feature_names

    def set_matrices(self, document_topic_matrix: np.ndarray, topic_term_matrix: np.ndarray) -> None:
        """Set the document-topic and topic-term matrices."""
        self.document_topic_matrix = document_topic_matrix
        self.topic_term_matrix = topic_term_matrix

    def get_topics(self, num_words: int = 10) -> List[Dict[str, Union[int, List[Tuple[str, float]]]]]:
        """
        Get top words for each topic.
        
        Args:
            num_words: Number of top words per topic.
            
        Returns:
            List of topics with words and weights.
            
        Raises:
            AppException: If model is not trained or feature names not set.
        """
        if not self.is_trained():
            raise AppException("NMF model is not trained yet.")
        
        if self.feature_names is None:
            raise AppException("Feature names not available.")
        
        if self.topic_term_matrix is None:
            raise AppException("Topic-term matrix not available.")
        
        # Validate matrix dimensions
        if len(self.feature_names) != self.topic_term_matrix.shape[1]:
            raise AppException(f"Feature names length ({len(self.feature_names)}) doesn't match topic-term matrix width ({self.topic_term_matrix.shape[1]})")
        
        topics = []
        for topic_idx in range(self.num_topics):
            # Get top word indices for this topic
            topic_weights = self.topic_term_matrix[topic_idx]
            
            # Check if topic has any non-zero weights
            if topic_weights.max() == 0:
                self.logger.warning(f"Topic {topic_idx} has all zero weights")
                # Still create the topic but with zero weights
                top_words = [(self.feature_names[i], 0.0) for i in range(min(num_words, len(self.feature_names)))]
            else:
                top_indices = topic_weights.argsort()[-num_words:][::-1]
                top_words = [
                    (self.feature_names[i], float(topic_weights[i]))
                    for i in top_indices
                ]
            
            topics.append({
                "topic_id": topic_idx,
                "words": [(word, round(weight, 4)) for word, weight in top_words]
            })
        
        return topics

    def get_document_topics(self) -> List[Dict[str, Union[int, float]]]:
        """
        Get dominant topic for each document.
        
        Returns:
            List of dicts with dominant topic and probabilities.
        """
        if not self.is_trained():
            raise AppException("NMF model is not trained yet.")
        
        if self.document_topic_matrix is None:
            raise AppException("Document-topic matrix not available.")
        
        document_topics = []
        for doc_idx, topic_dist in enumerate(self.document_topic_matrix):
            # Normalize to get probabilities
            topic_probs = topic_dist / (topic_dist.sum() + 1e-10)
            dominant_topic = int(np.argmax(topic_probs))
            dominant_prob = float(topic_probs[dominant_topic])
            
            document_topics.append({
                "document_id": doc_idx,
                "dominant_topic": dominant_topic,
                "dominant_probability": round(dominant_prob, 4),
                "topic_distribution": [round(float(p), 4) for p in topic_probs]
            })
        
        return document_topics

    def get_reconstruction_error(self) -> float:
        """Get the reconstruction error of the trained model."""
        if not self.is_trained():
            raise AppException("NMF model is not trained yet.")
        
        return float(self.model.reconstruction_err_)

    def save_model(self, path: str) -> None:
        """
        Save the trained model and associated data.
        
        Args:
            path: Path to save model.
        """
        if not self.is_trained():
            raise AppException("NMF model is not trained and cannot be saved.")
        
        import joblib
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'document_topic_matrix': self.document_topic_matrix,
            'topic_term_matrix': self.topic_term_matrix,
            'num_topics': self.num_topics,
            'init': self.init,
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'alpha': self.alpha,
            'l1_ratio': self.l1_ratio,
            'beta_loss': self.beta_loss,
            'solver': self.solver,
            'tol': self.tol
        }
        
        joblib.dump(model_data, path)
        self.logger.info(f"NMF model saved to: {path}")

    def load_model(self, path: str) -> None:
        """
        Load a model from path.
        
        Args:
            path: Path to load model from.
        """
        import joblib
        
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.document_topic_matrix = model_data['document_topic_matrix']
        self.topic_term_matrix = model_data['topic_term_matrix']
        self.num_topics = model_data['num_topics']
        self.init = model_data['init']
        self.max_iter = model_data['max_iter']
        self.random_state = model_data['random_state']
        self.alpha = model_data['alpha']
        self.l1_ratio = model_data['l1_ratio']
        self.beta_loss = model_data['beta_loss']
        self.solver = model_data['solver']
        self.tol = model_data['tol']
        
        self.logger.info(f"NMF model loaded from: {path}")


if __name__ == "__main__":
    # Example usage
    nmf = NMF(num_topics=10, init='nndsvda', max_iter=600)
    print(f"NMF model initialized with {nmf.num_topics} topics")
    print(f"Model parameters: {nmf.get_model_params()}")