"""
LDA Model Definition
Clean model structure for Latent Dirichlet Allocation topic modeling.
"""

import os
import sys
import logging
from typing import Any, List, Optional, Dict, Tuple, Union
from gensim.models import LdaModel, CoherenceModel
from gensim.corpora import Dictionary

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.utils.logger import get_logger
from src.utils.exceptions import DataValidationError, AppException


class LDA:
    """
    LDA Topic Modeling model definition using gensim.models.LdaModel.
    Provides model structure, hyperparameters, and basic operations.
    """
    
    def __init__(
        self,
        num_topics: int = 10,
        passes: int = 20,
        chunksize: int = 2000,
        alpha: str = 'auto',
        eta: Union[str, float] = 'auto',
        iterations: int = 800,
        update_every: int = 1,
        eval_every: Optional[int] = None,
        decay: float = 0.5,
        offset: float = 10.0,
        minimum_probability: float = 0.0,
        minimum_phi_value: float = 1e-8,
        random_state: int = 42
    ) -> None:
        """
        Initialize LDA model with hyperparameters.
        
        Args:
            num_topics: Number of topics for LDA.
            passes: Number of passes over corpus during training.
            chunksize: Number of documents to use in each training chunk.
            alpha: Document-topic density prior.
            eta: Topic-word density prior.
            iterations: Maximum number of iterations.
            update_every: Update model every N iterations.
            eval_every: Evaluate perplexity every N iterations.
            decay: Decay factor for learning rate.
            offset: Offset for learning rate.
            minimum_probability: Minimum probability threshold.
            minimum_phi_value: Minimum phi value threshold.
            random_state: Random state for reproducibility.
        """
        self.num_topics = num_topics
        self.passes = passes
        self.chunksize = chunksize
        self.alpha = alpha
        self.eta = eta
        self.iterations = iterations
        self.update_every = update_every
        self.eval_every = eval_every
        self.decay = decay
        self.offset = offset
        self.minimum_probability = minimum_probability
        self.minimum_phi_value = minimum_phi_value
        self.random_state = random_state
        
        self.model: Optional[LdaModel] = None
        self.logger = get_logger(__name__)
        
        self.logger.info(
            f"LDA model initialized with num_topics={num_topics}, passes={passes}, "
            f"chunksize={chunksize}, alpha={alpha}, eta={eta}"
        )

    def get_model_params(self) -> Dict[str, Any]:
        """Get model hyperparameters as dictionary."""
        return {
            'num_topics': self.num_topics,
            'passes': self.passes,
            'chunksize': self.chunksize,
            'alpha': self.alpha,
            'eta': self.eta,
            'iterations': self.iterations,
            'update_every': self.update_every,
            'eval_every': self.eval_every,
            'decay': self.decay,
            'offset': self.offset,
            'minimum_probability': self.minimum_probability,
            'minimum_phi_value': self.minimum_phi_value,
            'random_state': self.random_state
        }

    def create_model(self, corpus=None, id2word=None):
        """
        Create gensim LdaModel instance with configured parameters.
        
        Args:
            corpus: Gensim corpus for training.
            id2word: Gensim dictionary for training.
        
        Returns:
            Configured LdaModel instance.
        """
        return LdaModel(
            corpus=corpus,
            id2word=id2word,
            num_topics=self.num_topics,
            passes=self.passes,
            chunksize=self.chunksize,
            alpha=self.alpha,
            eta=self.eta,
            iterations=self.iterations,
            update_every=self.update_every,
            eval_every=self.eval_every,
            decay=self.decay,
            offset=self.offset,
            minimum_probability=self.minimum_probability,
            minimum_phi_value=self.minimum_phi_value,
            random_state=self.random_state
        )

    def set_model(self, model: LdaModel) -> None:
        """Set the trained model."""
        self.model = model
        self.logger.info("LDA model set successfully")

    def get_model(self) -> Optional[LdaModel]:
        """Get the current model."""
        return self.model

    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self.model is not None

    def get_topics(self, num_words: int = 10) -> List[Dict[str, Union[int, List[Tuple[str, float]]]]]:
        """
        Get top words for each topic.
        
        Args:
            num_words: Number of top words per topic.
            
        Returns:
            List of topics with words and weights.
            
        Raises:
            AppException: If model is not trained.
        """
        if not self.is_trained():
            raise AppException("LDA model is not trained yet.")
        
        topics = []
        for idx, topic in self.model.show_topics(
            num_topics=self.num_topics, 
            num_words=num_words, 
            formatted=False
        ):
            topics.append({
                "topic_id": idx,
                "words": [(word, round(weight, 4)) for word, weight in topic]
            })
        
        return topics

    def get_dominant_topic_per_doc(self, corpus: List[List[tuple]]) -> List[int]:
        """
        Get dominant topic for each document.
        
        Args:
            corpus: Gensim corpus (list of bag-of-words).
            
        Returns:
            List of dominant topic indices per document.
        """
        if not self.is_trained():
            raise AppException("LDA model is not trained yet.")
        
        dominant_topics = []
        for doc_bow in corpus:
            topic_probs = self.model.get_document_topics(doc_bow)
            if topic_probs:
                dominant_topic = max(topic_probs, key=lambda tup: tup[1])[0]
            else:
                dominant_topic = -1
            dominant_topics.append(dominant_topic)
        
        return dominant_topics

    def compute_perplexity(self, corpus: List[List[tuple]]) -> float:
        """
        Compute model perplexity.
        
        Args:
            corpus: Corpus to compute perplexity against.
            
        Returns:
            Perplexity value.
        """
        if not self.is_trained():
            raise AppException("LDA model is not trained yet.")
        
        return self.model.log_perplexity(corpus)

    def save_model(self, path: str) -> None:
        """
        Save the trained model.
        
        Args:
            path: Path to save model.
        """
        if not self.is_trained():
            raise AppException("LDA model is not trained and cannot be saved.")
        
        self.model.save(path)
        self.logger.info(f"LDA model saved to: {path}")

    def load_model(self, path: str) -> None:
        """
        Load a model from path.
        
        Args:
            path: Path to load model from.
        """
        self.model = LdaModel.load(path)
        self.logger.info(f"LDA model loaded from: {path}")


if __name__ == "__main__":
    # Example usage
    lda = LDA(num_topics=10, passes=20)
    print(f"LDA model initialized with {lda.num_topics} topics")
    print(f"Model parameters: {lda.get_model_params()}")