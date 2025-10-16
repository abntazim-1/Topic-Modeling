import logging
import joblib
import numpy as np
from typing import Any, List, Optional, Dict, Tuple, Union
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import issparse, csr_matrix
from src.utils.logger import get_logger
from src.utils.exceptions import DataValidationError, AppException


class NMFModeler:
    """
    Non-Negative Matrix Factorization (NMF) Topic Modeling wrapper using 
    sklearn.decomposition.NMF. Provides training, evaluation, topic extraction, 
    visualization, and persistence utilities with robust logging and error handling.
    """
    
    def __init__(
        self,
        num_topics: int = 10,
        init: str = 'nndsvda',
        max_iter: int = 500,
        random_state: int = 42,
        alpha: float = 0.1,
        l1_ratio: float = 0.5
    ) -> None:
        """
        Initializes the NMF Modeler.
        
        Args:
            num_topics: Number of topics to extract.
            init: Initialization method ('random', 'nndsvd', 'nndsvda', 'nndsvdar').
            max_iter: Maximum number of iterations.
            random_state: Random state for reproducibility.
            alpha: Regularization parameter (multiplier for regularization terms).
            l1_ratio: Ratio of L1 to L2 regularization (0.0 = L2 only, 1.0 = L1 only).
        """
        self.num_topics = num_topics
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        
        self.model: Optional[NMF] = None
        self.feature_names: Optional[List[str]] = None
        self.document_topic_matrix: Optional[np.ndarray] = None
        self.topic_term_matrix: Optional[np.ndarray] = None
        
        self.logger = get_logger(__name__)
        self.logger.info(
            f"NMFModeler initialized with num_topics={num_topics}, init={init}, "
            f"max_iter={max_iter}, alpha={alpha}, l1_ratio={l1_ratio}"
        )
    
    def train(
        self, 
        tfidf_matrix: Union[np.ndarray, csr_matrix], 
        feature_names: List[str]
    ) -> None:
        """
        Fits the NMF model to the provided TF-IDF matrix.
        
        Args:
            tfidf_matrix: TF-IDF matrix (can be sparse or dense).
            feature_names: List of feature names (vocabulary).
            
        Raises:
            DataValidationError: On invalid input data.
            AppException: On training errors.
        """
        try:
            # Validate inputs
            if tfidf_matrix is None or (isinstance(tfidf_matrix, np.ndarray) and tfidf_matrix.size == 0):
                raise DataValidationError("TF-IDF matrix is empty or None.")
            
            if not feature_names or not isinstance(feature_names, list):
                raise DataValidationError("feature_names must be a non-empty list.")
            
            # Convert to sparse matrix if dense
            if isinstance(tfidf_matrix, np.ndarray):
                self.logger.info("Converting dense matrix to sparse format for efficiency.")
                tfidf_matrix = csr_matrix(tfidf_matrix)
            
            # Validate matrix dimensions
            if tfidf_matrix.shape[1] != len(feature_names):
                raise DataValidationError(
                    f"Matrix columns ({tfidf_matrix.shape[1]}) must match "
                    f"feature_names length ({len(feature_names)})."
                )
            
            self.feature_names = feature_names
            
            self.logger.info(
                f"Starting NMF training on matrix of shape {tfidf_matrix.shape}..."
            )
            
            # Initialize and train NMF model
            self.model = NMF(
                n_components=self.num_topics,
                init=self.init,
                max_iter=self.max_iter,
                random_state=self.random_state,
                alpha_W=self.alpha,  # Changed from alpha to alpha_W
                l1_ratio=self.l1_ratio,
                verbose=0
            )
            
            # Fit and transform
            self.document_topic_matrix = self.model.fit_transform(tfidf_matrix)
            self.topic_term_matrix = self.model.components_
            
            self.logger.info(
                f"NMF training complete. Reconstruction error: "
                f"{self.model.reconstruction_err_:.4f}"
            )
            self.logger.info(
                f"Document-topic matrix shape: {self.document_topic_matrix.shape}"
            )
            self.logger.info(
                f"Topic-term matrix shape: {self.topic_term_matrix.shape}"
            )
            
        except DataValidationError as e:
            self.logger.error(f"Data validation error during NMF training: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error during NMF model training: {e}")
            raise AppException(f"NMF training failed: {str(e)}")
    
    def get_topics(
        self, 
        num_words: int = 10
    ) -> List[Dict[str, Union[int, List[Tuple[str, float]]]]]:
        """
        Returns the top words for each topic.
        
        Args:
            num_words: Number of top words per topic.
            
        Returns:
            List of topics, each as dict with topic_id and (word, weight) tuples.
            
        Raises:
            AppException: If model is not trained.
        """
        try:
            if self.model is None or self.topic_term_matrix is None:
                raise AppException("NMF model is not trained yet.")
            
            if self.feature_names is None:
                raise AppException("Feature names not available.")
            
            topics = []
            
            for topic_idx in range(self.num_topics):
                # Get top word indices for this topic
                top_indices = self.topic_term_matrix[topic_idx].argsort()[-num_words:][::-1]
                top_words = [
                    (self.feature_names[i], float(self.topic_term_matrix[topic_idx][i]))
                    for i in top_indices
                ]
                
                topics.append({
                    "topic_id": topic_idx,
                    "words": [(word, round(weight, 4)) for word, weight in top_words]
                })
                
                # Log top words
                words_str = ", ".join([f"{word}({weight:.3f})" for word, weight in top_words[:5]])
                self.logger.info(f"Topic {topic_idx}: {words_str}...")
            
            self.logger.info(f"Extracted top {num_words} words for {self.num_topics} topics.")
            return topics
            
        except AppException as e:
            self.logger.error(f"Failed to get topic words: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in get_topics: {e}")
            raise AppException(f"Failed to extract topics: {str(e)}")
    
    def get_document_topics(self) -> List[Dict[str, Union[int, float]]]:
        """
        Returns the dominant topic for each document along with topic distribution.
        
        Returns:
            List of dicts containing dominant_topic, probability, and full distribution.
            
        Raises:
            AppException: If model is not trained.
        """
        try:
            if self.document_topic_matrix is None:
                raise AppException("NMF model is not trained yet.")
            
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
            
            self.logger.info(
                f"Extracted dominant topics for {len(document_topics)} documents."
            )
            return document_topics
            
        except Exception as e:
            self.logger.error(f"Failed to extract document topics: {e}")
            raise AppException(f"Document topic extraction failed: {str(e)}")
    
    def compute_coherence(
        self,
        texts: List[List[str]],
        measure: str = 'c_v',
        num_words: int = 10
    ) -> float:
        """
        Compute coherence score for the extracted topics using Gensim.
        
        Args:
            texts: List of tokenized documents (list of list of strings).
            measure: Coherence measure ('c_v', 'u_mass', 'c_uci', 'c_npmi').
            num_words: Number of top words per topic to use for coherence.
            
        Returns:
            Coherence score (float).
            
        Raises:
            AppException: On error or if model not trained.
        """
        try:
            if self.model is None or self.topic_term_matrix is None:
                raise AppException("NMF model is not trained yet.")
            
            if not texts or not isinstance(texts, list):
                raise DataValidationError("texts must be a non-empty list of tokenized documents.")
            
            # Create Gensim dictionary from texts
            dictionary = Dictionary(texts)
            
            # Extract top words for each topic
            topic_words = []
            for topic_idx in range(self.num_topics):
                top_indices = self.topic_term_matrix[topic_idx].argsort()[-num_words:][::-1]
                top_words = [self.feature_names[i] for i in top_indices]
                topic_words.append(top_words)
            
            # Compute coherence using Gensim
            coherence_model = CoherenceModel(
                topics=topic_words,
                texts=texts,
                dictionary=dictionary,
                coherence=measure
            )
            
            coherence_score = coherence_model.get_coherence()
            
            self.logger.info(
                f"Coherence score ({measure}): {coherence_score:.4f}"
            )
            
            return coherence_score
            
        except DataValidationError as e:
            self.logger.error(f"Data validation error in coherence computation: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to compute coherence score: {e}")
            raise AppException(f"Coherence computation failed: {str(e)}")
    
    def reduce_dimensions_pca(
        self,
        n_components: int = 2
    ) -> np.ndarray:
        """
        Apply PCA to document-topic matrix for visualization.
        
        Args:
            n_components: Number of principal components (default: 2 for 2D viz).
            
        Returns:
            Reduced dimensional representation of documents.
            
        Raises:
            AppException: If model is not trained.
        """
        try:
            if self.document_topic_matrix is None:
                raise AppException("NMF model is not trained yet.")
            
            self.logger.info(
                f"Applying PCA to reduce dimensions to {n_components} components..."
            )
            
            pca = PCA(n_components=n_components, random_state=self.random_state)
            reduced_data = pca.fit_transform(self.document_topic_matrix)
            
            explained_variance = pca.explained_variance_ratio_
            self.logger.info(
                f"PCA complete. Explained variance: "
                f"{', '.join([f'{v:.2%}' for v in explained_variance])}"
            )
            
            return reduced_data
            
        except Exception as e:
            self.logger.error(f"PCA dimensionality reduction failed: {e}")
            raise AppException(f"PCA failed: {str(e)}")
    
    def visualize_topic_heatmap(
        self,
        num_words: int = 10,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize topic-term matrix as a heatmap.
        
        Args:
            num_words: Number of top words per topic to display.
            figsize: Figure size (width, height).
            save_path: Optional path to save the figure.
            
        Raises:
            AppException: If model is not trained.
        """
        try:
            if self.model is None or self.topic_term_matrix is None:
                raise AppException("NMF model is not trained yet.")
            
            if self.feature_names is None:
                raise AppException("Feature names not available.")
            
            self.logger.info("Creating topic-term heatmap visualization...")
            
            # Get top words for each topic
            top_words_per_topic = []
            heatmap_data = []
            
            for topic_idx in range(self.num_topics):
                top_indices = self.topic_term_matrix[topic_idx].argsort()[-num_words:][::-1]
                top_words = [self.feature_names[i] for i in top_indices]
                top_words_per_topic.extend(top_words)
                
                # Extract weights for these words
                weights = [self.topic_term_matrix[topic_idx][i] for i in top_indices]
                heatmap_data.append(weights)
            
            # Remove duplicates while preserving order
            unique_words = []
            seen = set()
            for word in top_words_per_topic:
                if word not in seen:
                    unique_words.append(word)
                    seen.add(word)
            
            # Create heatmap data matrix
            heatmap_matrix = np.zeros((self.num_topics, len(unique_words)))
            word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
            
            for topic_idx in range(self.num_topics):
                top_indices = self.topic_term_matrix[topic_idx].argsort()[-num_words:][::-1]
                for word_idx in top_indices:
                    word = self.feature_names[word_idx]
                    if word in word_to_idx:
                        heatmap_matrix[topic_idx, word_to_idx[word]] = \
                            self.topic_term_matrix[topic_idx][word_idx]
            
            # Create heatmap
            plt.figure(figsize=figsize)
            sns.heatmap(
                heatmap_matrix,
                xticklabels=unique_words,
                yticklabels=[f"Topic {i}" for i in range(self.num_topics)],
                cmap="YlOrRd",
                cbar_kws={'label': 'Weight'},
                linewidths=0.5
            )
            
            plt.title("Topic-Term Heatmap", fontsize=16, fontweight='bold')
            plt.xlabel("Top Terms", fontsize=12)
            plt.ylabel("Topics", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Heatmap saved to: {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Heatmap visualization failed: {e}")
            raise AppException(f"Visualization failed: {str(e)}")
    
    def save_model(self, path: str) -> None:
        """
        Saves the trained NMF model and associated data to a file using joblib.
        
        Args:
            path: Path to save the model.
            
        Raises:
            AppException: On error or if model not trained.
        """
        try:
            if self.model is None:
                raise AppException("NMF model is not trained and cannot be saved.")
            
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
                'l1_ratio': self.l1_ratio
            }
            
            joblib.dump(model_data, path)
            self.logger.info(f"NMF model saved to: {path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save NMF model: {e}")
            raise AppException(f"Model save failed: {str(e)}")
    
    def load_model(self, path: str) -> None:
        """
        Loads an NMF model from the given path using joblib.
        
        Args:
            path: Path to load the model from.
            
        Raises:
            AppException: On error loading the model.
        """
        try:
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
            
            self.logger.info(f"NMF model loaded from: {path}")
            self.logger.info(f"Model configuration: {self.num_topics} topics")
            
        except Exception as e:
            self.logger.error(f"Failed to load NMF model: {e}")
            raise AppException(f"Model load failed: {str(e)}")


# if __name__ == "__main__":
#     # Example usage demonstration
#     print("NMF Modeler - Production-Grade Topic Modeling")
#     print("=" * 50)
    
#     # Initialize the modeler
#     nmf_modeler = NMFModeler(
#         num_topics=5,
#         init='nndsvda',
#         max_iter=500,
#         random_state=42
#     )
    
#     print(f"\nNMF Modeler initialized with {nmf_modeler.num_topics} topics")
#     print("Ready for training with TF-IDF matrix")
#     print("\nUsage:")
#     print("  1. nmf_modeler.train(tfidf_matrix, feature_names)")
#     print("  2. topics = nmf_modeler.get_topics(num_words=10)")
#     print("  3. doc_topics = nmf_modeler.get_document_topics()")
#     print("  4. coherence = nmf_modeler.compute_coherence(texts)")
#     print("  5. nmf_modeler.visualize_topic_heatmap()")
    # print("  6. nmf_modeler.save_model('nmf_model.pkl')")