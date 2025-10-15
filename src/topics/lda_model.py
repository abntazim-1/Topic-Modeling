import logging
from typing import Any, List, Optional, Dict, Tuple, Union
from gensim.models import LdaModel, CoherenceModel
from gensim.corpora import Dictionary
from src.utils.logger import get_logger
from src.utils.exceptions import DataValidationError, AppException

class LDAModeler:
    """
    LDA Topic Modeling wrapper using gensim.models.LdaModel. Provides training,
    evaluation, topic extraction, and persistence utilities with robust logging.
    """
    def __init__(
        self,
        num_topics: int = 10,
        passes: int = 10,
        chunksize: int = 100,
        alpha: str = 'auto',
        random_state: int = 42
    ) -> None:
        """
        Initializes the LDA Modeler.
        Args:
            num_topics: Number of topics for LDA.
            passes: Number of passes over corpus during training.
            chunksize: Number of documents to use in each training chunk.
            alpha: Document-topic density prior.
            random_state: Random state for reproducibility.
        """
        self.num_topics = num_topics
        self.passes = passes
        self.chunksize = chunksize
        self.alpha = alpha
        self.random_state = random_state
        self.model: Optional[LdaModel] = None
        self.logger = get_logger(__name__)
        self.logger.info(f"LDAModeler initialized with num_topics={num_topics}, passes={passes}, chunksize={chunksize}, alpha={alpha}.")

    def train(self, corpus: List[List[tuple]], id2word: Dictionary) -> None:
        """
        Fits the LDA model to the provided corpus.
        Args:
            corpus: Gensim corpus (list of bag-of-words).
            id2word: Gensim dictionary object.
        Raises:
            DataValidationError: On invalid corpus/dictionary.
        """
        try:
            if not corpus or not isinstance(corpus, list):
                raise DataValidationError("Invalid or empty corpus provided.")
            if not isinstance(id2word, Dictionary):
                raise DataValidationError("id2word must be a gensim Dictionary object.")
            self.logger.info("Starting LDA model training...")
            self.model = LdaModel(
                corpus=corpus,
                id2word=id2word,
                num_topics=self.num_topics,
                passes=self.passes,
                chunksize=self.chunksize,
                alpha=self.alpha,
                random_state=self.random_state
            )
            self.logger.info(f"LDA model training complete. Number of topics: {self.model.num_topics}")
        except Exception as e:
            self.logger.error(f"Error during LDA model training: {e}")
            raise AppException(str(e))

    def get_topics(self, num_words: int = 10) -> List[Dict[str, Union[int, List[Tuple[str, float]]]]]:
        """
        Returns the top words for each topic.
        Args:
            num_words: Number of top words per topic.
        Returns:
            List of topics, each as dict of topic_id and (word, weight) tuples.
        Raises:
            AppException: If model is not trained.
        """
        try:
            if self.model is None:
                raise AppException("LDA model is not trained yet.")
            topics = []
            for idx, topic in self.model.show_topics(num_topics=self.num_topics, num_words=num_words, formatted=False):
                topics.append({
                    "topic_id": idx,
                    "words": [(word, round(weight, 4)) for word, weight in topic]
                })
            self.logger.info(f"Extracted top {num_words} words for each topic.")
            return topics
        except Exception as e:
            self.logger.error(f"Failed to get topic words: {e}")
            raise AppException(str(e))

    def compute_coherence_score(self, texts: List[List[str]], id2word: Dictionary, coherence: str = 'c_v') -> float:
        """
        Compute the coherence score for trained topics.
        Args:
            texts: Tokenized texts for coherence computation.
            id2word: Gensim dictionary object.
            coherence: Coherence type (default: 'c_v').
        Returns:
            Coherence score (float).
        Raises:
            AppException: On error or if model not trained.
        """
        try:
            if self.model is None:
                raise AppException("LDA model is not trained.")
            coherence_model = CoherenceModel(
                model=self.model,
                texts=texts,
                dictionary=id2word,
                coherence=coherence
            )
            score = coherence_model.get_coherence()
            self.logger.info(f"Coherence ({coherence}) score: {score:.4f}")
            return score
        except Exception as e:
            self.logger.error(f"Failed to compute coherence score: {e}")
            raise AppException(str(e))

    def compute_perplexity(self, corpus: List[List[tuple]]) -> float:
        """
        Compute model perplexity on the given corpus.
        Args:
            corpus: Corpus to compute perplexity against.
        Returns:
            Perplexity value (float).
        Raises:
            AppException: On error or if model not trained.
        """
        try:
            if self.model is None:
                raise AppException("LDA model is not trained.")
            perplexity = self.model.log_perplexity(corpus)
            self.logger.info(f"Model log perplexity: {perplexity:.4f}")
            return perplexity
        except Exception as e:
            self.logger.error(f"Failed to compute perplexity: {e}")
            raise AppException(str(e))

    def get_dominant_topic_per_doc(self, corpus: List[List[tuple]]) -> List[int]:
        """
        Returns the most likely topic for each document in the corpus.
        Args:
            corpus: Corpus in BOW format.
        Returns:
            List of dominant topic indices per document.
        Raises:
            AppException: On error or if model not trained.
        """
        try:
            if self.model is None:
                raise AppException("LDA model is not trained.")
            dominant_topics = []
            for doc_bow in corpus:
                topic_probs = self.model.get_document_topics(doc_bow)
                if topic_probs:
                    dominant_topic = max(topic_probs, key=lambda tup: tup[1])[0]
                else:
                    dominant_topic = -1
                dominant_topics.append(dominant_topic)
            self.logger.info("Extracted dominant topic for each document.")
            return dominant_topics
        except Exception as e:
            self.logger.error(f"Failed to extract dominant topics: {e}")
            raise AppException(str(e))

    def save_model(self, path: str) -> None:
        """
        Saves the trained model to a file.
        Args:
            path: Path to save model.
        Raises:
            AppException: On error or if model not trained.
        """
        try:
            if self.model is None:
                raise AppException("LDA model is not trained and cannot be saved.")
            self.model.save(path)
            self.logger.info(f"LDA model saved to: {path}")
        except Exception as e:
            self.logger.error(f"Failed to save LDA model: {e}")
            raise AppException(str(e))

    def load_model(self, path: str) -> None:
        """
        Loads a model from the given path.
        Args:
            path: Path to load model from.
        Raises:
            AppException: On error loading the model.
        """
        try:
            self.model = LdaModel.load(path)
            self.logger.info(f"LDA model loaded from: {path}")
        except Exception as e:
            self.logger.error(f"Failed to load LDA model: {e}")
            raise AppException(str(e))

if __name__ == "__main__":
    # Example: initialize and save LDA config (without training)
    lda = LDAModeler(num_topics=10, passes=10, chunksize=100)
    # You could print something, or implement custom CLI logic here
    print("LDA modeler initialized with config. Not trained.")
    # lda.save_model("lda_untrained.model")  # THIS will fail unless the model is trained!