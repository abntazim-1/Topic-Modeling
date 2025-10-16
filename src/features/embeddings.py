# features/embeddings.py

import logging
from typing import List, Optional
import os
import numpy as np

# Backend imports
try:
    import spacy
except ImportError:
    spacy = None

try:
    import gensim.downloader as api
    from gensim.models import KeyedVectors
except ImportError:
    api = None
    KeyedVectors = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class EmbeddingGenerator:
    """
    A flexible embedding generator class for converting text documents into dense vector embeddings.
    
    Supports spaCy, GloVe (via gensim), and Sentence Transformers.
    """

    def __init__(self, backend: str = "spacy", model_name: Optional[str] = None):
        """
        Initialize and load the embedding model.

        Parameters
        ----------
        backend : str
            The embedding backend to use: "spacy", "glove", "sentence-transformer"
        model_name : str, optional
            Name of the model to load (e.g., "en_core_web_md" for spaCy, "glove-wiki-gigaword-100" for GloVe)
        """
        self.backend = backend.lower()
        self.model_name = model_name
        self.model = None

        try:
            if self.backend == "spacy":
                if spacy is None:
                    raise ImportError("spaCy is not installed. Install with `pip install spacy`.")
                if not self.model_name:
                    self.model_name = "en_core_web_md"
                logger.info(f"Loading spaCy model '{self.model_name}'...")
                self.model = spacy.load(self.model_name)
                logger.info("spaCy model loaded successfully.")

            elif self.backend == "glove":
                if api is None or KeyedVectors is None:
                    raise ImportError("gensim is not installed. Install with `pip install gensim`.")
                if not self.model_name:
                    self.model_name = "glove-wiki-gigaword-100"
                logger.info(f"Loading GloVe model '{self.model_name}' from gensim-data...")
                self.model = api.load(self.model_name)
                logger.info("GloVe model loaded successfully.")

            elif self.backend == "sentence-transformer":
                if SentenceTransformer is None:
                    raise ImportError("sentence-transformers is not installed. Install with `pip install sentence-transformers`.")
                if not self.model_name:
                    self.model_name = "all-MiniLM-L6-v2"
                logger.info(f"Loading Sentence Transformer model '{self.model_name}'...")
                self.model = SentenceTransformer(self.model_name)
                logger.info("Sentence Transformer model loaded successfully.")

            else:
                raise ValueError(f"Unsupported backend '{self.backend}'. Choose 'spacy', 'glove', or 'sentence-transformer'.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def get_vector(self, text: str) -> np.ndarray:
        """
        Generate a single dense embedding for a document.

        Parameters
        ----------
        text : str
            Input text document.

        Returns
        -------
        np.ndarray
            Dense vector representation of the input text.
        """
        if not text.strip():
            return np.zeros(self._vector_size(), dtype=float)

        if self.backend == "spacy":
            doc = self.model(text)
            return doc.vector

        elif self.backend == "glove":
            words = text.split()
            vectors = [self.model[word] for word in words if word in self.model]
            if vectors:
                return np.mean(vectors, axis=0)
            else:
                return np.zeros(self.model.vector_size, dtype=float)

        elif self.backend == "sentence-transformer":
            return self.model.encode([text])[0]

        else:
            raise ValueError(f"Unsupported backend '{self.backend}'.")

    def get_batch_vectors(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of documents.

        Parameters
        ----------
        texts : List[str]
            List of input text documents.

        Returns
        -------
        np.ndarray
            2D array where each row is a document embedding.
        """
        if not texts:
            return np.empty((0, self._vector_size()))

        if self.backend == "sentence-transformer":
            return np.array(self.model.encode(texts))

        vectors = []
        for text in texts:
            vectors.append(self.get_vector(text))
        return np.vstack(vectors)

    def save_embeddings(self, path: str, vectors: np.ndarray):
        """
        Save embeddings to disk as a NumPy file.

        Parameters
        ----------
        path : str
            File path to save embeddings.
        vectors : np.ndarray
            Embeddings to save.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, vectors)
        logger.info(f"Embeddings saved to '{path}'.")

    def _vector_size(self) -> int:
        """
        Get the dimensionality of embeddings based on the backend.

        Returns
        -------
        int
            Dimension of the embedding vectors.
        """
        if self.backend == "spacy":
            return self.model.vocab.vectors_length
        elif self.backend == "glove":
            return self.model.vector_size
        elif self.backend == "sentence-transformer":
            return self.model.get_sentence_embedding_dimension()
        else:
            raise ValueError(f"Unsupported backend '{self.backend}'.")
