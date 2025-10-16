import os
import argparse
import joblib
import pandas as pd
from typing import Optional, List

from src.utils.logger import get_logger
from src.utils.exceptions import AppException, DataValidationError

# Import vectorizer wrappers
from src.features.tfidf import TFIDFVectorizerWrapper
#from src.features.embeddings import CountVectorizerWrapper  # hypothetical module

# Import modelers
from src.topics.lda_model import LDAModeler  # your provided class
from src.topics.nmf_model import NMFModeler  # assumed similar structure to LDA


class TopicModelTrainer:
    """
    Central training entry point for Topic Modeling (LDA / NMF).
    Handles data ingestion, feature preparation, model training, and artifact persistence.
    """

    def __init__(
        self,
        model_type: str = "lda",
        num_topics: int = 10,
        data_path: str = r"artifacts/preprocessed_bbc_news.csv",
        vectorizer_type: str = "tfidf",
        output_dir: str = "artifacts/",
        random_state: int = 42
    ):
        self.model_type = model_type.lower()
        self.num_topics = num_topics
        self.data_path = data_path
        self.vectorizer_type = vectorizer_type.lower()
        self.output_dir = output_dir
        self.random_state = random_state

        self.logger = get_logger(__name__)
        self.model = None
        self.vectorizer = None
        self.corpus = None
        self.id2word = None
        self.texts = None

        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"Initialized TopicModelTrainer for {self.model_type.upper()} with {self.num_topics} topics.")

    # ----------------------------------------------------------------------
    def load_data(self) -> None:
        """Loads preprocessed text data."""
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"File not found: {self.data_path}")

            df = pd.read_csv(self.data_path)
            if 'cleaned_text' not in df.columns:
                raise DataValidationError("Missing required 'cleaned_text' column in dataset.")
            
            # Use cleaned_text column for better topic modeling with valuable words
            self.texts = df['cleaned_text'].astype(str).tolist()
            self.logger.info(f"Loaded {len(self.texts)} documents from {self.data_path}.")
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise AppException(str(e))

    # ----------------------------------------------------------------------
    def prepare_features(self) -> None:
        """Initializes and fits the chosen vectorizer."""
        try:
            if not self.texts:
                raise DataValidationError("No texts available for vectorization.")

            if self.vectorizer_type == "tfidf":
                self.vectorizer = TFIDFVectorizerWrapper(
                    ngram_range=(1, 2), min_df=2, max_df=0.95, max_features=10000
                )
            elif self.vectorizer_type == "bow":
                self.vectorizer = CountVectorizerWrapper(min_df=2, max_df=0.95, ngram_range=(1, 2))
            else:
                raise ValueError(f"Unsupported vectorizer type: {self.vectorizer_type}")

            matrix, self.id2word = self.vectorizer.fit_transform(self.texts)
            
            # Convert sparse matrix to gensim corpus format (list of list of tuples)
            from scipy import sparse
            import numpy as np
            from gensim.corpora import Dictionary
            
            # Convert id2word dict to gensim Dictionary using a safer approach
            # Create a gensim Dictionary directly from feature names
            from gensim.corpora import Dictionary
            feature_names = self.vectorizer.get_feature_names()
            gensim_dict = Dictionary([feature_names])
            self.id2word = gensim_dict
            
            # Convert sparse matrix to gensim corpus format
            self.corpus = []
            for i in range(matrix.shape[0]):
                # Get non-zero elements in this document
                row = matrix[i].tocoo()
                # Add as (term_id, term_weight) tuples
                doc = [(int(term_id), float(weight)) for term_id, weight in zip(row.col, row.data)]
                self.corpus.append(doc)
                
            self.logger.info(f"Vectorization complete using {self.vectorizer_type.upper()} vectorizer.")
        except Exception as e:
            self.logger.error(f"Feature preparation failed: {e}")
            raise AppException(str(e))

    # ----------------------------------------------------------------------
    def train_model(self) -> None:
        """Trains either LDA or NMF model based on configuration."""
        try:
            if self.model_type == "lda":
                self.model = LDAModeler(num_topics=self.num_topics, random_state=self.random_state)
                self.model.train(self.corpus, self.id2word)

            elif self.model_type == "nmf":
                self.model = NMFModeler(num_topics=self.num_topics, random_state=self.random_state)
                # NMF needs the TF-IDF matrix and feature names
                from scipy import sparse
                import numpy as np
                
                # Convert corpus back to sparse matrix format for NMF
                rows = []
                cols = []
                data = []
                for doc_idx, doc in enumerate(self.corpus):
                    for term_id, weight in doc:
                        rows.append(doc_idx)
                        cols.append(term_id)
                        data.append(weight)
                
                # Create sparse matrix
                num_docs = len(self.corpus)
                num_terms = len(self.id2word)
                tfidf_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(num_docs, num_terms))
                
                # Get feature names from id2word
                feature_names = [self.id2word[i] for i in range(len(self.id2word))]
                
                # Train NMF model
                self.model.train(tfidf_matrix, feature_names)

            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            self.logger.info(f"{self.model_type.upper()} model training completed successfully.")
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            raise AppException(str(e))

    # ----------------------------------------------------------------------
    def save_artifacts(self) -> None:
        """Saves trained model, vectorizer, and topic outputs."""
        try:
            model_path = os.path.join(self.output_dir, f"{self.model_type}_model.pkl")
            vectorizer_path = os.path.join(self.output_dir, f"{self.vectorizer_type}_vectorizer.pkl")

            # Save model
            if self.model_type == "lda":
                self.model.save_model(model_path)
            else:
                joblib.dump(self.model, model_path)

            # Save vectorizer
            joblib.dump(self.vectorizer, vectorizer_path)
            self.logger.info(f"Saved model and vectorizer artifacts to {self.output_dir}")

            # Save topics summary
            topics = self.model.get_topics(num_words=10)
            topics_path = os.path.join(self.output_dir, f"{self.model_type}_topics.csv")
            pd.DataFrame(topics).to_csv(topics_path, index=False)
            self.logger.info(f"Saved topic summary to {topics_path}")
        except Exception as e:
            self.logger.error(f"Error saving artifacts: {e}")
            raise AppException(str(e))

    # ----------------------------------------------------------------------
    def log_training_summary(self) -> None:
        """Logs key training metrics and sample topics."""
        try:
            coherence = self.model.compute_coherence_score(self.texts, self.id2word)
            self.logger.info(f"Coherence Score (c_v): {coherence:.4f}")

            if self.model_type == "lda":
                perplexity = self.model.compute_perplexity(self.corpus)
                self.logger.info(f"Perplexity: {perplexity:.4f}")

            top_topics = self.model.get_topics(num_words=8)
            self.logger.info("Top topics and keywords:")
            for topic in top_topics:
                self.logger.info(f"Topic {topic['topic_id']}: {[w for w, _ in topic['words']]}")
        except Exception as e:
            self.logger.error(f"Failed to log training summary: {e}")
            raise AppException(str(e))


# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Topic Model (LDA / NMF)")
    parser.add_argument("--model_type", type=str, default="lda", help="Model type: lda or nmf")
    parser.add_argument("--num_topics", type=int, default=10, help="Number of topics")
    parser.add_argument("--data_path", type=str, default="artifacts\preprocessed_bbc_news.csv", help="Path to preprocessed data")
    parser.add_argument("--vectorizer_type", type=str, default="tfidf", help="Vectorizer: tfidf or bow")
    parser.add_argument("--output_dir", type=str, default="artifacts/", help="Directory to save model artifacts")
    args = parser.parse_args()

    trainer = TopicModelTrainer(
        model_type=args.model_type,
        num_topics=args.num_topics,
        data_path=args.data_path,
        vectorizer_type=args.vectorizer_type,
        output_dir=args.output_dir
    )

    trainer.load_data()
    trainer.prepare_features()
    trainer.train_model()
    trainer.save_artifacts()
    trainer.log_training_summary()
