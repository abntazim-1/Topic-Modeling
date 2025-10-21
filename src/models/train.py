import os
import sys
import argparse
import joblib
import pandas as pd
from typing import Optional, List

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.utils.logger import get_logger
from src.utils.exceptions import AppException, DataValidationError

# Import vectorizer wrappers
from src.features.tfidf import TFIDFVectorizerWrapper
#from src.features.embeddings import CountVectorizerWrapper  # hypothetical module

# Import model definitions
from src.topics.lda_model import LDA
from src.topics.nmf_model import NMF


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
            # Prefer tokenized column if available; fallback to cleaned_text
            token_col = None
            if 'tokens' in df.columns:
                token_col = 'tokens'
            elif 'cleaned_tokens' in df.columns:
                token_col = 'cleaned_tokens'

            if token_col:
                import ast
                try:
                    tokens_series = df[token_col]
                    if tokens_series.dtype == object and isinstance(tokens_series.iloc[0], str):
                        tokens_series = tokens_series.apply(ast.literal_eval)
                except Exception:
                    raise DataValidationError(f"Column '{token_col}' must contain lists or stringified lists of tokens.")
                self.texts = tokens_series.tolist()
            else:
                if 'cleaned_text' not in df.columns:
                    raise DataValidationError("Missing required 'tokens' or 'cleaned_text' column in dataset.")
                # Use cleaned_text as space-joined tokens
                self.texts = df['cleaned_text'].astype(str).apply(lambda s: s.split()).tolist()
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
                    ngram_range=(1, 1), min_df=3, max_df=0.9, max_features=30000
                )
            elif self.vectorizer_type == "bow":
                # Fallback: approximate BoW with TF-IDF configured as unigram, higher max_features
                self.vectorizer = TFIDFVectorizerWrapper(
                    ngram_range=(1, 1), min_df=2, max_df=0.95, max_features=30000
                )
            else:
                raise ValueError(f"Unsupported vectorizer type: {self.vectorizer_type}")

            # Store the matrix and feature names for both LDA and NMF
            # If texts are token lists, join to strings for TF-IDF
            joined = [" ".join(t) if isinstance(t, list) else str(t) for t in self.texts]
            self.matrix, self.feature_names = self.vectorizer.fit_transform(joined)
            
            # For LDA, convert to gensim corpus format
            if self.model_type == "lda":
                from gensim.corpora import Dictionary

                # Build dictionary from tokenized texts (better mapping than feature_names)
                if all(isinstance(x, list) for x in self.texts):
                    self.id2word = Dictionary(self.texts)
                    self.corpus = [self.id2word.doc2bow(tokens) for tokens in self.texts]
                else:
                    # Fallback: split cleaned strings
                    tokenized = [str(s).split() for s in self.texts]
                    self.id2word = Dictionary(tokenized)
                    self.corpus = [self.id2word.doc2bow(tokens) for tokens in tokenized]
                
            self.logger.info(f"Vectorization complete using {self.vectorizer_type.upper()} vectorizer.")
        except Exception as e:
            self.logger.error(f"Feature preparation failed: {e}")
            raise AppException(str(e))

    # ----------------------------------------------------------------------
    def train_model(self) -> None:
        """Trains either LDA or NMF model based on configuration."""
        try:
            if self.model_type == "lda":
                # Create LDA model with strong hyperparameters
                self.model = LDA(
                    num_topics=self.num_topics,
                    passes=20,
                    chunksize=2000,
                    iterations=800,
                    update_every=1,
                    alpha='auto',
                    eta='auto',
                    random_state=self.random_state
                )
                
                # Train the model (gensim LdaModel is trained during construction)
                lda_model = self.model.create_model(corpus=self.corpus, id2word=self.id2word)
                self.model.set_model(lda_model)
                
                self.logger.info(f"LDA model trained with {self.model.num_topics} topics")

            elif self.model_type == "nmf":
                # Create NMF model with strong hyperparameters
                self.model = NMF(
                    num_topics=self.num_topics,
                    init='nndsvda',
                    max_iter=600,
                    alpha=0.1,
                    l1_ratio=0.5,
                    random_state=self.random_state
                )
                
                # Train the model
                nmf_model = self.model.create_model()
                document_topic_matrix = nmf_model.fit_transform(self.matrix)
                topic_term_matrix = nmf_model.components_
                
                self.model.set_model(nmf_model)
                self.model.set_feature_names(self.feature_names)
                self.model.set_matrices(document_topic_matrix, topic_term_matrix)
                
                self.logger.info(f"NMF model trained with {self.model.num_topics} topics")

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
            self.model.save_model(model_path)

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
            top_topics = self.model.get_topics(num_words=8)
            self.logger.info("Top topics and keywords:")
            for topic in top_topics:
                self.logger.info(f"Topic {topic['topic_id']}: {[w for w, _ in topic['words']]}")
                
            if self.model_type == "lda":
                perplexity = self.model.compute_perplexity(self.corpus)
                self.logger.info(f"Perplexity: {perplexity:.4f}")
            elif self.model_type == "nmf":
                reconstruction_error = self.model.get_reconstruction_error()
                self.logger.info(f"Reconstruction Error: {reconstruction_error:.4f}")
                
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
