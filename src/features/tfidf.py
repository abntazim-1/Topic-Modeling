import logging
from typing import List, Optional, Tuple, Union, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import NotFittedError
import joblib
from src.utils.logger import get_logger
from src.utils.exceptions import AppException
import pandas as pd
import os


class TFIDFVectorizerWrapper:
    """
    Wrapper for scikit-learn's TfidfVectorizer with logging and configuration.
    Provides easy integration with text mining/topic modeling pipelines.
    """
    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: Union[int, float] = 0.95,
        max_features: Optional[int] = 10000,
        stop_words: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs: Any
    ):
        """
        Initialize the TF-IDF vectorizer wrapper.

        Args:
            ngram_range: The lower and upper boundary of the n-grams (default (1,2)).
            min_df: Minimum document frequency for terms (default 2).
            max_df: Max document frequency (int or proportion) (default 0.95).
            max_features: Max number of features (default 10000).
            stop_words: List of stopwords (optional, overrides TfidfVectorizer's built-ins).
            logger: Optional logger instance.
            **kwargs: Other parameters for scikit-learn TfidfVectorizer.
        """
        self.logger = logger or get_logger(__name__)
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            stop_words=stop_words,
            tokenizer=lambda x: x if isinstance(x, list) else x.split(),
            preprocessor=lambda x: x,  # Assume already preprocessed
            token_pattern=None,
            **kwargs
        )
        self.fitted = False
        self.logger.info(f"Initialized TFIDFVectorizerWrapper with params: ngram_range={ngram_range}, min_df={min_df}, max_df={max_df}, max_features={max_features}")

    def fit(self, texts: List[Union[str, List[str]]]) -> 'TFIDFVectorizerWrapper':
        """
        Fit the vectorizer to a corpus of texts.
        Args:
            texts: List of cleaned strings, or List of token lists.
        Returns:
            self
        Raises:
            AppException: If input is empty or invalid.
        """
        if not texts or not isinstance(texts, list):
            self.logger.error("Input for fit is empty or not a list.")
            raise AppException("TFIDFVectorizerWrapper.fit: Input text list is empty or not a list.")
        self.vectorizer.fit(texts)
        self.fitted = True
        self.logger.info(f"TF-IDF vectorizer fitted on {len(texts)} documents. Vocabulary size: {len(self.vectorizer.vocabulary_)}.")
        return self

    def transform(self, texts: List[Union[str, List[str]]]):
        """
        Transform new texts to the vectorized space.
        Args:
            texts: List of cleaned strings, or List of token lists.
        Returns:
            Sparse TF-IDF matrix
        Raises:
            AppException: If vectorizer not fitted or input invalid.
        """
        if not texts or not isinstance(texts, list):
            self.logger.error("Input for transform is empty or not a list.")
            raise AppException("TFIDFVectorizerWrapper.transform: Input text list is empty or not a list.")
        try:
            matrix = self.vectorizer.transform(texts)
            self.logger.info(f"Transformed {len(texts)} documents. Resulting shape: {matrix.shape}.")
            return matrix
        except NotFittedError as e:
            self.logger.error("TF-IDF vectorizer not fitted.")
            raise AppException("TFIDFVectorizerWrapper: Vectorizer must be fitted before use.") from e

    def fit_transform(self, texts: List[Union[str, List[str]]]):
        """
        Fit and transform the texts.
        Args:
            texts: List of cleaned strings, or List of token lists.
        Returns:
            Sparse TF-IDF matrix
        Raises:
            AppException: If input invalid.
        """
        if not texts or not isinstance(texts, list):
            self.logger.error("Input for fit_transform is empty or not a list.")
            raise AppException("TFIDFVectorizerWrapper.fit_transform: Input text list is empty or not a list.")
        matrix = self.vectorizer.fit_transform(texts)
        self.fitted = True
        self.logger.info(f"TF-IDF vectorizer fit and transformed {len(texts)} documents. Shape: {matrix.shape}.")
        return matrix

    def get_feature_names(self) -> List[str]:
        """
        Get the feature (vocabulary) names.
        Returns:
            List of feature names.
        Raises:
            AppException: If vectorizer not yet fitted.
        """
        if not self.fitted:
            self.logger.error("Tried to get feature names from an unfitted vectorizer.")
            raise AppException("TFIDFVectorizerWrapper: Vectorizer must be fitted before getting feature names.")
        return self.vectorizer.get_feature_names_out().tolist()

    def save_vectorizer(self, path: str) -> None:
        """
        Save the vectorizer to disk using joblib.
        Args:
            path: Path to save the model.
        """
        joblib.dump(self.vectorizer, path)
        self.logger.info(f"TFIDFVectorizer saved to {path}.")

    def load_vectorizer(self, path: str) -> None:
        """
        Load a vectorizer from disk using joblib.
        Args:
            path: Path to a saved vectorizer.
        """
        self.vectorizer = joblib.load(path)
        self.fitted = True
        self.logger.info(f"TFIDFVectorizer loaded from {path}.")


if __name__ == "__main__":


# Assume logger, AppException, and TFIDFVectorizerWrapper class already defined above.

# Path to preprocessed data (adapt if needed)
    csv_path = os.path.join("artifacts", "preprocessed_bbc_news.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find preprocessed BBC news CSV at {csv_path}")

    # The file is expected to have a 'tokens' column with lists of tokens
    df = pd.read_csv(csv_path)

    # Parse the stringified token lists, if necessary
    import ast
    if df['tokens'].dtype == object and isinstance(df['tokens'].iloc[0], str):
        try:
            df['tokens'] = df['tokens'].apply(ast.literal_eval)
        except Exception:
            raise ValueError("Could not parse token column as list; ensure 'tokens' column contains stringified Python lists.")

    # Load tokens as a list of lists for TF-IDF
    docs = df['tokens'].tolist()

    # (Optional) convert tokens list back to whitespace-joined strings if required by your TFIDFVectorizerWrapper
    # If your TFIDFVectorizerWrapper expects strings (not tokens), uncomment the following line:
    # docs = [' '.join(tokens) for tokens in docs]

    # Create and apply TFIDF vectorizer
    vectorizer = TFIDFVectorizerWrapper()
    tfidf_matrix = vectorizer.fit_transform(docs)

    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

# Optionally: print sample features or save the matrix/vectorizer
# print("TF-IDF Features:", vectorizer.get_feature_names()[:20])
# vectorizer.save_vectorizer("artifacts/tfidf_vectorizer.joblib")