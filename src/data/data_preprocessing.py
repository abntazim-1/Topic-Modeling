import os
import re
import html
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
import pandas as pd
import spacy
from tqdm import tqdm
from gensim.models.phrases import Phrases, Phraser
from src.utils.logger import get_logger
from src.utils.exceptions import AppException


def merge_stopwords(custom: List[str], nlp) -> set:
    s = set(nlp.Defaults.stop_words).union({w.lower() for w in custom})
    return s

@dataclass
class PreprocessingConfig:
    """
    Configuration for text preprocessing.
    """
    text_column: str = "content"
    stopwords: List[str] = field(default_factory=lambda: ['said', 'mr', 'say', 'also', 'would', 'one', 'two', 'us'])
    spacy_model: str = "en_core_web_lg"
    output_dir: str = "artifacts/"
    min_token_length: int = 2
    gen_bigrams: bool = True
    gen_trigrams: bool = True
    logging_level: str = "INFO"
    output_file: str = "preprocessed_bbc_news.csv"

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "PreprocessingConfig":
        return cls(**cfg)

class TextCleaner:
    """
    Text normalization utilities (lowercase, punctuation, HTML, contractions, etc.)
    """
    def __init__(self, logger=None):
        self.logger = logger or get_logger(__name__)
        self.CONTRACTIONS = {"it's": "it is", "can't": "cannot", "'re": " are", "'s": " is", "'d": " would", "'ll": " will", "'t": " not", "'ve": " have", "'m": " am"}
    
    def clean(self, text: str) -> str:
        try:
            text = html.unescape(text)
            text = re.sub(r'<.*?>', ' ', text)
            text = text.lower()
            for k,v in self.CONTRACTIONS.items():
                text = re.sub(k, v, text)
            text = re.sub(r'[^a-z\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        except Exception as exc:
            self.logger.error(f"Text cleaning failed: {exc}")
            raise AppException(f"Text cleaning failed: {exc}")

class TextTokenizer:
    """
    Tokenization, stopword removal, lemmatization, and n-gram handling.
    """
    def __init__(self, nlp, stopwords: set, min_token_length: int = 2, logger=None):
        self.nlp = nlp
        self.stopwords = stopwords
        self.min_token_length = min_token_length
        self.logger = logger or get_logger(__name__)

    def doc_to_tokens(self, doc) -> List[str]:
        return [
            t.lemma_.lower() for t in doc
            if not t.is_punct and not t.is_space and t.lemma_ != "-PRON-" \
                and len(t) >= self.min_token_length and t.text not in self.stopwords
        ]

    def process_text(self, text: str) -> List[str]:
        try:
            doc = self.nlp(text)
            return self.doc_to_tokens(doc)
        except Exception as exc:
            self.logger.error(f"Tokenization failed: {exc}")
            raise AppException(f"Tokenization failed: {exc}")

    def process_batch(self, texts: List[str]) -> List[List[str]]:
        results = []
        for doc in tqdm(self.nlp.pipe(texts, disable=["parser", "ner"]), total=len(texts), desc="Tokenizing"):
            results.append(self.doc_to_tokens(doc))
        return results

    @staticmethod
    def build_ngrams(docs: List[List[str]], make_bigrams=True, make_trigrams=True, logger=None) -> List[List[str]]:
        logger = logger or get_logger(__name__)
        try:
            bigram = Phrases(docs, min_count=5, threshold=100)
            bigram_mod = Phraser(bigram) if make_bigrams else None
            if make_bigrams:
                docs = [bigram_mod[doc] for doc in docs]
            if make_trigrams:
                trigram = Phrases(docs, min_count=5, threshold=100)
                trigram_mod = Phraser(trigram)
                docs = [trigram_mod[doc] for doc in docs]
            return docs
        except Exception as exc:
            logger.error(f"N-gram generation failed: {exc}")
            raise AppException(f"N-gram generation failed: {exc}")

class PreprocessingPipeline:
    """
    Orchestrates full text preprocessing: cleaning, tokenization, n-grams. Accepts DataFrame or list.
    """
    def __init__(self, config: PreprocessingConfig, logger=None):
        self.config = config
        self.logger = logger or get_logger(__name__)
        try:
            self.nlp = spacy.load(self.config.spacy_model, disable=["parser", "ner"])
        except OSError:
            self.logger.error(f"spaCy model not found: {self.config.spacy_model}. Please run: python -m spacy download {self.config.spacy_model}")
            raise AppException(f"spaCy model not found: {self.config.spacy_model}")
        self.stopwords = merge_stopwords(self.config.stopwords, self.nlp)
        self.cleaner = TextCleaner(self.logger)
        self.tokenizer = TextTokenizer(self.nlp, self.stopwords, self.config.min_token_length, self.logger)

    def run(self, data: Union[pd.DataFrame, List[str]]) -> pd.DataFrame:
        try:
            if isinstance(data, pd.DataFrame):
                texts = data[self.config.text_column].astype(str).tolist()
            elif isinstance(data, list):
                texts = list(map(str, data))
            else:
                raise AppException("Input data must be a pandas DataFrame or list of strings.")

            self.logger.info(f"Preprocessing {len(texts)} texts.")

            cleaned = [self.cleaner.clean(t) for t in tqdm(texts, desc='Cleaning')]
            tokens = self.tokenizer.process_batch(cleaned)
            ngrams = self.tokenizer.build_ngrams(tokens, self.config.gen_bigrams, self.config.gen_trigrams, logger=self.logger)
            joined = [' '.join(doc) for doc in ngrams]

            df_out = pd.DataFrame({
                self.config.text_column: texts,
                "cleaned_text": joined,
                "tokens": ngrams
            })
            self.logger.info("Preprocessing pipeline complete.")
            return df_out
        except Exception as exc:
            self.logger.error(f"Preprocessing pipeline failed: {exc}")
            raise AppException(f"Preprocessing pipeline failed: {exc}")

    def save(self, df: pd.DataFrame) -> str:
        os.makedirs(self.config.output_dir, exist_ok=True)
        out_path = os.path.join(self.config.output_dir, self.config.output_file)
        try:
            df.to_csv(out_path, index=False)
            self.logger.info(f"Preprocessed data saved to {out_path}")
            return out_path
        except Exception as exc:
            self.logger.error(f"Failed to save results: {exc}")
            raise AppException(f"Failed to save results: {exc}")

class TextVisualizer:
    """
    Utility for word frequency, word cloud, and top-n tokens visualization (optional class, extensible for more viz later).
    """
    def __init__(self, logger=None):
        self.logger = logger or get_logger(__name__)
        try:
            import matplotlib.pyplot as plt
            from wordcloud import WordCloud
            self.plt = plt
            self.WordCloud = WordCloud
        except ImportError:
            self.logger.warning("matplotlib or wordcloud not installed; TextVisualizer limited.")
            self.plt = None
            self.WordCloud = None

    def plot_word_freq(self, docs: List[List[str]], top_n=25):
        from collections import Counter
        if not self.plt:
            self.logger.warning('No plotting backend available.')
            return
        all_tokens = [token for doc in docs for token in doc]
        freq = Counter(all_tokens).most_common(top_n)
        tokens, counts = zip(*freq)
        self.plt.figure(figsize=(12,4))
        self.plt.bar(tokens, counts)
        self.plt.title(f"Top {top_n} Word Frequencies")
        self.plt.xticks(rotation=60)
        self.plt.tight_layout()
        self.plt.show()

    def word_cloud(self, docs: List[List[str]]):
        if not self.WordCloud:
            self.logger.warning('No WordCloud backend available.')
            return
        all_tokens = [token for doc in docs for token in doc]
        text = ' '.join(all_tokens)
        wc = self.WordCloud(width=800, height=300, background_color='white').generate(text)
        self.plt.figure(figsize=(15, 5))
        self.plt.imshow(wc, interpolation='bilinear')
        self.plt.axis('off')
        self.plt.show()

if __name__ == "__main__":
    import sys
    import yaml

    # Example: Load config from YAML
    sample_config_path = "config/config.yaml"
    if not os.path.isfile(sample_config_path):
        print(f"Sample config not found at {sample_config_path}. Create one as shown in docs.")
        sys.exit(1)
    with open(sample_config_path, 'r') as f:
        config_dict = yaml.safe_load(f).get("preprocessing", {})
    config = PreprocessingConfig.from_dict(config_dict)

    logger = get_logger("data_preprocessing")

    csv_in = r"artifacts/bbc-news-data.csv"
    if not os.path.isfile(csv_in):
        logger.error(f"Input data file not found at {csv_in}.")
        sys.exit(1)
    df = pd.read_csv(csv_in, sep='\t')
    pipeline = PreprocessingPipeline(config, logger)
    try:
        result_df = pipeline.run(df)
        out_path = pipeline.save(result_df)
        logger.info("Preprocessing complete.")
    except AppException as exc:
        logger.error(f"Preprocessing failed: {exc}")
        sys.exit(1)
