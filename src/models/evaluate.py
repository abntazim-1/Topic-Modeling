import os
import json
import logging
import warnings
import ast
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
from gensim.models import LdaModel, CoherenceModel
from gensim.corpora import Dictionary
from scipy.sparse import csr_matrix
import pyLDAvis
import pyLDAvis.gensim_models

from src.utils.logger import get_logger, setup_logging
from src.utils.exceptions import DataValidationError, AppException
import pickle
import json
import csv

# ------------------------------------------------------------------
# Global warning suppression and env toggles to minimize noisy output
# ------------------------------------------------------------------
# Reduce TensorFlow / oneDNN noise if indirectly imported by dependencies
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # ERROR only
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # disable oneDNN opts

# Suppress common noisy warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")
warnings.filterwarnings("ignore", module="pyLDAvis")
warnings.filterwarnings("ignore", module="matplotlib")
warnings.filterwarnings("ignore", module="tensorflow")
warnings.filterwarnings("ignore", module="tf_keras")
warnings.filterwarnings("ignore", message="is deprecated. Please use")

try:
    import tensorflow as tf  # may be pulled by other modules
    tf.get_logger().setLevel("ERROR")
except Exception:
    pass

def load_pickle(file_path):
    """Load a pickled object using joblib for better compatibility"""
    return joblib.load(file_path)
        
def save_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
        
def save_json(obj, file_path):
    with open(file_path, 'w') as f:
        json.dump(obj, f, indent=4)
        
def save_csv(data, file_path, headers=None):
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        if headers:
            writer.writerow(headers)
        writer.writerows(data)
from src.topics.lda_model import LDAModeler
from src.topics.nmf_model import NMFModeler
from src.features.tfidf import TFIDFVectorizerWrapper
import ast


class TopicModelEvaluator:
    """
    A unified evaluation framework for trained topic models (LDA or NMF).
    Measures interpretability, diversity, and quality of topics using
    statistical and semantic metrics.
    """
    
    def __init__(
        self,
        model_path: str,
        vectorizer_path: str,
        data_path: str,
        model_type: str,
        id2word: Optional[Dictionary] = None,
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
            id2word: Gensim Dictionary object (required for LDA).
            output_dir: Directory to save evaluation results.
            random_state: Random state for reproducibility.
        """
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.data_path = data_path
        self.model_type = model_type.lower()
        self.id2word = id2word
        self.output_dir = output_dir
        self.random_state = random_state
        
        # Will be populated in load_artifacts
        self.model = None
        self.vectorizer = None
        self.data = None
        self.texts = None
        self.corpus = None
        self.feature_names = None
        self.document_topic_matrix = None
        
        # Evaluation metrics
        self.metrics = {
            "coherence_score": None,
            "perplexity": None,
            "topic_diversity": None,
            "silhouette_score": None,
            "evaluation_time": None
        }
        
        # Set up logger
        self.logger = get_logger(__name__)
        self.logger.info(
            f"TopicModelEvaluator initialized for {model_type} model at {model_path}"
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_artifacts(self) -> None:
        """
        Load the trained model, vectorizer, and data.
        
        Raises:
            AppException: If artifacts cannot be loaded.
        """
        try:
            # Load model
            self.logger.info(f"Loading model from {self.model_path}")
            if self.model_type == 'lda':
                # Load LDA model using gensim to ensure internal state arrays are restored
                self.model = LdaModel.load(self.model_path)
            else:
                # For NMF and other sklearn models, use joblib
                self.model = load_pickle(self.model_path)
            
            # Load vectorizer
            self.logger.info(f"Loading vectorizer from {self.vectorizer_path}")
            self.vectorizer = load_pickle(self.vectorizer_path)
            
            # Load data
            self.logger.info(f"Loading data from {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            
            # Extract texts based on available columns
            if 'cleaned_text' in self.data.columns:
                self.texts = self.data['cleaned_text'].tolist()
            elif 'tokens' in self.data.columns:
                self.texts = self.data['tokens'].tolist()
            else:
                self.texts = self.data['content'].tolist()
                
            # Get feature names from vectorizer
            if hasattr(self.vectorizer, 'get_feature_names_out'):
                self.feature_names = self.vectorizer.get_feature_names_out()
            elif hasattr(self.vectorizer, 'get_feature_names'):
                self.feature_names = self.vectorizer.get_feature_names()
            else:
                self.feature_names = list(self.vectorizer.vocabulary_.keys())
            
            # Prepare corpus based on model type
            if self.model_type == 'lda':
                # Try to load id2word from the model or from a separate file
                if self.id2word is None:
                    if hasattr(self.model, 'id2word') and self.model.id2word is not None:
                        self.id2word = self.model.id2word
                    else:
                        # Try to load id2word from the model file with .id2word extension
                        id2word_path = f"{self.model_path}.id2word"
                        try:
                            self.id2word = Dictionary.load(id2word_path)
                            self.logger.info(f"Loaded id2word dictionary from {id2word_path}")
                        except Exception as e:
                            self.logger.warning(f"Could not load id2word from {id2word_path}: {e}")
                            # Create a simple dictionary from feature names as fallback
                            self.logger.info("Creating dictionary from feature names as fallback")
                            self.id2word = Dictionary([self.feature_names])

                # Build corpus using the gensim dictionary to ensure token ids match the trained model
                # Tokenize texts
                tokenized_texts = []
                for text in self.texts:
                    if isinstance(text, str):
                        tokenized_texts.append(text.split())
                    elif isinstance(text, list):
                        tokenized_texts.append(text)

                self.corpus = [self.id2word.doc2bow(tokens) for tokens in tokenized_texts]

                # Optionally build document-topic matrix (robust to inference errors)
                try:
                    if hasattr(self.model, 'get_document_topics') and self.corpus is not None:
                        self.document_topic_matrix = np.zeros((len(self.corpus), self.model.num_topics))
                        for i, doc in enumerate(self.corpus):
                            topic_probs = self.model.get_document_topics(doc)
                            for topic_id, prob in topic_probs:
                                self.document_topic_matrix[i, topic_id] = prob
                except Exception as e:
                    # If inference fails due to version mismatches, continue without silhouette
                    self.logger.warning(f"Skipping document-topic matrix computation: {e}")
            
            elif self.model_type == 'nmf':
                # For NMF, we need the document-term matrix
                if hasattr(self.vectorizer, 'transform'):
                    dtm = self.vectorizer.transform(self.texts)
                    self.corpus = dtm
                    
                # Get document-topic matrix
                if hasattr(self.model, 'document_topic_matrix'):
                    self.document_topic_matrix = self.model.document_topic_matrix
                elif hasattr(self.model, 'transform'):
                    self.document_topic_matrix = self.model.transform(self.corpus)
            
            self.logger.info("Successfully loaded all artifacts")
            
        except Exception as e:
            self.logger.error(f"Error loading artifacts: {str(e)}")
            raise AppException(f"Failed to load evaluation artifacts: {str(e)}")
    
    def compute_coherence(self, texts: Optional[List[str]] = None) -> float:
        """
        Calculate topic coherence using gensim.models.CoherenceModel.
        
        Args:
            texts: List of tokenized texts. If None, uses self.texts.
            
        Returns:
            Coherence score (c_v measure).
        """
        try:
            if texts is None:
                texts = self.texts
                
            # Tokenize texts if they're not already tokenized
            tokenized_texts = []
            for text in texts:
                if isinstance(text, str):
                    tokenized_texts.append(text.split())
                elif isinstance(text, list):
                    tokenized_texts.append(text)
            
            if self.model_type == 'lda':
                # Get topic-word distributions
                topics = self.model.show_topics(formatted=False, num_words=20)
                topic_words = [[word for word, _ in topic[1]] for topic in topics]
                
                # Calculate coherence (c_v) and fallback to u_mass if needed
                try:
                    # For c_v, build dictionary from the evaluation texts
                    cv_dictionary = Dictionary(tokenized_texts)
                    coherence_model = CoherenceModel(
                        topics=topic_words,
                        texts=tokenized_texts,
                        dictionary=cv_dictionary,
                        coherence='c_v'
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        coherence = coherence_model.get_coherence()
                    coherence_type = 'c_v'
                    if coherence is None or np.isnan(coherence):
                        raise ValueError("c_v coherence returned NaN")
                except Exception as e:
                    self.logger.info(f"c_v coherence failed ({e}), falling back to u_mass")
                    coherence_model = CoherenceModel(
                        model=self.model,
                        corpus=self.corpus,
                        dictionary=self.id2word,
                        coherence='u_mass'
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        coherence = coherence_model.get_coherence()
                    coherence_type = 'u_mass'
                
            elif self.model_type == 'nmf':
                # Get top words for each topic
                topic_words = []
                for topic_idx in range(self.model.components_.shape[0]):
                    top_word_indices = np.argsort(self.model.components_[topic_idx])[::-1][:20]
                    topic_words.append([self.feature_names[i] for i in top_word_indices])
                
                # Create dictionary from tokenized texts
                dictionary = Dictionary(tokenized_texts)
                
                # Calculate coherence
                coherence_model = CoherenceModel(
                    topics=topic_words,
                    texts=tokenized_texts,
                    dictionary=dictionary,
                    coherence='c_v'
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    coherence = coherence_model.get_coherence()
            
            self.metrics["coherence_score"] = coherence
            # Track coherence type in metrics for clarity
            self.metrics["coherence_type"] = coherence_type if 'coherence_type' in locals() else 'c_v'
            self.logger.info(f"Coherence score ({self.metrics['coherence_type']}): {coherence:.4f}")
            return coherence

        except Exception as e:
            self.logger.error(f"Error computing coherence: {str(e)}")
            self.metrics["coherence_score"] = None
            return 0.0
    
    def compute_perplexity(self, corpus: Optional[List[List[tuple]]] = None) -> Optional[float]:
        """
        Calculate perplexity (for LDA only).
        
        Args:
            corpus: Gensim corpus. If None, uses self.corpus.
            
        Returns:
            Perplexity score or None if not applicable.
        """
        if self.model_type != 'lda':
            self.logger.info("Perplexity is only applicable for LDA models")
            return None
            
        try:
            if corpus is None:
                corpus = self.corpus
                
            perplexity = self.model.log_perplexity(corpus)
            self.metrics["perplexity"] = perplexity
            self.logger.info(f"Perplexity: {perplexity:.4f}")
            return perplexity
            
        except Exception as e:
            self.logger.error(f"Error computing perplexity: {str(e)}")
            self.metrics["perplexity"] = None
            return None
    
    def compute_topic_diversity(self, top_n: int = 10) -> float:
        """
        Measure how unique topic words are across all topics.
        
        Args:
            top_n: Number of top words per topic to consider.
            
        Returns:
            Topic diversity score (0-1, higher is better).
        """
        try:
            # Get top words for each topic
            top_words = self.get_top_words_per_topic(n=top_n)
            
            # Flatten the list of top words
            all_top_words = []
            for topic_id, words in top_words.items():
                all_top_words.extend([word for word, _ in words])
            
            # Calculate diversity as ratio of unique words to total words
            unique_words = set(all_top_words)
            diversity = len(unique_words) / len(all_top_words) if all_top_words else 0
            
            self.metrics["topic_diversity"] = diversity
            self.logger.info(f"Topic diversity (top {top_n} words): {diversity:.4f}")
            return diversity
            
        except Exception as e:
            self.logger.error(f"Error computing topic diversity: {str(e)}")
            self.metrics["topic_diversity"] = None
            return 0.0
    
    def compute_silhouette_score(self) -> Optional[float]:
        """
        Calculate silhouette score using topic-document embeddings.
        
        Returns:
            Silhouette score or None if computation fails.
        """
        try:
            if self.document_topic_matrix is None or self.document_topic_matrix.shape[0] < 2:
                self.logger.info("Document-topic matrix not available or too small for silhouette score")
                return None
                
            # Get document-topic distribution
            doc_topic_dist = self.document_topic_matrix
            
            # Get the dominant topic for each document
            dominant_topics = np.argmax(doc_topic_dist, axis=1)
            
            # Calculate silhouette score
            if len(set(dominant_topics)) < 2:
                self.logger.info("All documents assigned to same topic, silhouette score undefined")
                return None
                
            silhouette = silhouette_score(doc_topic_dist, dominant_topics)
            self.metrics["silhouette_score"] = silhouette
            self.logger.info(f"Silhouette score: {silhouette:.4f}")
            return silhouette
            
        except Exception as e:
            self.logger.error(f"Error computing silhouette score: {str(e)}")
            self.metrics["silhouette_score"] = None
            return None
    
    def get_top_words_per_topic(self, n: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """
        Extract and format topic keywords.
        
        Args:
            n: Number of top words per topic.
            
        Returns:
            Dictionary mapping topic IDs to lists of (word, weight) tuples.
        """
        top_words = {}
        
        try:
            if self.model_type == 'lda':
                # Get top words for each topic from LDA model
                for topic_id in range(self.model.num_topics):
                    topic_words = self.model.show_topic(topic_id, topn=n)
                    top_words[topic_id] = topic_words
                    
            elif self.model_type == 'nmf':
                # Get top words for each topic from NMF model
                for topic_id in range(self.model.components_.shape[0]):
                    top_indices = np.argsort(self.model.components_[topic_id])[::-1][:n]
                    topic_words = [(self.feature_names[i], self.model.components_[topic_id, i]) for i in top_indices]
                    top_words[topic_id] = topic_words
            
            return top_words
            
        except Exception as e:
            self.logger.error(f"Error getting top words per topic: {str(e)}")
            return {}
    
    def visualize_topics(self, output_path: Optional[str] = None) -> None:
        """
        Create visualizations for topic model.
        
        Args:
            output_path: Path to save visualizations. If None, uses default.
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"{self.model_type}_visualization")
            
        os.makedirs(output_path, exist_ok=True)
        
        try:
            # 1. Topic-word heatmap
            self._create_topic_word_heatmap(output_path)
            
            # 2. Topic distribution visualization
            self._create_topic_distribution_plot(output_path)
            
            # 3. pyLDAvis visualization (for LDA)
            if self.model_type == 'lda':
                self._create_pyldavis_visualization(output_path)
                
            self.logger.info(f"Visualizations saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
    
    def _create_topic_word_heatmap(self, output_path: str) -> None:
        """
        Create a heatmap of top words per topic.
        
        Args:
            output_path: Directory to save the heatmap.
        """
        try:
            # Get top words for each topic
            top_words = self.get_top_words_per_topic(n=10)
            
            # Create a matrix for the heatmap
            topics = sorted(top_words.keys())
            all_words = set()
            for topic_id in topics:
                all_words.update([word for word, _ in top_words[topic_id]])
            all_words = sorted(all_words)
            
            # Create the matrix
            matrix = np.zeros((len(topics), len(all_words)))
            for i, topic_id in enumerate(topics):
                for word, weight in top_words[topic_id]:
                    j = all_words.index(word)
                    matrix[i, j] = weight
            
            # Create the heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(matrix, xticklabels=all_words, yticklabels=[f"Topic {t}" for t in topics], cmap="YlGnBu")
            plt.title(f"{self.model_type.upper()} Topic-Word Heatmap")
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, "topic_word_heatmap.png"), dpi=300)
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating topic-word heatmap: {str(e)}")
    
    def _create_topic_distribution_plot(self, output_path: str) -> None:
        """
        Create a plot showing the distribution of topics across documents.
        
        Args:
            output_path: Directory to save the plot.
        """
        try:
            if self.document_topic_matrix is None:
                self.logger.warning("Document-topic matrix not available for distribution plot")
                return
                
            # Get the dominant topic for each document
            dominant_topics = np.argmax(self.document_topic_matrix, axis=1)
            
            # Count documents per topic
            topic_counts = np.bincount(dominant_topics, minlength=self.document_topic_matrix.shape[1])
            
            # Create the bar plot
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(topic_counts)), topic_counts)
            plt.xlabel("Topic ID")
            plt.ylabel("Number of Documents")
            plt.title(f"{self.model_type.upper()} Topic Distribution")
            plt.xticks(range(len(topic_counts)))
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, "topic_distribution.png"), dpi=300)
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating topic distribution plot: {str(e)}")
    
    def _create_pyldavis_visualization(self, output_path: str) -> None:
        """
        Create pyLDAvis visualization for LDA model.
        
        Args:
            output_path: Directory to save the visualization.
        """
        try:
            if self.model_type != 'lda':
                return
                
            # Prepare the visualization
            vis_data = pyLDAvis.gensim_models.prepare(
                self.model, 
                self.corpus, 
                self.id2word, 
                mds='tsne', 
                sort_topics=False
            )
            
            # Save the visualization
            pyLDAvis.save_html(vis_data, os.path.join(output_path, "pyldavis.html"))
            
        except Exception as e:
            self.logger.error(f"Error creating pyLDAvis visualization: {str(e)}")
    
    def generate_report(self, output_format: str = 'json') -> str:
        """
        Export evaluation metrics and sample topics.
        
        Args:
            output_format: Format for the report ('json' or 'csv').
            
        Returns:
            Path to the generated report file.
        """
        try:
            # Prepare report data
            report = {
                "model_type": self.model_type,
                "model_path": self.model_path,
                "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "metrics": self.metrics,
                "top_words_per_topic": {
                    f"topic_{topic_id}": [{"word": word, "weight": float(weight)} for word, weight in words]
                    for topic_id, words in self.get_top_words_per_topic(n=10).items()
                }
            }
            
            # Save report
            if output_format.lower() == 'json':
                output_path = os.path.join(self.output_dir, f"{self.model_type}_evaluation.json")
                save_json(report, output_path)
            else:  # csv
                # Flatten the report for CSV
                metrics_df = pd.DataFrame([self.metrics])
                metrics_path = os.path.join(self.output_dir, f"{self.model_type}_metrics.csv")
                metrics_df.to_csv(metrics_path, index=False)
                
                # Save top words as CSV
                top_words = self.get_top_words_per_topic(n=10)
                words_data = []
                for topic_id, words in top_words.items():
                    for rank, (word, weight) in enumerate(words):
                        words_data.append({
                            "topic_id": topic_id,
                            "rank": rank + 1,
                            "word": word,
                            "weight": weight
                        })
                words_df = pd.DataFrame(words_data)
                words_path = os.path.join(self.output_dir, f"{self.model_type}_top_words.csv")
                words_df.to_csv(words_path, index=False)
                
                output_path = metrics_path
            
            self.logger.info(f"Evaluation report saved to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            raise AppException(f"Failed to generate evaluation report: {str(e)}")
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Run all evaluation metrics and generate report.
        
        Returns:
            Dictionary of evaluation metrics.
        """
        try:
            start_time = datetime.now()
            
            # Load artifacts if not already loaded
            if self.model is None:
                self.load_artifacts()
            
            # Compute all metrics
            self.compute_coherence()
            if self.model_type == 'lda':
                self.compute_perplexity()
            self.compute_topic_diversity()
            self.compute_silhouette_score()
            
            # Record evaluation time
            self.metrics["evaluation_time"] = (datetime.now() - start_time).total_seconds()
            
            # Generate report
            self.generate_report()
            
            # Create visualizations
            self.visualize_topics()
            
            self.logger.info(f"Evaluation complete for {self.model_type} model")
            self.logger.info(f"Metrics: {self.metrics}")
            
            return self.metrics
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            raise AppException(f"Evaluation failed: {str(e)}")


def main():
    """
    Command-line interface for model evaluation.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate topic models (from model artifacts or topics CSV)")
    parser.add_argument("--model_type", type=str, required=True, choices=["lda", "nmf"], help="Type of model")
    parser.add_argument("--output_dir", type=str, default="artifacts/evaluation", help="Directory to save evaluation results")
    parser.add_argument("--report_format", type=str, default="json", choices=["json", "csv"], help="Format for the evaluation report")
    # Model artifacts mode
    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained model")
    parser.add_argument("--vectorizer_path", type=str, default=None, help="Path to the trained vectorizer")
    parser.add_argument("--data_path", type=str, default=None, help="Path to the preprocessed data")
    # Topics CSV mode
    parser.add_argument("--topics_csv", type=str, default=None, help="Path to topics CSV (enables CSV-based evaluation)")
    
    args = parser.parse_args()
    # Configure logging to save per-evaluation run logs
    try:
        os.makedirs(args.output_dir or "artifacts/evaluation", exist_ok=True)
    except Exception:
        pass
    log_path = os.path.join(args.output_dir or "artifacts/evaluation", f"{args.model_type}_evaluation.log")
    setup_logging(log_level="INFO", log_file=log_path, force=True)
    
    # Decide evaluation mode
    if args.topics_csv:
        # CSV-based evaluation
        res = evaluate_topics_csv(args.model_type, args.topics_csv, output_dir=args.output_dir)
        print(f"Evaluation complete. Report saved to {res['report_path']}")
        print(f"Metrics: {res['metrics']}")
    else:
        # Validate required args for model mode
        missing = []
        if not args.model_path: missing.append("--model_path")
        if not args.vectorizer_path: missing.append("--vectorizer_path")
        if not args.data_path: missing.append("--data_path")
        if missing:
            raise AppException(f"Missing required arguments for model evaluation: {', '.join(missing)}")
        # Initialize and run evaluator
        evaluator = TopicModelEvaluator(
            model_path=args.model_path,
            vectorizer_path=args.vectorizer_path,
            data_path=args.data_path,
            model_type=args.model_type,
            output_dir=args.output_dir
        )
        metrics = evaluator.evaluate()
        report_path = evaluator.generate_report(output_format=args.report_format)
        print(f"Evaluation complete. Report saved to {report_path}")
        print(f"Metrics: {metrics}")


 
 # -----------------------
 # CSV-based Evaluation
 # -----------------------
 
def evaluate_topics_csv(model_type: str, topics_csv_path: str, output_dir: str = "artifacts/evaluation", topn: int = 5):
     """
     Evaluate a topic model using a topics CSV file containing a 'words' column
     with stringified list of (word, weight) tuples.
     
     Args:
         model_type: 'lda' or 'nmf'
         topics_csv_path: Path to CSV with columns ['topic_id', 'words']
         output_dir: Directory to save evaluation outputs
         topn: Number of top words per topic to visualize
     
     Returns:
         A dict with metrics and paths to saved artifacts
     """
     os.makedirs(output_dir, exist_ok=True)
     logger = get_logger(__name__)
     logger.info(f"Evaluating {model_type} topics from CSV: {topics_csv_path}")
     
     try:
         topics_df = pd.read_csv(topics_csv_path)
     except Exception as e:
         raise AppException(f"Error loading topics CSV: {e}")
     
     # Parse words column
     all_words: List[str] = []
     topic_words: List[Tuple[int, List[str], List[float]]] = []
     for _, row in topics_df.iterrows():
         words_str = row.get('words', '')
         topic_id = int(row.get('topic_id', len(topic_words)))
         try:
             tuples = ast.literal_eval(words_str)
             words = [w for w, _ in tuples]
             weights = [float(p) if isinstance(p, (int, float)) else 0.0 for _, p in tuples]
             all_words.extend(words)
             topic_words.append((topic_id, words, weights))
         except Exception as e:
             logger.warning(f"Error parsing words for topic {topic_id}: {e}")
             continue
     
     # Metrics: topic diversity
     diversity = (len(set(all_words)) / len(all_words)) if all_words else 0.0
     metrics = {
         "topic_diversity": float(diversity),
         "num_topics": int(len(topic_words))
     }
     logger.info(f"Topic diversity: {diversity:.4f} ({len(set(all_words))}/{len(all_words)})")
     
     # Top words per topic (topn)
     top_words_json: Dict[str, List[Dict[str, Union[str, float]]]] = {}
     for topic_id, words, weights in topic_words:
         pairs = []
         for w, p in zip(words[:topn], weights[:topn]):
             pairs.append({"word": w, "weight": float(p)})
         top_words_json[f"topic_{topic_id}"] = pairs
     
     # Heatmap
     heatmap_records: List[Dict[str, Union[str, float]]] = []
     for topic_id, words, weights in topic_words:
         for w, p in zip(words[:topn], weights[:topn]):
             heatmap_records.append({"topic": f"Topic {topic_id}", "word": w, "weight": float(p)})
     if heatmap_records:
         heatmap_df = pd.DataFrame(heatmap_records)
         pivot_df = pd.pivot_table(heatmap_df, values='weight', index='topic', columns='word', fill_value=0)
         plt.figure(figsize=(12, 8))
         sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlGnBu")
         plt.title(f"{model_type.upper()} Top-{topn} Words per Topic")
         heatmap_path = os.path.join(output_dir, f"{model_type}_topic_heatmap.png")
         plt.tight_layout()
         plt.savefig(heatmap_path, dpi=300)
         plt.close()
         logger.info(f"Saved topic heatmap to {heatmap_path}")
     else:
         heatmap_path = None
         logger.warning("No heatmap data available from topics CSV")
     
     # Report JSON
     report = {
         "model_type": model_type,
         "topics_csv": topics_csv_path,
         "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         "metrics": metrics,
         "top_words_per_topic": top_words_json
     }
     report_path = os.path.join(output_dir, f"{model_type}_evaluation.json")
     save_json(report, report_path)
     logger.info(f"Saved CSV-based evaluation report to {report_path}")
     
     return {
         "metrics": metrics,
         "report_path": report_path,
         "heatmap_path": heatmap_path
     }
 
 
def compare_models_csv(lda_metrics, nmf_metrics, output_dir: str = "artifacts/evaluation"):
     """
     Create a comparison bar chart using topic diversity metrics from CSV evaluations.
     """
     os.makedirs(output_dir, exist_ok=True)
     plt.figure(figsize=(10, 6))
     models = ['LDA', 'NMF']
     diversity_scores = [
         float(lda_metrics.get('topic_diversity', 0.0)),
         float(nmf_metrics.get('topic_diversity', 0.0))
     ]
     plt.bar(models, diversity_scores)
     plt.ylabel('Topic Diversity Score')
     plt.title('Topic Diversity Comparison')
     plt.ylim(0, 1)
     for i, v in enumerate(diversity_scores):
         plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
     comparison_path = os.path.join(output_dir, "model_comparison.png")
     plt.tight_layout()
     plt.savefig(comparison_path, dpi=300)
     plt.close()
     return comparison_path


if __name__ == "__main__":
    main()