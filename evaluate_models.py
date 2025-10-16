import os
import warnings
from src.utils.logger import setup_logging, get_logger

# Suppress noisy warnings before importing evaluation modules
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="tensorflow")
warnings.filterwarnings("ignore", module="tf_keras")
warnings.filterwarnings("ignore", message="is deprecated. Please use")
warnings.filterwarnings("ignore", module="pyLDAvis")

try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
except Exception:
    pass

from src.models.evaluate import evaluate_topics_csv, compare_models_csv

# Configure logging to save a wrapper run log into artifacts/evaluation
OUTPUT_DIR = "artifacts/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)
setup_logging(log_level="INFO", log_file=os.path.join(OUTPUT_DIR, "evaluate_models.log"), force=True)
logger = get_logger(__name__)

OUTPUT_DIR = OUTPUT_DIR


def evaluate_topic_model(model_type: str, topics_csv_path: str):
    print(f"\n===== Evaluating {model_type.upper()} Model =====")
    logger.info(f"Evaluating {model_type} using CSV: {topics_csv_path}")
    res = evaluate_topics_csv(model_type, topics_csv_path, output_dir=OUTPUT_DIR, topn=5)
    print(f"Evaluation report saved to {res['report_path']}")
    logger.info(f"Evaluation report saved to {res['report_path']}")
    return res


def main():
    # Evaluate LDA
    lda_results = evaluate_topic_model('lda', 'artifacts/lda_topics.csv')

    # Evaluate NMF
    nmf_results = evaluate_topic_model('nmf', 'artifacts/nmf_topics.csv')

    # Compare models
    if lda_results and nmf_results:
        print("\n===== Model Comparison =====")
        print(f"LDA Topic Diversity: {lda_results['metrics']['topic_diversity']:.4f}")
        print(f"NMF Topic Diversity: {nmf_results['metrics']['topic_diversity']:.4f}")
        comparison_path = compare_models_csv(lda_results['metrics'], nmf_results['metrics'], output_dir=OUTPUT_DIR)
        print(f"Model comparison visualization saved to {comparison_path}")


if __name__ == "__main__":
    main()