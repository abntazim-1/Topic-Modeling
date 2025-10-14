import pandas as pd
# from src.data.data_preprocessing import PreprocessingConfig, PreprocessingPipeline

# Load a few examples from the BBC News dataset
DATA_PATH = r"G:\Topic Modeling Project\artifacts\preprocessed_bbc_news.csv"
df = pd.read_csv(DATA_PATH, sep='\t') if DATA_PATH.endswith('.tsv') else pd.read_csv(DATA_PATH)
print(df["tokens"].head(5))