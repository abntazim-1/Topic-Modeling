import os
import yaml
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import pandas as pd
from src.utils.logger import get_logger
from src.utils.exceptions import DataValidationError, AppException


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Loads YAML config from file path."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@dataclass
class DataIngestionConfig:
    """
    Handles data ingestion configuration settings (paths, csv delimiter, encoding, logging) loaded from YAML.
    """
    csv_path: str
    expected_columns: Optional[List[str]] = None
    delimiter: str = ","
    encoding: str = "utf-8"
    log_level: str = "INFO"
    allow_duplicates: bool = False
    config_file_used: str = field(default=None, init=False)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DataIngestionConfig":
        cfg = load_yaml_config(yaml_path)
        return cls(
            csv_path=cfg['data']['csv_path'],
            expected_columns=cfg['data'].get('expected_columns'),
            delimiter=cfg['data'].get('delimiter', ','),
            encoding=cfg['data'].get('encoding', 'utf-8'),
            log_level=cfg.get('logging', {}).get('level', 'INFO'),
            allow_duplicates=cfg['data'].get('allow_duplicates', False)
        )

class DataValidator:
    """
    Handles data file validation: existence, schema, nulls, duplicates.
    """
    def __init__(self, config: DataIngestionConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or get_logger(__name__)

    def validate_file_exists(self) -> None:
        """Raise if the CSV file does not exist."""
        if not os.path.isfile(self.config.csv_path):
            self.logger.error(f"CSV file not found: {self.config.csv_path}")
            raise DataValidationError(f"CSV file not found: {self.config.csv_path}")
        self.logger.info(f"CSV file found: {self.config.csv_path}")

    def validate_schema(self, df: pd.DataFrame) -> None:
        """Raise if DataFrame does not have expected columns (if provided)."""
        if self.config.expected_columns:
            missing = set(self.config.expected_columns) - set(df.columns)
            if missing:
                self.logger.error(f"Missing columns: {missing}")
                raise DataValidationError(f"Missing columns in CSV: {missing}")
            self.logger.info("Schema validated: all expected columns are present.")

    def validate_nulls(self, df: pd.DataFrame) -> None:
        """Raise if DataFrame has null values."""
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()
        if total_nulls > 0:
            self.logger.warning(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
            raise DataValidationError(f"Missing values found in data: {null_counts[null_counts > 0].to_dict()}")
        self.logger.info("Null value check passed.")

    def validate_duplicates(self, df: pd.DataFrame) -> None:
        """Raise if duplicates are found and not allowed."""
        num_dupes = df.duplicated().sum()
        if num_dupes > 0 and not self.config.allow_duplicates:
            self.logger.warning(f"{num_dupes} duplicate rows found.")
            raise DataValidationError(f"{num_dupes} duplicate rows found in data.")
        self.logger.info("Duplicate check passed.")

class DataIngestion:
    """
    Handles reading and processing of a CSV file as per configuration.
    Returns a cleaned pandas.DataFrame.
    """
    def __init__(self, config: DataIngestionConfig, validator: Optional[DataValidator] = None, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or get_logger(__name__)
        self.validator = validator or DataValidator(config, self.logger)

    def ingest(self) -> pd.DataFrame:
        """
        Reads the CSV, validates and cleans it, returns a DataFrame.
        """
        try:
            self.logger.info(f"Starting ingestion for {self.config.csv_path}")
            self.validator.validate_file_exists()

            df = pd.read_csv(self.config.csv_path, delimiter=self.config.delimiter, encoding=self.config.encoding)
            self.logger.info(f"CSV read successfully: {df.shape[0]} rows x {df.shape[1]} columns.")
            self.validator.validate_schema(df)
            self.validator.validate_nulls(df)
            self.validator.validate_duplicates(df)

            if not self.config.allow_duplicates:
                df.drop_duplicates(inplace=True)

            df = self.basic_clean(df)
            self.logger.info("Data ingestion and cleaning completed.")
            self.print_summary(df)
            return df
        except Exception as exc:
            self.logger.error(f"Data ingestion failed: {exc}")
            raise AppException(f"Data ingestion failed: {exc}") from exc

    def basic_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies basic cleaning: strip whitespace from string columns.
        """
        str_cols = df.select_dtypes(include='object').columns
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip()
        self.logger.info('Whitespace stripped from object columns.')
        return df

    def print_summary(self, df: pd.DataFrame) -> None:
        """
        Prints basic summary of the ingested data.
        """
        summary = (
            f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n"
            f"Missing values per column: {df.isnull().sum().to_dict()}\n"
            f"Columns: {list(df.columns)}"
        )
        print("Data Summary:")
        print(summary)
        self.logger.info(f"Data summary: {summary}")


if __name__ == "__main__":
    import sys
    # Sample usage with YAML config
    sample_config_path = "config/config.yaml"
    if not os.path.isfile(sample_config_path):
        print(f"Sample config not found at {sample_config_path}. Create one as shown in docstring.")
        sys.exit(1)
    config = DataIngestionConfig.from_yaml(sample_config_path)
    logger = get_logger("data_ingestion")
    logger.info("Loaded configuration.")

    validator = DataValidator(config, logger)
    ingestion = DataIngestion(config, validator, logger)
    try:
        df = ingestion.ingest()
        logger.info("Data ingestion pipeline completed.")
    except AppException as exc:
        logger.error(f"Pipeline failed: {exc}")
        sys.exit(1)
    logger.info("Script finished successfully.")
