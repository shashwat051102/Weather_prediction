from dataclasses import dataclass
from pathlib import Path

# In dataclass we dont need to use self keyword

@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    
    
@dataclass
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: Path
    unzip_data_dir: Path
    all_schema: dict
    
@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    
@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    sequence_length: int
    hidden_size: int
    dropout: float
    batch_size: int
    learning_rate: float
    epochs: int
    target_column: str
    
@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    scaler_path: Path
    target_column: str
    mlflow_uri: str