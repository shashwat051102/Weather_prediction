import os
import yaml
from src.Predict_weather import logger
import json
import joblib
from ensure import ensure_annotations
from box.config_box import ConfigBox
from pathlib import Path
from typing import Any
from box.exceptions import BoxValueError

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads yaml files and return 
    
    
    Args:
        path_to_yaml (Path): path like input where yaml file is located
    
    Returns:
        ValueError: If yaml file is empty
        e: empty file
    
    Returns:
        ConfigBox: ConfigBox type object which is dictionary like
    """
    
    
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("The provided YAML file is empty or improperly formatted")
    except Exception as e:
        raise e
    
    

@ensure_annotations

def create_directories(path_to_directories: list, verbose=True):
    """
    Create list of directories
    
    Args:
        Path_to_directories (list): List of path of directories to be created
        ignore_log (bool, optional): ignore if multiple directories are to be created. Defaults to False.
        
    
    """
    
    
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")
            
            

@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Save json data
    
    Args:
        path (Path): path to the json file
        data (dict): data to be saved in json file
    
    """
    
    
    with open(path, "w") as f:
        json.dump(data, f, indent = 4)
    
    logger.info(f"json file saved at: {path}")
    


@ensure_annotations
def load_json(path:Path) -> ConfigBox:
    """
    load json files data
    
    Args:
        path(Path): path to json file
    
    Returns:
        ConfigBox: data as class attributes instead of dict
    
    """
    
    
    with open(path) as f:
        content = json.load(f)
        
    logger.info(f"json file loaded sucessfully from: {path}")
    return ConfigBox(content)




@ensure_annotations
def save_bin(data: Any, path: Path):
    """
    Save binary file
    
    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
        
        
    
    """
    
    joblib.dump(value=data, filename = path)
    logger.info(f"binary file saved at: {path}")
    
