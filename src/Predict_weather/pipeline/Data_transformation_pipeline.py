from src.Predict_weather.config.configuration import ConfigurationManager
from src.Predict_weather.components.Data_transformation import DataTransformation
from src.Predict_weather import logger
from pathlib import Path

STAGE_NAME = "Data Transformation Stage"


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass
    
    
    def initiate_data_transformation(self):
        
        try:
            with open(Path("artifacts/data_validation/status.txt"),"r") as f:
                status = f.read().split(" ")[-1]
                if status == "True":
                    config = ConfigurationManager()
                    data_transformation_config = config.get_data_transformation_config()
                    data_transformation = DataTransformation(config=data_transformation_config)
                    data_transformation.transform_data()
                else:
                    raise Exception("Data Validation Failed. Data Transformation cannot be initiated")
        except Exception as e:
            print(e)
