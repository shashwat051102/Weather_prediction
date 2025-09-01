from src.Predict_weather.config.configuration import ConfigurationManager
from src.Predict_weather.components.Data_ingestion import DataIngestion
from src.Predict_weather import logger


STAGE_NAME = "Data Ingestion Stage"


class DataIngestionTrainingPipeline:
    
    def __init__(self):
        pass
    
    
    def initiate_data_ingestion(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config = data_ingestion_config)
        data_ingestion.download_data()
        
        

if __name__ == "__main__":
    try:
        logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.initiate_data_ingestion()
        logger.info(f">>>>> Stage {STAGE_NAME} completed <<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e