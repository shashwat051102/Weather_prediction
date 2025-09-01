from src.Predict_weather import logger
from src.Predict_weather.pipeline.Data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.Predict_weather.pipeline.Data_validation_pipeline import DataValidationTrainingPipeline

if __name__ == "__main__":
    STAGE_NAME = "Data Ingestion Stage"
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.initiate_data_ingestion()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
        
    STAGE_NAME = "Data Validation Stage"
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.initiate_data_validation()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<<<<<\n\nx==========x")
        
    except Exception as e:
        logger.exception(e)
        raise e