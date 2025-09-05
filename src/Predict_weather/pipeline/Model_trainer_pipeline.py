from src.Predict_weather.config.configuration import ConfigurationManager
from src.Predict_weather.components.Model_trainer import ModelTrainer
from src.Predict_weather import logger



STAGE_NAME = "Model Trainer Stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass
    
    def initiate_model_trainer(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config = model_trainer_config)
        model_trainer.train()