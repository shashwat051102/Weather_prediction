from src.Predict_weather import logger
from src.Predict_weather.config.configuration import ConfigurationManager
from src.Predict_weather.components.Model_evaluation import ModelEvaluation






STAGE_NAME = "Model Evaluation Stage"


class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_evaluation(self):
        config = ConfigurationManager()
        model_eval_config = config.get_model_evaluation_config()
        
        model_evaluation = ModelEvaluation(config=model_eval_config)
        model_evaluation.log_into_mlflow()
        

