import pandas as pd
import joblib
from scipy.stats import boxcox
from src.components.model_trainer import ModelTrainer
from src.components.config import Config

class PredictPipeline:
    def __init__(self):
        # Load the pre-trained model
        self.trainer = ModelTrainer(Config())
        self.trainer.model = joblib.load('artifacts/best_model.pkl')
        self.lambda_value = 0.4469754727550972

    def preprocess_input(self, user_data):
        # Preprocess user input
        user_data['minutes played'] = boxcox(user_data['minutes played'] + 1, lmbda=self.lambda_value)
        return user_data

    def make_prediction(self, user_data):
        # Make a prediction
        prediction = self.trainer.predict(user_data)
        return int(prediction[0])