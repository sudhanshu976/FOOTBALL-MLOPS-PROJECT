# model_trainer.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.metrics import mean_squared_error
import joblib
from src.exception import CustomException 
from src.logger import logging  # Assuming your logger is in src/logger.py
from src.components.config import Config

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.logger = logging.getLogger("ModelTrainer")

    def _load_data(self, path):
        return pd.read_csv(path)

    def _train_model(self, X_train, y_train):
        param_dist = {
            'n_estimators': randint(50, 200),
            'learning_rate': uniform(0.01, 0.2),
            'max_depth': randint(3, 6),
            'min_samples_split': randint(2, 11),
            'min_samples_leaf': randint(1, 5),
            'subsample': uniform(0.8, 0.2)
        }

        gb_regressor = GradientBoostingRegressor()
        random_search = RandomizedSearchCV(
            estimator=gb_regressor, 
            param_distributions=param_dist, 
            n_iter=self.config.RANDOM_SEARCH_ITER, 
            scoring='neg_mean_squared_error', 
            cv=self.config.CV_FOLDS, 
            n_jobs=self.config.N_JOBS, 
            random_state=self.config.RANDOM_STATE
        )

        random_search.fit(X_train, y_train)
        self.model = random_search.best_estimator_

    def train_and_evaluate(self):
        try:
            # Load training data
            train_data = self._load_data(self.config.TRAIN_DATA_PATH)
            X_train, X_val, y_train, y_val = train_test_split(
                train_data.drop('highest_value', axis=1),
                train_data['highest_value'],
                test_size=0.2,
                random_state=self.config.RANDOM_STATE
            )

            # Load testing data
            test_data = self._load_data(self.config.TEST_DATA_PATH)
            X_test = test_data.drop('highest_value', axis=1)
            y_test = test_data['highest_value']

            # Train the model
            self._train_model(X_train, y_train)

            # Save the trained model
            model_filename = f'{self.config.ARTIFACTS_PATH}best_model.pkl'
            joblib.dump(self.model, model_filename)
            self.logger.info(f"Best model saved to {model_filename}")

            # Make predictions on the validation set
            y_pred_val = self.model.predict(X_val)

            # Evaluate the model on the validation set
            mse_val = mean_squared_error(y_val, y_pred_val, squared=False)
            self.logger.info(f'Mean Squared Error on Validation Set: {mse_val}')

            # Make predictions on the test set
            y_pred_test = self.model.predict(X_test)

            # Evaluate the model on the test set
            mse_test = mean_squared_error(y_test, y_pred_test, squared=False)
            self.logger.info(f'Mean Squared Error on Test Set: {mse_test}')

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    logging.info("Model Training Started")
    # Create a ModelTrainer instance with the provided configuration
    trainer = ModelTrainer(Config())

    # Train and evaluate the model
    trainer.train_and_evaluate()

    logging.info("Model Training Completed")
