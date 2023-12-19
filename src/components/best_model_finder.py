import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from src.logger import logging
from config import Config

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("ModelTrainer")

    def _load_data(self, path):
        return pd.read_csv(path)

    def train_and_evaluate(self, model, X_train, y_train, X_test, y_test):
        try:
            # Train the model
            model.fit(X_train, y_train)

            # Make predictions on the training set
            y_pred_train = model.predict(X_train)

            # Evaluate the model on the training set using RMSE and R2 Score
            rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
            r2_train = r2_score(y_train, y_pred_train)

            self.logger.info(f'{model.__class__.__name__}: Train RMSE: {rmse_train}')
            self.logger.info(f'{model.__class__.__name__}: Train R2 Score: {r2_train}')

            # Make predictions on the test set
            y_pred_test = model.predict(X_test)

            # Evaluate the model on the test set using RMSE and R2 Score
            rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
            r2_test = r2_score(y_test, y_pred_test)

            self.logger.info(f'{model.__class__.__name__}: Test RMSE: {rmse_test}')
            self.logger.info(f'{model.__class__.__name__}: Test R2 Score: {r2_test}')

        except Exception as e:
            self.logger.error(f"An error occurred while training and evaluating {model.__class__.__name__}: {str(e)}")

if __name__ == "__main__":
    # Load configuration
    config = Config()

    # Load datasets
    trainer = ModelTrainer(config)
    train_data = trainer._load_data(config.TRAIN_DATA_PATH)
    test_data = trainer._load_data(config.TEST_DATA_PATH)

    # Extract features and target variable
    X_train, y_train = train_data.drop('highest_value', axis=1), train_data['highest_value']
    X_test, y_test = test_data.drop('highest_value', axis=1), test_data['highest_value']

    # Initialize regressor models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'ElasticNet Regression': ElasticNet(),
        'Decision Tree Regressor': DecisionTreeRegressor(random_state=config.RANDOM_STATE),
        'Random Forest Regressor': RandomForestRegressor(random_state=config.RANDOM_STATE),
        'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=config.RANDOM_STATE),
        'AdaBoost Regressor': AdaBoostRegressor(random_state=config.RANDOM_STATE),
        'Support Vector Regressor': SVR(),
        'XGBoost Regressor': XGBRegressor(objective='reg:squarederror', random_state=config.RANDOM_STATE),
        'LightGBM Regressor': LGBMRegressor(random_state=config.RANDOM_STATE),
        'CatBoost Regressor': CatBoostRegressor(random_state=config.RANDOM_STATE, verbose=False)
    }

    # Train and evaluate each model
    for model_name, model in models.items():
        trainer.train_and_evaluate(model, X_train, y_train, X_test, y_test)