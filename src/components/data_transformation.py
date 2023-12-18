# data_transformation.py
import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from src.logger import logging
from src.exception import CustomException 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox

class DataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoders = {}

    def drop_columns(self, df):
        columns_to_drop = ['player', 'name', 'height', 'goals', 'assists', 'yellow cards', 'second yellow cards', 'red cards',
                           'goals conceded', 'clean sheets', 'days_injured', 'position_encoded', 'winger']
        df.drop(columns=columns_to_drop, axis=1, inplace=True)

    def encode_top_n_teams(self, df):
        label_encoder = LabelEncoder()
        top_n_teams = df['team'].value_counts().nlargest(20).index
        df['team'] = df['team'].apply(lambda x: x if x in top_n_teams else 'Other')
        df['team'] = label_encoder.fit_transform(df['team'])
        logging.info(f"Encoded classes are :{label_encoder.classes_}")

    def encode_position_column(self, df):
        category_counts = df['position'].value_counts()
        categories_to_group = category_counts[category_counts < 500].index
        df['position'] = df['position'].replace(categories_to_group, 'Other')
        label_encoder = LabelEncoder()
        df['position'] = label_encoder.fit_transform(df['position'])
        logging.info(f"Encoded classes are :{label_encoder.classes_}")

    def handle_appearance(self, df):
        median_appearance = df['appearance'].median()
        df['appearance'] = df['appearance'].replace(0, median_appearance)

    def handle_minutes_played(self, df):
        non_zero_median = df[df['minutes played'] > 0]['minutes played'].median()
        df['minutes_played'] = df['minutes played'].replace(0, non_zero_median)
        df['minutes_played'], lamda_value = boxcox(df['minutes_played'] + 1)
        logging.info(f"Lambda value for boxcox transformation is :{lamda_value}")

    def create_binary_columns(self, df):
        df['games_injured'] = (df['games_injured'] == 0).astype(int)
        df['award'] = (df['award'] == 0).astype(int)

    def remove_zero_value_players(self, df):
        df = df[df['current_value'] != 0]

    def fit(self, X, y=None):
        return self

    def transform(self, datasets):
        try:
            df1 = datasets[0].copy()
            df2 = datasets[1].copy()

            self.drop_columns(df1)
            self.drop_columns(df2)

            self.encode_top_n_teams(df1)
            self.encode_top_n_teams(df2)

            self.encode_position_column(df1)
            self.encode_position_column(df2)

            self.handle_appearance(df1)
            self.handle_appearance(df2)

            self.handle_minutes_played(df1)
            self.handle_minutes_played(df2)

            self.create_binary_columns(df1)
            self.create_binary_columns(df2)
            self.remove_zero_value_players(df1)
            self.remove_zero_value_players(df2)

            return df1, df2
        except Exception as e:
            raise CustomException(e, sys)

    def save_transformed_data(self, df1, df2, filepath1='artifacts/transformed_train_data.csv', filepath2='artifacts/transformed_test_data.csv'):
        os.makedirs(os.path.dirname(filepath1), exist_ok=True)
        os.makedirs(os.path.dirname(filepath2), exist_ok=True)
        df1.to_csv(filepath1, index=False, header=True)
        df2.to_csv(filepath2, index=False, header=True)

    def save_label_encoders(self, filepath='artifacts/preprocessor.pkl'):
        joblib.dump(self.label_encoders, filepath)

    def load_label_encoders(self, filepath='artifacts/preprocessor.pkl'):
        self.label_encoders = joblib.load(filepath)

if __name__ == "__main__":
    logging.info("Data Transformation Started")

    transformer = DataTransformer()

    # Load raw data
    train_data = pd.read_csv('artifacts/train.csv')
    test_data = pd.read_csv('artifacts/test.csv')

    # Split the data into train and test sets

    # Transform the data
    transformed_train_data, transformed_test_data = transformer.transform((train_data, test_data))

    # Save transformed data
    transformer.save_transformed_data(transformed_train_data, transformed_test_data)

    # Save label encoders
    transformer.save_label_encoders()

    logging.info("Data Transformation Completed")

