class Config:
    # Add your configuration settings here
    TRAIN_DATA_PATH = 'artifacts/transformed_train_data.csv'
    TEST_DATA_PATH = 'artifacts/transformed_test_data.csv'
    ARTIFACTS_PATH = 'artifacts/'
    RANDOM_SEARCH_ITER = 10
    CV_FOLDS = 5
    RANDOM_STATE = 42
    N_JOBS = -1