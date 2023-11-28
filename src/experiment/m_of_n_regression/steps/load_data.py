import pandas as pd
import os
from typing import Any, Tuple
import pandas as pd
from logging import Logger
from experiments_utils import step
from m_of_n_regression import config


def load_data(
    dataset_name: str,
    fold_number: int,
    discretization_enabled: bool,
    logger: Logger
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, bool]:
    if discretization_enabled:
        discretized_dataset_path = f'{config.DATASET_BASE_PATH}/{dataset_name}/entropy_mdl'
        if os.path.exists(discretized_dataset_path) and os.path.isdir(discretized_dataset_path):
            logger.info(
                f'Using discretized version of dataset: "{dataset_name}"')
            dataset_path = discretized_dataset_path
        else:
            logger.info(
                f'Dataset: "{dataset_name}" is nominal - using original dataset')
            discretization_enabled = False
    if not discretization_enabled:
        dataset_path = f'{config.DATASET_BASE_PATH}/{dataset_name}'

    if fold_number is not None and 'monk' not in dataset_name:
        dataset_path = f'{dataset_path}/cv/{str(fold_number if fold_number != "full_dataset" else 1)}'
    else:
        dataset_path = f'{dataset_path}/train_test'

    df_train = pd.read_parquet(f'{dataset_path}/train.parquet')
    y_train = df_train['class']
    X_train = df_train.drop('class', axis=1)

    df_test = pd.read_parquet(f'{dataset_path}/test.parquet')
    y_test = df_test['class']
    X_test = df_test.drop('class', axis=1)

    if fold_number == 'full_dataset':
        # special case - run on full dataset
        X_train = pd.concat([X_train, X_test])
        X_test = pd.concat([X_train, X_test])
        y_train = pd.concat([y_train, y_test])
        y_test = pd.concat([y_train, y_test])

    # scale labels min max to [0, 1] range
    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min())
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min())

    return X_train, y_train, X_test, y_test
