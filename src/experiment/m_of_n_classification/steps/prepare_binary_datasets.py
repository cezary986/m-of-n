from logging import getLogger
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from main import config
from m_of_n_classification.helpers import *
from utils.rulekit.classification import RuleClassifier
from experiments_utils import step
from main import config
import pandas as pd
import os
from typing import Any, List, Dict, Tuple
from jpype import JClass
from utils.rulekit.classification import RuleClassifier
from utils.rulekit.helpers import create_example_set
import uuid


def filter_complex_conditions_from_binary_df(
    X_bin: pd.DataFrame
) -> pd.DataFrame:
    columns_to_stay = []
    for column in X_bin.columns.tolist():
        if ' or ' in column.lower():
            continue
        if '{' in column and ', ' in column:
            continue
        else:
            columns_to_stay.append(column)
    return X_bin[columns_to_stay]


def _generate_binary_dataset(
    model: RuleClassifier,
    example_set,
    tmp_dir: str = './tmp'
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:

    id: str = str(uuid.uuid4())
    BinaryDatasetGenerator = JClass(
        'adaa.analytics.rules.utils.BinaryDatasetGenerator')
    tmp_file_path: str = os.path.join(tmp_dir, f'{id}.csv')
    os.makedirs(tmp_dir, exist_ok=True)
    BinaryDatasetGenerator.writeBinaryDatasetToCsvFile(
        model.model._java_object,
        example_set,
        tmp_file_path
    )
    binary_df = pd.read_csv(tmp_file_path, sep=';')

    mappings_tmp_file_path = tmp_file_path.replace('.csv', '.mappings.txt')
    conditions: List[str] = binary_df.columns.tolist()
    attributes_mappings: Dict[str, List[str]] = {}
    with open(mappings_tmp_file_path, 'r') as file:
        for i, line in enumerate(file.readlines()):
            if conditions[i] not in attributes_mappings:
                attributes_mappings[conditions[i]] = list(
                    filter(lambda e: len(e) > 0, line.strip().split(','))
                )
    os.remove(tmp_file_path)
    os.remove(mappings_tmp_file_path)
    return binary_df, attributes_mappings


def generate_binary_datasets(
    model: RuleClassifier,
    X_train: pd.DataFrame,
    y_train,
    X_test: pd.DataFrame,
    y_test,
    tmp_dir: str = './tmp'
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    train_example_set = create_example_set(X_train, y_train)
    test_example_set = create_example_set(X_test, y_test)

    X_train_bin, attributes_mappings = _generate_binary_dataset(
        model, train_example_set, tmp_dir)
    X_test_bin, _ = _generate_binary_dataset(model, test_example_set, tmp_dir)
    return X_train_bin, X_test_bin, attributes_mappings


def prepare_binary_datasets(
    experiment_config: ExperimentConfig,
    plain_clf: RuleClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    """Generates binary dataset based on ruleset model.
    """
    X_train_bin, X_test_bin, attributes_mappings = generate_binary_datasets(
        model=plain_clf,
        X_train=X_train.copy(),
        y_train=y_train,
        X_test=X_test.copy(),
        y_test=y_test,
        tmp_dir=config.TMP_DIR
    )

    if not config.ALLOW_COMPLEX_CONDITIONS_IN_M_OF_N:
        X_train_bin = filter_complex_conditions_from_binary_df(X_train_bin)
        X_test_bin = filter_complex_conditions_from_binary_df(X_test_bin)

    if experiment_config.use_all_possible_values:
        X_train, X_test = one_hot_encode_nominal_dataset(X_train, X_test)
        for column in X_train.columns.tolist():
            attributes_mappings[column] = [
                column.replace('!', '').split('=')[0].strip()]
        # wszystkie przedziały dyskretyzacji + warunki
        X_train_bin = pd.concat([X_train.reset_index(drop=True), X_train_bin.reset_index(drop=True)],
                                axis=1)
        X_train_bin = X_train_bin.loc[:, ~
                                      X_train_bin.columns.duplicated()].copy()
        X_test_bin = pd.concat(
            [X_test.reset_index(drop=True), X_test_bin.reset_index(drop=True)], axis=1)
        X_test_bin = X_test_bin.loc[:, ~X_test_bin.columns.duplicated()].copy()

    getLogger().info(
        f'Bazowa tablica binarna posiada: {X_train_bin.shape[1]} kolumn')
    return X_train_bin.astype(str), X_test_bin.astype(str), attributes_mappings


def one_hot_encode_nominal_dataset(X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    oh: OneHotEncoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown='ignore'
    ).set_output(transform="pandas")
    oh.fit(X_train)
    one_hot_encoded: pd.DataFrame = oh.transform(X_train)
    one_hot_encoded = one_hot_encoded.rename(columns={
        column_name: column_name.replace('_', '=') for column_name in one_hot_encoded.columns.tolist()
    })
    X_train = pd.concat([X_train, one_hot_encoded], axis=1).drop(
        columns=X_train.columns
    ).astype(int)

    one_hot_encoded = oh.transform(X_test)
    one_hot_encoded = one_hot_encoded.rename(columns={
        column_name: column_name.replace('_', '=') for column_name in one_hot_encoded.columns.tolist()
    })
    X_test = pd.concat([X_test, one_hot_encoded], axis=1).drop(
        columns=X_test.columns
    ).astype(int)
    X_train = X_train.rename(columns={
        column_name: column_name.replace('_', '=') for column_name in X_train.columns.tolist()
    })
    X_test = X_test.rename(columns={
        column_name: column_name.replace('_', '=') for column_name in X_train.columns.tolist()
    })
    return X_train, X_test
