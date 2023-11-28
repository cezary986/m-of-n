import pandas as pd
import numpy as np
import os
from typing import Any, List, Dict, Tuple
from jpype import JClass
from utils.rulekit.classification import RuleClassifier
from utils.rulekit.helpers import create_example_set
import uuid

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

    X_train_bin, attributes_mappings = _generate_binary_dataset(model, train_example_set, tmp_dir)
    X_test_bin, _ = _generate_binary_dataset(model, test_example_set, tmp_dir)
    return X_train_bin, X_test_bin, attributes_mappings


