import os
import numpy as np
from experiments_utils.results.tables import Tables
from pandas.api.types import is_numeric_dtype


def string_supporting_mean(col):
    if is_numeric_dtype(col):
        return col.mean()
    else:
        return col.unique() if col.nunique() == 1 else np.NaN


def print_rules(dataset_name: str, model_type: str, fold_index: int = 0):
    base_dir: str = Tables._directory
    if fold_index is not None:
        path = os.path.join(
            base_dir,
            dataset_name,
            'no_discretization',
            model_type,
            'cv',
            str(fold_index),
            'rules',
            'rules.txt'
        )
    else:
        path = os.path.join(
            base_dir,
            dataset_name,
            'no_discretization',
            model_type,
            'rules',
            'rules.txt'
        )
    with open(path, 'r') as file:
        start_printing = False
        for line in file.readlines():
            if '_____' in line:
                start_printing = True
            if start_printing:
                print(line.strip())
