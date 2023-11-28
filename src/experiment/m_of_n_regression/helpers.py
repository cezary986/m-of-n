import pandas as pd
from dataclasses import dataclass
import os
from typing import Any, Tuple
import pandas as pd
from logging import Logger
from utils.rulekit.regression import RuleRegressor
from experiments_utils import Store

@dataclass
class ExperimentConfig:
    dataset_name: str
    variant: str
    discretization_enabled: str
    use_all_possible_values: bool



class ExperimentModels:
    plain_rules: RuleRegressor
    complex_rules: RuleRegressor
    exact_m_of_n_rules: RuleRegressor
    at_least_m_of_n_rules: RuleRegressor


class ExperimentDatasets:
    X_train: pd.DataFrame
    y_train: pd.DataFrame

    X_test: pd.DataFrame
    y_test: pd.DataFrame

    X_bin_train: pd.DataFrame
    y_bin_train: pd.DataFrame

    X_bin_test: pd.DataFrame
    y_bin_test: pd.DataFrame

    X_exact_binary_train: pd.DataFrame
    X_exact_binary_test: pd.DataFrame

    X_at_least_binary_train: pd.DataFrame
    X_at_least_binary_test: pd.DataFrame

    attributes_mapping: dict

    def set_original_dataset(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def set_binary_dataset(self, X_train, X_test, attributes_mapping: dict):
        self.X_bin_train = X_train
        self.X_bin_test = X_test
        self.attributes_mapping = attributes_mapping

    def set_exact_m_of_n_binary_dataset(self, X_train, X_test, attributes_mapping: dict):
        self.X_exact_binary_train = X_train
        self.X_exact_binary_test = X_test
        self.attributes_mapping = attributes_mapping

    def set_at_least_m_of_n_binary_dataset(self, X_train, X_test, attributes_mapping: dict):
        self.X_at_least_binary_train = X_train
        self.X_at_least_binary_test = X_test
        self.attributes_mapping = attributes_mapping


class CVFoldStore:

    def __init__(self, store: Store, fold_number: int) -> None:
        object.__setattr__(self, '__store__', store)
        object.__setattr__(self, '__fold_number__', fold_number)

    def __setattr__(self, name: str, value: Any) -> None:
        store: Store = object.__getattribute__(self, '__store__')
        fold_number: int = object.__getattribute__(self, '__fold_number__')

        try:
            new_value = getattr(store, name)
        except NameError:
            new_value = {}
        if not isinstance(new_value, dict):
            new_value = {}
        new_value[fold_number] = value
        setattr(store, name, new_value)

    def __getattribute__(self, name: str) -> Any:
        store: Store = object.__getattribute__(self, '__store__')
        fold_number: int = object.__getattribute__(self, '__fold_number__')
        value = getattr(store, name)
        if fold_number not in value:
            raise NameError(f"name '{name}' is not defined")
        return value[fold_number]
