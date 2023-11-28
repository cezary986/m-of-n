import pandas as pd
from experiments_utils import step
from m_of_n_regression import config
from utils.rulekit.regression import RuleRegressor
from utils.rulekit.exceptions import JavaBackendException


def train_complex_rules(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RuleRegressor:
    """Train ruleset with complex condition on original data.

    Args:
        X_train (pd.DataFrame):
        y_train (pd.Series):
    Returns:
        RuleRegressor: ruleset with complex condition trained on original data
    """
    reg = RuleRegressor()
    reg.set_params(**(config.get_rulekit_params(complex_condition_enabled=True)))
    try:
        reg.fit(X_train, y_train)
    except JavaBackendException as e:
        print(e.message)
        print(e.java_stack_trace)
        raise e
    return reg
