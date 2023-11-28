import pandas as pd
from experiments_utils import step
from m_of_n_regression import config
from utils.rulekit.regression import RuleRegressor


def train_plain_rules(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RuleRegressor:
    """Train plain ruleset on original data.

    Args:
        X_train (pd.DataFrame):
        y_train (pd.Series):
    Returns:
        RuleRegressor: plain ruleset trained on original data
    """
    reg = RuleRegressor()
    reg.set_params(
        **(config.get_rulekit_params(complex_condition_enabled=False))
    )
    reg.fit(X_train, y_train)
    return reg
