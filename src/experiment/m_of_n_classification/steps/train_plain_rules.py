import pandas as pd
from experiments_utils import step
from main import config
from utils.rulekit.classification import RuleClassifier


def train_plain_rules(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RuleClassifier:
    """Train plain ruleset on original data.

    Args:
        X_train (pd.DataFrame):
        y_train (pd.Series):
    Returns:
        RuleClassifier: plain ruleset trained on original data
    """
    clf = RuleClassifier(
        **(config.get_rulekit_params(complex_condition_enabled=False))
    )
    clf.fit(X_train, y_train)
    return clf
