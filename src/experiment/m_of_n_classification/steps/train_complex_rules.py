import pandas as pd
from experiments_utils import step
from m_of_n_classification import config
from utils.rulekit.classification import RuleClassifier



def train_complex_rules(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RuleClassifier:
    """Train ruleset with complex condition on original data.

    Args:
        X_train (pd.DataFrame):
        y_train (pd.Series):
    Returns:
        RuleClassifier: ruleset with complex condition trained on original data
    """
    clf = RuleClassifier(
        **(config.get_rulekit_params(complex_condition_enabled=True))
    )
    clf.fit(X_train, y_train)
    return clf
