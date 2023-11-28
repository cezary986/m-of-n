from typing import Dict
from experiments_utils import step
from main import config
from m_of_n_regression.helpers import *


def train_rules_with_m_of_n(
    X_bin_ext_train,
    y_train: pd.Series,
    attributes_mappings: Dict[str, int]
) -> RuleRegressor:
    from java.lang import System
    from java.util import HashMap, HashSet

    clf_with_m_of_n = RuleRegressor()
    clf_with_m_of_n.set_params(
        **config.get_rulekit_params(complex_condition_enabled=False)
    )
    try:
        # prepare attributes mappings
        clf_with_m_of_n.fit(
            X_bin_ext_train,
            y_train,
            attributes_mappings=HashMap({
                key: HashSet(value) for key, value in attributes_mappings.items()
            })
        )
    except Exception as e:
        print(e.java_stack_trace, e.message)
    return clf_with_m_of_n
