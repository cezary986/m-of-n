from typing import Dict
from experiments_utils import step
from main import config
from m_of_n_survival.helpers import *
from utils.rulekit.survival import SurvivalRules

def train_rules_with_m_of_n(
    X_bin_ext_train,
    y_train: pd.Series,
    attributes_mappings: Dict[str, int]
) -> SurvivalRules:
    from java.lang import System
    from java.util import HashMap, HashSet

    clf_with_m_of_n = SurvivalRules()
    clf_with_m_of_n.set_params(
        **config.get_rulekit_params(complex_condition_enabled=False)
    )

    # prepare attributes mappings
    clf_with_m_of_n.fit(
        X_bin_ext_train,
        y_train,
        attributes_mappings=HashMap({
            key: HashSet(value) for key, value in attributes_mappings.items()
        })
    )
    return clf_with_m_of_n
