import os
from experiments_utils import *
from experiments_utils import settings
from utils.rulekit.params import Measures
from m_of_n_classification.helpers import ExperimentConfig

dir_path = os.path.dirname(os.path.realpath(__file__))
settings.EXPERIMENT_BASE_LOGGING_DIR = f'{dir_path}/logs'


INDUCTION_MEASURE: Measures = Measures.C2  # Measures.C2
PRUNING_MEASURE: Measures = Measures.C2  # Measures.C2
VOTING_MEASURE: Measures = Measures.C2  # Measures.C2
EXACT_M_OF_N: bool = True
VERSION: str = '3.0.1__min_supp_20'

ALLOW_COMPLEX_CONDITIONS_IN_M_OF_N: bool = False

DATASET_BASE_PATH: str = f'{dir_path}/../../../datasets/classification'
RESULTS_BASE_PATH: str = f'{dir_path}/results/{VERSION}'
TMP_DIR = './.tmp'
JAR_PATH: str = f'{dir_path}/../../utils/rulekit/jar/rulekit-1.8.4-all.jar'
LABEL_NAME: str = 'class'

M: int = 2
N: int = 3

MAX_CANDIDATES: int = 1000
MIN_CANDIDATES: int = 10

# min support for frequent dataset not smaller than MIN_SUPP_FRACTION of original dataset
MIN_SUPP_FRACTION: float = 0.20

# JVM initial heap size and max heap size - could be decreased probably
RULEKIT_INITIAL_HEAP_SIZE: int = 20480  # (mb) = 20gb
RULEKIT_MAX_HEAP_SIZE: int = 100920  # (mb) = 80gb
MODEL_TYPES: list[str] = ['plain', 'complex',
                          'exact_M-of-N', 'at_least_M-of-N']


def get_rulekit_params(complex_condition_enabled: bool) -> dict:
    return {
        'min_rule_covered': 2,
        'max_uncovered_fraction': 0.02,
        'max_growing': 0,
        'ignore_missing': False,
        'select_best_candidate': True,

        'induction_measure': INDUCTION_MEASURE,
        'pruning_measure': PRUNING_MEASURE,
        'voting_measure': VOTING_MEASURE,
        'control_apriori_precision': True,

        'discrete_set_conditions_enabled': complex_condition_enabled,
        'negated_conditions_enabled': complex_condition_enabled,
        'intervals_conditions_enabled': complex_condition_enabled,
        'numerical_attributes_conditions_enabled': complex_condition_enabled,
        'nominal_attributes_conditions_enabled': complex_condition_enabled,
        'inner_alternatives_enabled': complex_condition_enabled
    }


CV_ENABLED: bool = True
CV_FOLDS = 10

DATASETS = [
    'hayes-roth',
    'lymphography',
    'glass',
    'ecoli',
    'titanic',
    'anneal',
    'autos',
    'iris',
    'hepatitis',
    'heart-statlog',
    'cylinder-bands',
    'echocardiogram',
    'monk-2',
    'car',
    'nursery',
    'bupa-liver-disorders',
    'mushroom',
    'monk-1',
    'zoo',
    'auto-mpg',
    'monk-3',
    'balance-scale',
    'wine',
    'tic-tac-toe',
    'sonar',
    'flag',
    'cleveland',
    'heart-c',
    'vote',
    'soybean',
]


VARIANTS = {
    'no_discretization': {
        'discretization_enabled': False,
        'use_all_possible_values': False,
    },
}


def generate_paramsets():
    paramsets = []
    for variant_name, variant_data in VARIANTS.items():
        for dataset in DATASETS:
            paramsets.append((
                f'{dataset}__{variant_name}', {
                    'experiment_config': ExperimentConfig(
                        dataset_name=dataset,
                        variant=variant_name,
                        discretization_enabled=variant_data['discretization_enabled'],
                        use_all_possible_values=variant_data['use_all_possible_values']
                    ),
                }))
    return paramsets
