from logging import getLogger
import _
import os
from glob import glob
from joblib import Parallel, delayed
import time
from experiments_utils import *
from experiments_utils.results.tables import *
import config
from utils.rulekit import RuleKit
from m_of_n_classification.steps import *
from m_of_n_classification.helpers import *
import warnings
import shutil


def initialize_experiment(experiment_config: ExperimentConfig):
    Tables.configure(config.RESULTS_BASE_PATH)
    print(f'Using rulekit version: {RuleKit.version}')


def finish_experiment(experiment_config: ExperimentConfig):
    for model_type in config.MODEL_TYPES:
        cv_dir_path = os.path.join(
            config.RESULTS_BASE_PATH,
            experiment_config.dataset_name,
            experiment_config.variant,
            model_type,
            'cv',
        )
        src_path: str = os.path.join(
            cv_dir_path,
            'full_dataset'
        )
        dest_path: str = os.path.join(
            config.RESULTS_BASE_PATH,
            experiment_config.dataset_name,
            experiment_config.variant,
            model_type,
        )
        shutil.move(src_path, dest_path)

        if len(os.listdir(cv_dir_path)) == 0:
            shutil.rmtree(cv_dir_path)


def run_experiment(
    experiment_config: ExperimentConfig,
    fold_number: int,
):
    models = ExperimentModels()
    datasets = ExperimentDatasets()

    X_train, y_train, X_test, y_test = load_data(
        experiment_config.dataset_name,
        fold_number,
        experiment_config.discretization_enabled,
        getLogger()
    )
    datasets.set_original_dataset(X_train, y_train, X_test, y_test)

    # 1. trenowanie klasycznych reguł
    models.plain_rules = RuleClassifier = train_plain_rules(
        datasets.X_train,
        datasets.y_train
    )

    # 2. trenowanie reguł z warunkami złożonymi
    models.complex_rules: RuleClassifier = train_complex_rules(
        datasets.X_train,
        datasets.y_train
    )

    # 3. przygotowanie tablicy binarnej
    X_train_bin, X_test_bin, attributes_mappings = prepare_binary_datasets(
        experiment_config,
        models.complex_rules,
        datasets.X_train,
        datasets.X_test,
        datasets.y_train,
        datasets.y_test
    )
    datasets.set_binary_dataset(X_train_bin, X_test_bin, attributes_mappings)

    start_time = time.time()
    # 5. wygenerowanie rozszerzonej tablicy binarnej - warunki dokładnie M-of-N
    (
        X_exact_binary_train,
        X_exact_binary_test,
        attributes_mapping,
    ) = find_m_of_n_candidates(
        datasets,
        exact_m_of_n=True,
        logger=getLogger(experiment_config.dataset_name)
    )
    X_exact_binary_train = X_exact_binary_train
    X_exact_binary_test = X_exact_binary_test
    attributes_mapping = attributes_mapping
    datasets.set_exact_m_of_n_binary_dataset(
        X_exact_binary_train,
        X_exact_binary_test,
        attributes_mapping
    )

    # 6. trenowanie reguł z warunkami dokładnie m-of-n
    models.exact_m_of_n_rules = train_rules_with_m_of_n(
        datasets.X_exact_binary_train,
        datasets.y_train,
        datasets.attributes_mapping
    )
    models.exact_m_of_n_rules.model.stats.time_total_s = time.time() - start_time

    # 7. wygenerowanie rozszerzonej tablicy binarnej - warunki przynajmniej M-of-N
    start_time = time.time()
    (
        X_at_least_binary_train,
        X_at_least_binary_test,
        attributes_mapping,
    ) = find_m_of_n_candidates(
        datasets,
        exact_m_of_n=False,
        logger=getLogger()
    )
    datasets.set_at_least_m_of_n_binary_dataset(
        X_at_least_binary_train,
        X_at_least_binary_test,
        attributes_mapping
    )

    # 8. trenowanie reguł z warunkami przynajmniej m-of-n
    models.at_least_m_of_n_rules = train_rules_with_m_of_n(
        datasets.X_at_least_binary_train,
        datasets.y_train,
        datasets.attributes_mapping
    )
    models.at_least_m_of_n_rules.model.stats.time_total_s = time.time()

    # 9. Generowanie i zapis wyników
    generate_results(
        experiment_config,
        models,
        datasets,
        fold_number
    )


def run_expriment_with_cross_validation(
    experiment_config: ExperimentConfig
):
    run_experiment(experiment_config, 'full_dataset')
    print(experiment_config)
    for fold_number in range(1, config.CV_FOLDS + 1):
        print(f'Fold {fold_number}')
        run_experiment(experiment_config, fold_number)
    aggreage_cv_folds_results(
        experiment_config.dataset_name,
        experiment_config.variant,
    )


def run_experiment_train_test(
    experiment_config: ExperimentConfig
):
    run_experiment(experiment_config, 'full_dataset')

    run_experiment(experiment_config, None)


def main(
    experiment_config: ExperimentConfig
):
    RuleKit.init(
        jar_file_path=config.JAR_PATH,
        initial_heap_size=config.RULEKIT_INITIAL_HEAP_SIZE,
        max_heap_size=config.RULEKIT_MAX_HEAP_SIZE
    )
    initialize_experiment(experiment_config)

    if config.CV_ENABLED and 'monk' not in experiment_config.dataset_name:
        run_expriment_with_cross_validation(experiment_config)
    else:
        run_experiment_train_test(experiment_config)
    finish_experiment(experiment_config)


if __name__ == '__main__':
    paramsets = [e[1]['experiment_config']
                 for e in config.generate_paramsets()]
    results = Parallel(n_jobs=4)(delayed(main)(paramset) for paramset in paramsets)
