from logging import getLogger
import _
import os
import pickle
from glob import glob
from joblib import Parallel, delayed
import time
from experiments_utils import *
from experiments_utils.results.tables import *
import config
from utils.rulekit import RuleKit
from utils.rulekit.survival import SurvivalRules
from m_of_n_survival.steps import *
from m_of_n_survival.helpers import *
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

def save_model(model: SurvivalRules, dataset: str, variant: str, model_type: str):
    path = os.path.join(config.RESULTS_BASE_PATH, dataset, variant, model_type)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'model.pickle'), 'wb') as file:
        pickle.dump(model, file)
    


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
    models.plain_rules = SurvivalRules = train_plain_rules(
        datasets.X_train,
        datasets.y_train
    )
    save_model(models.plain_rules, experiment_config.dataset_name, experiment_config.variant, model_type='plain')

    # 2. trenowanie reguł z warunkami złożonymi
    models.complex_rules: SurvivalRules = train_complex_rules(
        datasets.X_train,
        datasets.y_train
    )
    save_model(models.complex_rules, experiment_config.dataset_name, experiment_config.variant, model_type='complex')

    # 3. przygotowanie tablicy binarnej
    survival_times_train = X_train['survival_time']
    survival_times_test = X_test['survival_time']
    X_train_bin, X_test_bin, attributes_mappings = prepare_binary_datasets(
        experiment_config,
        models.complex_rules,
        datasets.X_train.drop('survival_time', axis=1),
        datasets.X_test.drop('survival_time', axis=1),
        datasets.y_train,
        datasets.y_test
    )
    # X_train_bin['survival_time'] = survival_times_train.tolist()
    # X_test_bin['survival_time'] = survival_times_test.tolist()
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
    X_exact_binary_train['survival_time'] = survival_times_train.tolist()
    X_exact_binary_test['survival_time'] = survival_times_test.tolist()

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
    save_model(models.exact_m_of_n_rules, experiment_config.dataset_name, experiment_config.variant, model_type='exact_M-of-N')


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
    X_at_least_binary_train['survival_time'] = survival_times_train.tolist()
    X_at_least_binary_test['survival_time'] = survival_times_test.tolist()

    # 8. trenowanie reguł z warunkami przynajmniej m-of-n
    models.at_least_m_of_n_rules = train_rules_with_m_of_n(
        datasets.X_at_least_binary_train,
        datasets.y_train,
        datasets.attributes_mapping
    )
    models.at_least_m_of_n_rules.model.stats.time_total_s = time.time()
    save_model(models.at_least_m_of_n_rules, experiment_config.dataset_name, experiment_config.variant, model_type='at_least_M-of-N')


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
    paramsets = [e[1]['experiment_config'] for e in config.generate_paramsets()]
    results = Parallel(n_jobs=4)(delayed(main)(paramset) for paramset in paramsets)
