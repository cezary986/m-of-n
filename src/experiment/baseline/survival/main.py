from typing import Any
import pandas as pd
import numpy as np
from sksurv import tree
from sklearn import metrics
from sklearn import preprocessing
from experiments_utils import experiment, Store, step
from experiments_utils import settings
from sksurv.metrics import integrated_brier_score
from experiments_utils.results.tables import *
import sys
sys.path.append('../../')
sys.path.append('../../m_of_n_survival')
sys.path.append('../../../')
from m_of_n_survival.steps.load_data import load_data as _load_data  # nopep8
from m_of_n_survival.steps.produce_results import _string_supporting_mean, _string_supporting_std  # nopep8
import m_of_n_survival.config as config  # nopep8

VERSION: str = '1.0.0'
dir_path = '.'
settings.EXPERIMENT_BASE_LOGGING_DIR = f'{dir_path}/logs'
config.RESULTS_BASE_PATH: str = f'{dir_path}/results/{VERSION}'


def _adjust_data(X_train, y_train, X_test, y_test, model):
    lower, upper = np.percentile(y_test["time"], [10, 90])
    times = np.arange(
        lower, upper
    )  # upper + 1 results in max vlue greater than in test dataset

    # adjust times, X_test, and y_test
    y_train_min = min(y_train[y_train["cens"]]["time"])
    y_train_max = max(y_train[y_train["cens"]]["time"])
    X_test = X_test[(y_test["time"] < y_train_max) &
                    (y_test["time"] > y_train_min)]
    y_test = y_test[(y_test["time"] < y_train_max) &
                    (y_test["time"] > y_train_min)]
    y_test_min = min(y_test[y_test["cens"]]["time"])
    y_test_max = max(y_test[y_test["cens"]]["time"])
    times = times[(times < y_test_max) & (times > y_test_min)]

    y_train_proba = np.row_stack(
        [fn(times) for fn in model.predict_survival_function(X_train)]
    )

    y_test_proba = np.row_stack(
        [fn(times) for fn in model.predict_survival_function(X_test)]
    )

    return y_train, y_test, y_train_proba, y_test_proba, times


def load_data(
    dataset_name,
    fold_number,
    discretization_enabled,
    logger
):
    X_train, y_train, X_test, y_test = _load_data(
        dataset_name,
        fold_number=fold_number,
        discretization_enabled=discretization_enabled,
        logger=logger
    )
    y_train = y_train.astype(bool)
    y_test = y_test.astype(bool)
    survival_times_train = X_train['survival_time'].to_numpy()
    survival_times_test = X_test['survival_time'].to_numpy()
    # decision trees require all features to be numerical - one hot encode
    oh = preprocessing.OneHotEncoder(handle_unknown='ignore')
    X_train = oh.fit_transform(X_train)

    X_test = oh.transform(X_test)
    y_train_ = np.empty(len(y_train), dtype=object)
    y_test_ = np.empty(len(y_test), dtype=object)
    y_train_[:] = [
        (survival_status, survival_times_train[i])
        for i, survival_status in enumerate(y_train.tolist())
    ]
    y_test_[:] = [
        (survival_status, survival_times_test[i])
        for i, survival_status in enumerate(y_test.tolist())
    ]
    y_test_ = y_test_.astype(dtype=[('cens', '?'), ('time', '<f8')])
    y_train_ = y_train_.astype(dtype=[('cens', '?'), ('time', '<f8')])
    return X_train, y_train_, X_test, y_test_, survival_times_train, survival_times_test


def get_node_depths(tree):
    def get_node_depths_(current_node, current_depth, l, r, depths):
        depths += [current_depth]
        if l[current_node] != -1 and r[current_node] != -1:
            get_node_depths_(l[current_node], current_depth + 1, l, r, depths)
            get_node_depths_(r[current_node], current_depth + 1, l, r, depths)

    depths = []
    get_node_depths_(0, 0, tree.children_left, tree.children_right, depths)
    return np.array(depths)


@step()
def train_model(X_train, y_train) -> Any:
    clf = tree.SurvivalTree()
    clf.fit(X_train, y_train)
    return clf


@step()
def evaluate_model(
    model: tree.SurvivalTree,
    data: tuple
) -> Any:
    X_train, y_train, X_test, y_test, times_train, times_test = data
    tree_depths = get_node_depths(model.tree_)

    times_train = np.unique(times_train)
    times_test = np.unique(times_test)
    times_test = times_test[(times_test >= times_train.min()) & (
        times_test <= times_train.max())]

    y_train, y_test, y_train_proba, y_test_proba, times = _adjust_data(
        X_train, y_train, X_test, y_test, model
    )
    ibs_train = integrated_brier_score(
        y_train, y_train, y_train_proba, times
    )
    ibs_test = integrated_brier_score(
        y_train, y_test, y_test_proba, times
    )

    return pd.DataFrame([{
        'dataset': evaluate_model.paramset_name,
        'integrated_brier_score (train)': ibs_train,
        'integrated_brier_score (test)': ibs_test,
        'rules': int(model.tree_.n_outputs),
        'conditions_count': model.tree_.node_count,
        'avg conditions per rule': tree_depths.mean(),
    }])


def run_experiment_for_cv_fold(
    fold_index: int,
    data: tuple,
    dataset_name: str,
    store: Store
):
    models = []
    X_train, y_train = data[0:2]
    model = train_model(X_train, y_train)
    models.append(model)
    store.models = models
    metrics_df: pd.DataFrame = evaluate_model(model, data)
    metrics_path = [dataset_name, 'cv']
    if fold_index is not None:
        metrics_path.append(str(fold_index))
    else:
        metrics_path.append('1')
    metrics_path.append('metrics')
    metrics_table = Tables.get(*metrics_path)
    metrics_table.set_df(metrics_df)
    metrics_table.save()


def aggregate_cv_results(
    dataset: str
):
    df: pd.DataFrame = pd.concat(
        Tables.query(dataset, 'cv', '[!f]', 'metrics', as_pandas=True)
    )
    df_summary = df.groupby(['dataset']).agg(
        _string_supporting_mean)
    df_std = df.groupby(['dataset']).agg(
        _string_supporting_std).reset_index(drop=True)

    columns = []
    for column in df_std.columns.tolist():
        if pd.api.types.is_numeric_dtype(df[column]):
            columns.append(column)
            columns.append(f'{column} (std)')
            df_summary[f'{column} (std)'] = df_std[column].tolist()
    df_summary['dataset'] = dataset
    summary_results_table = Tables.get(
        dataset, 'metrics')
    summary_results_table.set_df(df_summary.reset_index(drop=True))
    summary_results_table.save()

    results_table = Tables.get(dataset, 'cv_metrics')
    results_table.set_df(df)
    results_table.save()


@experiment(
    name='decision_tree_baseline',
    version=VERSION
)
def decision_tree_survival_baseline(
    dataset_name: str
):
    store = Store()
    Tables.configure(config.RESULTS_BASE_PATH)
    for fold_index in range(1, 11):
        data = load_data(
            dataset_name,
            fold_number=fold_index,
            discretization_enabled=False,
            logger=decision_tree_survival_baseline.logger,
        )
        run_experiment_for_cv_fold(fold_index, data, dataset_name, store)
    aggregate_cv_results(dataset_name)


if __name__ == '__main__':
    decision_tree_survival_baseline([
        (dataset_name, {'dataset_name': dataset_name})
        for dataset_name in config.DATASETS
    ])
