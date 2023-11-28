from typing import Any
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import metrics
from sklearn import preprocessing
from experiments_utils import experiment, Store, step
from experiments_utils import settings
from experiments_utils.results.tables import *
import sys
sys.path.append('./src/')
sys.path.append('./src/experiment')
sys.path.append('./src/experiment/m_of_n_classification')
sys.path.append('../../')
sys.path.append('../../m_of_n_classification')
sys.path.append('../../../')
from m_of_n_classification.steps.load_data import load_data as _load_data  # nopep8
from m_of_n_classification.steps.produce_results import _string_supporting_mean, _string_supporting_std  # nopep8
import m_of_n_classification.config as config  # nopep8


VERSION: str = '1.0.0'
# dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = './'
settings.EXPERIMENT_BASE_LOGGING_DIR = f'{dir_path}/logs'

config.RESULTS_BASE_PATH: str = f'{dir_path}/results/{VERSION}'


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
    # decision trees require all features to be numerical - one hot encode
    oh = preprocessing.OneHotEncoder(handle_unknown='ignore')
    X_train = oh.fit_transform(X_train)
    X_test = oh.transform(X_test)

    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    return X_train, y_train, X_test, y_test


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
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    return clf


@step()
def evaluate_model(
    model: tree.DecisionTreeClassifier,
    data: tuple
) -> Any:
    X_train, y_train, X_test, y_test = data
    tree_depths = get_node_depths(model.tree_)
    return pd.DataFrame([{
        'dataset': evaluate_model.paramset_name,
        'BAcc (test)': metrics.balanced_accuracy_score(y_test, model.predict(X_test)),
        'BAcc (train)': metrics.balanced_accuracy_score(y_train, model.predict(X_train)),
        'Acc (test)': metrics.accuracy_score(y_test, model.predict(X_test)),
        'Acc (train)': metrics.accuracy_score(y_train, model.predict(X_train)),
        'rules': int(model.get_n_leaves()),
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
    X_train, y_train, X_test, y_test = data
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
def decision_tree_classification_baseline(
    dataset_name: str
):
    store = Store()
    Tables.configure(config.RESULTS_BASE_PATH)
    if 'monk' not in dataset_name:
        for fold_index in range(1, 11):
            data = load_data(
                dataset_name,
                fold_number=fold_index,
                discretization_enabled=False,
                logger=decision_tree_classification_baseline.logger,
            )
            run_experiment_for_cv_fold(fold_index, data, dataset_name, store)
    else:
        data = load_data(
            dataset_name,
            fold_number=None,
            discretization_enabled=False,
            logger=decision_tree_classification_baseline.logger,
        )
        run_experiment_for_cv_fold(None, data, dataset_name, store)

    aggregate_cv_results(dataset_name)


if __name__ == '__main__':
    decision_tree_classification_baseline([
        (dataset_name, {'dataset_name': dataset_name})
        for dataset_name in config.DATASETS
    ])
