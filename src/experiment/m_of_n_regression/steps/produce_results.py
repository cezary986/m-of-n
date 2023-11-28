from logging import getLogger
import os
from pandas.api.types import is_numeric_dtype
from experiments_utils import *
from experiments_utils.results.tables import *
from m_of_n_regression.helpers import *
from m_of_n_regression import config
import pandas as pd
from utils.rulekit.regression import RuleRegressor
from sklearn import metrics


def write_rules_to_file(clf: RuleRegressor, file_path: str):
    with open(os.path.join(file_path), 'w+') as f:
        f.write(f'Rules qualities:\n')
        for i, rule in enumerate(clf.model.rules):
            f.write(f'r{i + 1}: {rule.weight}\n')
        f.write(f'________________________________\n')
        for i, rule in enumerate(clf.model.rules):
            f.write(
                f'r{i + 1}: {str(rule)} (p={rule.weighted_p}, n={rule.weighted_n}, P={rule.weighted_P}, N={rule.weighted_N})\n')


def generate_results(
    experiment_config: ExperimentConfig,
    experiment_models: ExperimentModels,
    experiment_datasets: ExperimentDatasets,
    fold_index: int = None
):
    # zapisanie wyników
    def produce_model_results(
            model_type: str,
            clf: RuleRegressor,
            _X_train: pd.DataFrame,
            _y_train: pd.Series,
            _X_test: pd.DataFrame,
            _y_test: pd.DataFrame) -> dict:

        prediction_train = clf.predict(_X_train)
        prediction_test = clf.predict(_X_test)

        tmp = ' '.join(
            list(map(lambda r: str(r), clf.model.rules)))
        m_of_n_count = tmp.count(f'{config.M}-of-{config.N}')
        getLogger().info(
            f'\n\nWarunki M-of-N pojawiły się: {m_of_n_count} razy w zbiorze reguł'
        )
        if fold_index is not None:
            result_table: Table = Tables.get(
                experiment_config.dataset_name,
                experiment_config.variant,
                model_type,
                'cv',
                str(fold_index),
                'metrics'
            )
        else:
            result_table: Table = Tables.get(
                experiment_config.dataset_name,
                experiment_config.variant,
                model_type,
                'metrics'
            )
        RMSE_test = np.sqrt(metrics.mean_squared_error(_y_test, prediction_test))
        MAE_test = metrics.mean_absolute_error(_y_test, prediction_test)
        MAPE_test = metrics.mean_absolute_percentage_error(_y_test, prediction_test)
        
        RMSE_train = np.sqrt(metrics.mean_squared_error(_y_train, prediction_train))
        MAE_train = metrics.mean_absolute_error(_y_train, prediction_train)
        MAPE_train = metrics.mean_absolute_percentage_error(_y_train, prediction_train)
        
        result_table.rows = [
            {
                'model_type': model_type,
                'variant': experiment_config.variant,
                'dataset': experiment_config.dataset_name,
                'M-of-N count': m_of_n_count,

                'RMSE (test)': RMSE_test,
                'MAE (test)': MAE_test,
                'MAPE (test)': MAPE_test,
                'rRMSE (test)': RMSE_test / np.mean(_y_test),
                'rMAE (test)': MAE_test / np.mean(_y_test),
                'rMAPE (test)': MAPE_test / 100,
                'maxError (test)': metrics.max_error(_y_test, prediction_test),
                'R2 (test)': metrics.r2_score(_y_test, prediction_test),

                'RMSE (train)': RMSE_train,
                'MAE (train)': MAE_train,
                'MAPE (train)': MAPE_train,
                'rRMSE (train)': RMSE_train / np.mean(_y_train),
                'rMAE (train)': MAE_train / np.mean(_y_train),
                'rMAPE (train)': MAPE_train / 100,
                'maxError (train)': metrics.max_error(_y_train, prediction_train),
                'R2 (train)': metrics.r2_score(_y_train, prediction_train),

                'rules':  clf.model.stats.rules_count,
                'conditions_count': clf.model.stats.rules_count * clf.model.stats.conditions_per_rule,
                'avg conditions per rule': clf.model.stats.conditions_per_rule,
                'avg rule quality': clf.model.stats.avg_rule_quality,
                'avg rule precision': clf.model.stats.avg_rule_precision,
                'avg rule coverage': clf.model.stats.avg_rule_coverage,
                'training time total (s)': clf.model.stats.time_total_s,
                'training time growing (s)': clf.model.stats.time_growing_s,
                'training time pruning (s)': clf.model.stats.time_pruning_s,

                'induction measure': clf.get_params()['induction_measure'].value.replace('Measures.', ''),
                'pruning measure': clf.get_params()['pruning_measure'].value.replace('Measures.', ''),
                'voting measure': clf.get_params()['voting_measure'].value.replace('Measures.', ''),
            }
        ]
        result_table.save()
        rules_base_path: str = os.path.join(
            os.path.dirname(result_table._file_path), 'rules'
        )
        os.makedirs(rules_base_path, exist_ok=True)
        write_rules_to_file(
            clf, os.path.join(rules_base_path, 'rules.txt')
        )
        return result_table

    produce_model_results(
        'plain',
        experiment_models.plain_rules,
        experiment_datasets.X_train,
        experiment_datasets.y_train,
        experiment_datasets.X_test,
        experiment_datasets.y_test
    ),
    produce_model_results(
        'complex',
        experiment_models.complex_rules,
        experiment_datasets.X_train,
        experiment_datasets.y_train,
        experiment_datasets.X_test,
        experiment_datasets.y_test
    ),
    produce_model_results(
        'exact_M-of-N',
        experiment_models.exact_m_of_n_rules,
        experiment_datasets.X_exact_binary_train,
        experiment_datasets.y_train,
        experiment_datasets.X_exact_binary_test,
        experiment_datasets.y_test
    )
    produce_model_results(
        'at_least_M-of-N',
        experiment_models.at_least_m_of_n_rules,
        experiment_datasets.X_at_least_binary_train,
        experiment_datasets.y_train,
        experiment_datasets.X_at_least_binary_test,
        experiment_datasets.y_test
    )


def _string_supporting_mean(col):
    if is_numeric_dtype(col):
        return col.mean()
    else:
        return col.unique()[0] if col.nunique() == 1 else np.NaN


def _string_supporting_std(col):
    if is_numeric_dtype(col):
        return col.std()
    else:
        return col.unique()[0] if col.nunique() == 1 else np.NaN


def aggreage_cv_folds_results(
    dataset: str,
    variant: str,
):
    for model_type in ['plain', 'complex', 'exact_M-of-N', 'at_least_M-of-N']:
        df: pd.DataFrame = pd.concat(
            Tables.query(dataset, variant, model_type, 'cv',
                         '[!f]', 'metrics', as_pandas=True)
        )
        df_summary = df.groupby(['variant', 'model_type']).agg(
            _string_supporting_mean)
        df_std = df.groupby(['variant', 'model_type']).agg(
            _string_supporting_std).reset_index(drop=True)

        df_summary['variant'] = variant
        df_summary['model_type'] = model_type

        columns = []
        for column in df_std.columns.tolist():
            if is_numeric_dtype(df[column]):
                columns.append(column)
                columns.append(f'{column} (std)')
                df_summary[f'{column} (std)'] = df_std[column].tolist()

        summary_results_table = Tables.get(
            dataset, variant, model_type, 'metrics')
        summary_results_table.set_df(df_summary.reset_index(drop=True))
        summary_results_table.save()

        results_table = Tables.get(dataset, variant, model_type, 'cv_metrics')
        results_table.set_df(df)
        results_table.save()
