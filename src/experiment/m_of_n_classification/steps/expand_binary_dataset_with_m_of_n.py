from logging import Logger, getLogger
from dataclasses import dataclass
from m_of_n_classification.helpers import *
from experiments_utils import step
from experiments_utils.store import Store
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from mlxtend.frequent_patterns import fpgrowth
from itertools import combinations
from utils.rulekit.classification import RuleClassifier
from m_of_n_classification import config


def find_frequent_sets(X_train_bin, y_train) -> pd.DataFrame:
    min_support: float = config.MIN_SUPP_FRACTION

    getLogger().info(
        f'Szukanie zbiorów częstych (min_support = {min_support})')

    fsets: pd.DataFrame = fpgrowth(
        X_train_bin,
        min_support=min_support,
        max_len=config.M,
        use_colnames=True
    )
    # filter only fsets with two items
    fsets = fsets[fsets['itemsets'].map(len) == config.M]
    getLogger().info(
        f'Znaleziono {fsets.shape[0]} zbiorów częstych')
    return fsets


class MapItem():

    def __init__(self) -> None:
        self.items: list = []
        self.supports: dict[str, float] = {}


def prepare_candidates(
    fsets: pd.DataFrame,
    elementary_conditions_count: int
) -> List[Tuple[str]]:
    items_map: Dict[str, MapItem] = {}

    for _, row in fsets.iterrows():
        fset: set = row['itemsets']
        fset_support = row['support']
        for key in list(combinations(fset, r=config.M - 1)):
            if key not in items_map:
                items_map[key] = MapItem()

            item = items_map[key]
            tmp_set = set(fset)
            for e in key:
                tmp_set.remove(e)
            tmp_item = tmp_set.pop()
            item.items.append(tmp_item)
            item.supports[tmp_item] = fset_support

    candidates: List[List[str]] = []
    for key_item, map_item in items_map.items():
        map_item: MapItem = map_item
        new_candidates = list(combinations(map_item.items, r=config.N - 1))
        mapped_candidates = []
        candidates_to_remove = []
        for new_candidate in new_candidates:
            # make sure attrubute appears only once in conditions
            items_attributes = {}
            new_candidate = list(new_candidate)
            new_candidate += list(key_item)
            mapped_candidate = {
                'candidate': new_candidate,
                'avg_support': (
                    (map_item.supports[new_candidate[0]] +
                     map_item.supports[new_candidate[1]]) / 2
                )
            }
            mapped_candidates.append(mapped_candidate)
            for item in new_candidate:
                attribute: str = item.split(' ')[0].strip().split('=')[0]
                if attribute in items_attributes:
                    candidates_to_remove.append(mapped_candidate)
                    break
                else:
                    items_attributes[attribute] = True
        for e in candidates_to_remove:
            mapped_candidates.remove(e)
        candidates += mapped_candidates

    candidates = sorted(
        candidates, key=lambda k: k['avg_support'], reverse=True)

    limit: int = 10000# config.N * elementary_conditions_count

    if limit > config.MAX_CANDIDATES and len(candidates) > limit:
        candidates = candidates[:config.MAX_CANDIDATES]
    elif limit < config.MIN_CANDIDATES and len(candidates) > limit:
        candidates = candidates[:config.MIN_CANDIDATES]
    else:
        candidates = candidates[:limit]

    candidates = [
        e['candidate'] for e in candidates
    ]

    getLogger().info(f'{len(candidates)} kandydatów na M-of-N')
    return candidates


def expand_binary_dataset(
    df: pd.DataFrame,
    m_of_n_candidates: List[Tuple[str]],
    exact_m_of_n: bool,
    attributes_mappings: Dict[str, List[str]] = None,
) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    columns = sorted(df.columns.tolist())
    data_np: np.ndarray = df[columns].to_numpy()
    columns_indices: Dict[str, int] = {
        col: i for i, col in enumerate(columns)
    }

    column_names = []
    masks = []
    for candidate in m_of_n_candidates:
        mask = None
        # rozbicie warunku M-of-N na alternatywe koniunkcji
        for i in range(0, config.N - 1):
            for j in range(1, config.N):
                if i == j:
                    continue
                tmp2 = [i, j]
                submask = (data_np[:, columns_indices[candidate[i]]] == '1')
                submask &= (data_np[:, columns_indices[candidate[j]]] == '1')
                if exact_m_of_n:
                    for k in candidate:
                        if k != candidate[i] and k != candidate[j]:
                            tmp2.append(columns_indices[k])
                            submask &= (data_np[:, columns_indices[k]] == '0')
                if mask is None:
                    mask = submask
                else:
                    mask |= submask
        column_name: str = f'{config.M}-of-{config.N}({", ".join(candidate)})'
        if attributes_mappings is not None:
            m_of_n_attr_mappings = []
            for item in candidate:
                m_of_n_attr_mappings += attributes_mappings[item]
            attributes_mappings[column_name] = m_of_n_attr_mappings
        column_names.append(column_name)
        masks.append(mask)

    df = pd.concat([
        df,
        pd.DataFrame({
            column_name: mask for column_name, mask in zip(column_names, masks)
        }, index=df.index)
    ], axis=1)

    return df.astype(int).astype(str)


def expand_train_dataset(
    X_train,
    m_of_n_candidate,
    exact_m_of_n: bool,
    attributes_mappings: Dict[str, List[str]]
) -> pd.DataFrame:
    return expand_binary_dataset(X_train, m_of_n_candidate, exact_m_of_n, attributes_mappings)


def expand_test_dataset(
    X_test,
    m_of_n_candidate,
    exact_m_of_n: bool,
) -> pd.DataFrame:
    return expand_binary_dataset(X_test, m_of_n_candidate, exact_m_of_n)


def find_m_of_n_candidates(
    experiment_datasets: ExperimentDatasets,
    exact_m_of_n: bool,
    logger: Logger,
):
    # Wyszkianie zbiorów częstych

    fsets: pd.DataFrame = find_frequent_sets(
        experiment_datasets.X_bin_train.dropna().astype(int),
        experiment_datasets.y_train
    )
    #  Generowanie trójek
    candidates: List[Tuple[str]] = prepare_candidates(
        fsets, experiment_datasets.X_bin_train.shape[1])
    attributes_mapping: dict = experiment_datasets.attributes_mapping

    # Rozszerzenie tablicy binarej o nowe warunki
    X_bin_ext_train = expand_train_dataset(
        experiment_datasets.X_bin_train.copy(),
        candidates,
        exact_m_of_n,
        attributes_mapping
    )
    logger.info(f'Liczba kolumn po roszerzeniu: {X_bin_ext_train.shape[1]}')

    X_bin_ext_test = expand_test_dataset(
        experiment_datasets.X_bin_test.copy(),
        candidates,
        exact_m_of_n
    )
    return X_bin_ext_train, X_bin_ext_test, attributes_mapping
