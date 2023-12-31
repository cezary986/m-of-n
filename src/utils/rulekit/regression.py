from typing import Union, Any, List, Tuple
from numbers import Number
import numpy as np
import pandas as pd
from sklearn import metrics
from .helpers import PredictionResultMapper
from .operator import BaseOperator, ExpertKnowledgeOperator, Data, DEFAULT_PARAMS_VALUE
from .params import Measures


class RuleRegressor(BaseOperator):
    """Regression model."""

    def __init__(self,
                 min_rule_covered: int = DEFAULT_PARAMS_VALUE['min_rule_covered'],
                 induction_measure: Measures = DEFAULT_PARAMS_VALUE['induction_measure'],
                 pruning_measure: Union[Measures,
                                        str] = DEFAULT_PARAMS_VALUE['pruning_measure'],
                 voting_measure: Measures = DEFAULT_PARAMS_VALUE['voting_measure'],
                 max_growing: float = DEFAULT_PARAMS_VALUE['max_growing'],
                 enable_pruning: bool = DEFAULT_PARAMS_VALUE['enable_pruning'],
                 ignore_missing: bool = DEFAULT_PARAMS_VALUE['ignore_missing'],
                 max_uncovered_fraction: float = DEFAULT_PARAMS_VALUE['max_uncovered_fraction'],
                 select_best_candidate: bool = DEFAULT_PARAMS_VALUE['select_best_candidate']):
        """
        Parameters
        ----------
        min_rule_covered : int = 5
            positive integer representing minimum number of previously uncovered examples to be covered by a new rule
            (positive examples for classification problems); default: 5
        induction_measure : :class:`rulekit.params.Measures` = :class:`rulekit.params.Measures.Correlation`
            measure used during induction; default measure is correlation
        pruning_measure : Union[:class:`rulekit.params.Measures`, str] = :class:`rulekit.params.Measures.Correlation`
            measure used during pruning. Could be user defined (string), for example  :code:`2 * p / n`;
            default measure is correlation
        voting_measure : :class:`rulekit.params.Measures` = :class:`rulekit.params.Measures.Correlation`
            measure used during voting; default measure is correlation
        max_growing : int = 0.0
            non-negative integer representing maximum number of conditions which can be added to the rule in the growing phase
            (use this parameter for large datasets if execution time is prohibitive); 0 indicates no limit; default: 0,
        enable_pruning : bool = True
            enable or disable pruning, default is True.
        ignore_missing : bool = False
            boolean telling whether missing values should be ignored (by default, a missing value of given attribute is always
            considered as not fulfilling the condition build upon that attribute); default: False.
        max_uncovered_fraction : float = 0.0
            Floating-point number from [0,1] interval representing maximum fraction of examples that may remain uncovered by the rule set, default: 0.0.
        select_best_candidate : bool = False
            Flag determining if best candidate should be selected from growing phase; default: False.
        """
        super().__init__(
            min_rule_covered=min_rule_covered,
            induction_measure=induction_measure,
            pruning_measure=pruning_measure,
            voting_measure=voting_measure,
            max_growing=max_growing,
            enable_pruning=enable_pruning,
            ignore_missing=ignore_missing,
            max_uncovered_fraction=max_uncovered_fraction,
            select_best_candidate=select_best_candidate)

    def fit(self, values: Data, labels: Data, attributes_mappings: dict[str, List[str]] = None) -> Any:
        """Train model on given dataset.

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes
        labels : :class:`rulekit.operator.Data`
            target values
        Returns
        -------
        self : RuleRegressor
        """
        if isinstance(labels, (pd.DataFrame, pd.Series)):
            first_label = labels.iloc[0]
        else:
            first_label = labels[0]
        if not isinstance(first_label, Number):
            raise ValueError(
                'DecisionTreeRegressor requires lables values to be numeric')
        super().fit(values, labels, attributes_mappings=attributes_mappings)
        return self

    def predict(self, values: Data) -> np.ndarray:
        """Perform prediction and returns predicted values.

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes

        Returns
        -------
        result : np.ndarray
            predicted values
        """
        return self._map_result(super().predict(values))

    def score(self, values: Data, labels: Data) -> float:
        """Return the coefficient of determination R2 of the prediction

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes
        labels : :class:`rulekit.operator.Data`
            true target values

        Returns
        -------
        score : float
            R2 of self.predict(values) wrt. labels.
        """
        predicted_labels = self.predict(values)
        return metrics.r2_score(labels, predicted_labels)

    def _map_result(self, predicted_example_set) -> np.ndarray:
        return PredictionResultMapper.map_to_numerical(predicted_example_set, remap=False)


class ExpertRuleRegressor(ExpertKnowledgeOperator, RuleRegressor):
    """Expert Regression model."""

    def __init__(self,
                 min_rule_covered: int = DEFAULT_PARAMS_VALUE['min_rule_covered'],
                 induction_measure: Measures = DEFAULT_PARAMS_VALUE['induction_measure'],
                 pruning_measure: Union[Measures,
                                        str] = DEFAULT_PARAMS_VALUE['pruning_measure'],
                 voting_measure: Measures = DEFAULT_PARAMS_VALUE['voting_measure'],
                 max_growing: float = DEFAULT_PARAMS_VALUE['max_growing'],
                 enable_pruning: bool = DEFAULT_PARAMS_VALUE['enable_pruning'],
                 ignore_missing: bool = DEFAULT_PARAMS_VALUE['ignore_missing'],
                 max_uncovered_fraction: float = DEFAULT_PARAMS_VALUE['max_uncovered_fraction'],
                 select_best_candidate: bool = DEFAULT_PARAMS_VALUE['select_best_candidate'],

                 extend_using_preferred: bool = DEFAULT_PARAMS_VALUE['extend_using_preferred'],
                 extend_using_automatic: bool = DEFAULT_PARAMS_VALUE['extend_using_automatic'],
                 induce_using_preferred: bool = DEFAULT_PARAMS_VALUE['induce_using_preferred'],
                 induce_using_automatic: bool = DEFAULT_PARAMS_VALUE['induce_using_automatic'],
                 preferred_conditions_per_rule: int = DEFAULT_PARAMS_VALUE[
                     'preferred_conditions_per_rule'],
                 preferred_attributes_per_rule: int = DEFAULT_PARAMS_VALUE['preferred_attributes_per_rule']):
        """
        Parameters
        ----------
        min_rule_covered : int = 5
            positive integer representing minimum number of previously uncovered examples to be covered by a new rule
            (positive examples for classification problems); default: 5
        induction_measure : :class:`rulekit.params.Measures` = :class:`rulekit.params.Measures.Correlation`
            measure used during induction; default measure is correlation
        pruning_measure : Union[:class:`rulekit.params.Measures`, str] = :class:`rulekit.params.Measures.Correlation`
            measure used during pruning. Could be user defined (string), for example  :code:`2 * p / n`;
            default measure is correlation
        voting_measure : :class:`rulekit.params.Measures` = :class:`rulekit.params.Measures.Correlation`
            measure used during voting; default measure is correlation
        max_growing : int = 0.0
            non-negative integer representing maximum number of conditions which can be added to the rule in the growing phase
            (use this parameter for large datasets if execution time is prohibitive); 0 indicates no limit; default: 0,
        enable_pruning : bool = True
            enable or disable pruning, default is True.
        ignore_missing : bool = False
            boolean telling whether missing values should be ignored (by default, a missing value of given attribute is always
            considered as not fulfilling the condition build upon that attribute); default: False.
        max_uncovered_fraction : float = 0.0
            Floating-point number from [0,1] interval representing maximum fraction of examples that may remain uncovered by the rule set, default: 0.0.
        select_best_candidate : bool = False
            Flag determining if best candidate should be selected from growing phase; default: False.

        extend_using_preferred : bool = False
            boolean indicating whether initial rules should be extended with a use of preferred conditions and attributes; default is False
        extend_using_automatic : bool = False
            boolean indicating whether initial rules should be extended with a use of automatic conditions and attributes; default is False
        induce_using_preferred : bool = False
            boolean indicating whether new rules should be induced with a use of preferred conditions and attributes; default is False
        induce_using_automatic : bool = False
            boolean indicating whether new rules should be induced with a use of automatic conditions and attributes; default is False
        preferred_conditions_per_rule : int = None
            maximum number of preferred conditions per rule; default: unlimited,
        preferred_attributes_per_rule : int = None
            maximum number of preferred attributes per rule; default: unlimited.
        """
        ExpertKnowledgeOperator.__init__(
            self,
            min_rule_covered=min_rule_covered,
            induction_measure=induction_measure,
            pruning_measure=pruning_measure,
            voting_measure=voting_measure,
            max_growing=max_growing,
            enable_pruning=enable_pruning,
            ignore_missing=ignore_missing,
            max_uncovered_fraction=max_uncovered_fraction,
            select_best_candidate=select_best_candidate,
            extend_using_preferred=extend_using_preferred,
            extend_using_automatic=extend_using_automatic,
            induce_using_preferred=induce_using_preferred,
            induce_using_automatic=induce_using_automatic,
            preferred_conditions_per_rule=preferred_conditions_per_rule,
            preferred_attributes_per_rule=preferred_attributes_per_rule
        )

    def fit(self,
            values: Data,
            labels: Data,

            expert_rules: List[Union[str, Tuple[str, str]]] = None,
            expert_preferred_conditions: List[Union[str,
                                                    Tuple[str, str]]] = None,
            expert_forbidden_conditions: List[Union[str, Tuple[str, str]]] = None) -> Any:
        """Train model on given dataset.

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes
        labels : :class:`rulekit.operator.Data`
            target values

        expert_rules : List[Union[str, Tuple[str, str]]]
             set of initial rules, either passed as a list of strings representing rules or as list of tuples where first
             element is name of the rule and second one is rule string.
        expert_preferred_conditions : List[Union[str, Tuple[str, str]]]
             multiset of preferred conditions (used also for specifying preferred attributes by using special value Any). Either passed as a list of strings representing rules or as list of tuples where first
             element is name of the rule and second one is rule string.
        expert_forbidden_conditions : List[Union[str, Tuple[str, str]]]
             set of forbidden conditions (used also for specifying forbidden attributes by using special valye Any). Either passed as a list of strings representing rules or as list of tuples where first
             element is name of the rule and second one is rule string.
        Returns
        -------
        self : ExpertRuleRegressor
        """
        if isinstance(labels, (pd.DataFrame, pd.Series)):
            first_label = labels.iloc[0]
        else:
            first_label = labels[0]
        if not isinstance(first_label, Number):
            raise ValueError(
                'ExpertRuleRegressor requires lables values to be numeric')
        return ExpertKnowledgeOperator.fit(
            self,
            values,
            labels,
            expert_rules=expert_rules,
            expert_preferred_conditions=expert_preferred_conditions,
            expert_forbidden_conditions=expert_forbidden_conditions
        )

    def predict(self, values: Data) -> np.ndarray:
        return self._map_result(ExpertKnowledgeOperator.predict(self, values))
