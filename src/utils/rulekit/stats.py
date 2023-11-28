from typing import Any, Dict, List


def _dict_to_str(dictionary: dict, level: int = 0) -> str:
    string = ''
    for key, value in dictionary.items():
        if isinstance(value, dict):
            string += ('    ' * level) + f'{key}:\n'
            string += _dict_to_str(value, level=level + 1)
        else:
            string += ('    ' * level) + f'{key}: {value}\n'
    return string


def _sum_dictionaries(dictionary_1: dict, dictionary_2: dict):
    for key, value in dictionary_2.items():
        if isinstance(value, dict):
            _sum_dictionaries(dictionary_1[key], value)
        else:
            dictionary_1[key] += value


class RuleConditionsStatistics:

    def __init__(self, rule) -> None:
        self.stats: dict = {
            'Plain conditions': 0,
            'Numerical attributes conditions': 0,
            'Nominal attributes conditions': 0,
            'Discrete set conditions': 0,
            'Interval conditions': 0,
            'Inner alternatives conditions': 0,
            'Inner alternatives statistics': {
                'Plain conditions': 0,
                'Numerical attributes conditions': 0,
                'Nominal attributes conditions': 0,
                'Discrete set conditions': 0,
                'Interval conditions': 0,
            }
        }
        premise = rule._java_object.getPremise()
        for subcondition in premise.getSubconditions():
            condition_class_name = subcondition.getClass().getSimpleName()
            if condition_class_name == 'CompoundCondition':
                self._process_compound_condition(subcondition)
            else:
                self._process_elementary_condition(subcondition)

    def _process_compound_condition(self, condition: Any) -> None:
        self.stats['Inner alternatives conditions'] += 1
        for subcondition in condition.getSubconditions():
            condition_class_name = subcondition.getClass().getSimpleName()
            if condition_class_name == 'CompoundCondition':
                self._process_compound_condition(subcondition)
            else:
                self._process_elementary_condition(
                    subcondition, parent=condition)

    def _process_elementary_condition(self, condition: Any, parent: Any = None) -> None:
        parent_class_name = None if parent is None else parent.getClass().getSimpleName()
        condition_class_name = condition.getClass().getSimpleName()

        condition_type: str
        if condition_class_name == 'AttributesCondition':
            condition_type = 'Numerical attributes conditions'
        if condition_class_name == 'AttributesCondition':
            condition_type = 'Numerical attributes conditions'
        if condition_class_name == 'NominalAttributesCondition':
            condition_type = 'Nominal attributes conditions'
        if condition_class_name == 'ElementaryCondition':
            value_set_class_name = condition.getValueSet().getClass().getSimpleName()
            if value_set_class_name == 'DiscreteSet':
                condition_type = 'Discrete set conditions'
            if value_set_class_name == 'SingletonSet':
                condition_type = 'Plain conditions'
            if value_set_class_name == 'Universum':
                condition_type = 'Plain conditions'
            if value_set_class_name == 'Interval':
                if hasattr(condition.getValueSet(), 'isRealInterval'):
                    is_real_interval = condition.getValueSet().isRealInterval()
                else:
                    is_real_interval = False
                if is_real_interval:
                    condition_type = 'Interval conditions'
                else:
                    condition_type = 'Plain conditions'
        if parent_class_name == 'CompoundCondition':
            self.stats['Inner alternatives statistics'][condition_type] += 1
        self.stats[condition_type] += 1

    def __str__(self) -> str:
        return _dict_to_str(self.stats)


class RuleSetConditionsStatistics:

    def __init__(self, ruleset) -> None:
        self.rules_stats: List[RuleConditionsStatistics] = [
            RuleConditionsStatistics(rule) for rule in ruleset.rules
        ]
        self.stats: Dict[str, int] = None
        for rule_stats in self.rules_stats:
            if self.stats is None:
                self.stats = rule_stats.stats
            else:
                _sum_dictionaries(self.stats, rule_stats.stats)

    def __str__(self) -> str:
        return _dict_to_str(self.stats)


class RuleStatistics:
    """Statistics for single rule.

    Attributes
    ----------
    p : float
        Number of positives covered by the rule (accounting weights).
    n : float
        Number of negatives covered by the rule (accounting weights).
    P : float
        Number of positives in the training set (accounting weights).
    N : float
        Number of negatives in the training set (accounting weights).
    weight : float
        Rule weight.
    pvalue : float
        Rule significance.
    """

    def __init__(self, rule):
        self.p = rule.weighted_p
        self.n = rule.weighted_n
        self.P = rule.weighted_P
        self.N = rule.weighted_N
        self.weight = rule.weight
        self.pvalue = rule.pvalue

    def __str__(self):
        """Returns string representation of the object."""
        return f'(p = {self.p}, n = {self.n}, P = {self.P}, ' + \
               f'N = {self.N}, weight = {self.weight}, pvalue = {self.pvalue})'


class RuleSetStatistics:
    """Statistics for ruleset.

    Attributes
    ----------
    SIGNIFICANCE_LEVEL : float
        Significance level, default value is *0.05*


    time_total_s : float
        Time of constructing the rule set in seconds.
    time_growing_s : float
        Time of growing in seconds.
    time_pruning_s : float
        Time of pruning in seconds.
    rules_count : int
        Number of rules in ruleset.
    conditions_per_rule : float
        Average number of conditions per rule.
    induced_conditions_per_rule : float
        Average number of induced conditions.
    avg_rule_coverage : float
        Average rule coverage.
    avg_rule_precision : float
        Average rule precision.
    avg_rule_quality : float
        Average rule quality.
    pvalue : float
        rule set significance.
    FDR_pvalue : float
        Significance of the rule set with false discovery rate correction.
    FWER_pvalue : float
        Significance of the rule set with familiy-wise error rate correction.
    fraction_significant : float
        Fraction of rules significant at assumed level
    fraction_FDR_significant : float
        Fraction of rules significant, set with false discovery rate correction, at assumed level.
    fraction_FWER_significant : float
        Fraction of rules significant, set with familiy-wise error rate correction, at assumed level.
    """
    SIGNIFICANCE_LEVEL = 0.05

    def __init__(self, ruleset):
        self._ruleset = ruleset
        self.time_total_s = ruleset.total_time
        self.time_growing_s = ruleset.growing_time
        self.time_pruning_s = ruleset.pruning_time

        self.rules_count = len(ruleset.rules)
        self.conditions_per_rule = ruleset.calculate_conditions_count()
        self.induced_conditions_per_rule = ruleset.calculate_induced_conditions_count()

        self.avg_rule_coverage = ruleset.calculate_avg_rule_coverage()
        self.avg_rule_precision = ruleset.calculate_avg_rule_precision()
        self.avg_rule_quality = ruleset.calculate_avg_rule_quality()

        self.pvalue = ruleset.calculate_significance(
            RuleSetStatistics.SIGNIFICANCE_LEVEL)['p']
        self.FDR_pvalue = ruleset.calculate_significance_fdr(
            RuleSetStatistics.SIGNIFICANCE_LEVEL)['p']
        self.FWER_pvalue = ruleset.calculate_significance_fwer(
            RuleSetStatistics.SIGNIFICANCE_LEVEL)['p']

        self.fraction_significant = ruleset.calculate_significance(
            RuleSetStatistics.SIGNIFICANCE_LEVEL)['fraction']
        self.fraction_FDR_significant = ruleset.calculate_significance_fdr(RuleSetStatistics.SIGNIFICANCE_LEVEL)[
            'fraction']
        self.fraction_FWER_significant = ruleset.calculate_significance_fwer(RuleSetStatistics.SIGNIFICANCE_LEVEL)[
            'fraction']
        self._conditions_stats: Dict[str, int] = None

    @property
    def conditions_stats(self) -> RuleSetConditionsStatistics:
        if self._conditions_stats is None:
            self._conditions_stats = RuleSetConditionsStatistics(self._ruleset)
        return self._conditions_stats

    def __str__(self):
        stats_str = '\n'.join([
            f'{key}: {value}' for key, value in self.to_json().items()
        ])
        if self.conditions_stats.stats is not None:
            stats_str += f'\nConditions stats:\n' + \
                _dict_to_str(self.conditions_stats.stats, level=1)
        return stats_str

    def to_json(self) -> dict:
        dictionary = {
            'Time total [s]': self.time_total_s,
            'Time growing [s]': self.time_growing_s,
            'Time pruning [s]': self.time_pruning_s,

            'Rules count': self.rules_count,
            'Conditions per rule': self.conditions_per_rule,
            'Induced conditions per rule': self.induced_conditions_per_rule,
            'Average rule coverage': self.avg_rule_coverage,
            'Average rule precision': self.avg_rule_precision,
            'Average rule quality': self.avg_rule_quality,
            'pvalue': self.pvalue,
            'FDR pvalue': self.FDR_pvalue,
            'FWER pvalue': self.FWER_pvalue,
            f'Fraction {RuleSetStatistics.SIGNIFICANCE_LEVEL} significant': self.fraction_significant,
            f'Fraction {RuleSetStatistics.SIGNIFICANCE_LEVEL} FDR significant': self.fraction_FDR_significant,
            f'Fraction {RuleSetStatistics.SIGNIFICANCE_LEVEL} FWER significant': self.fraction_FWER_significant
        }
        if self.conditions_stats.stats is not None:
            dictionary = {
                **dictionary,
                'Conditions stats': self.conditions_stats.stats
            }
