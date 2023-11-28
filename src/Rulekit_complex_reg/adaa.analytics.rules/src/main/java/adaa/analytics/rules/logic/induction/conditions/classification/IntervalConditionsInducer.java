package adaa.analytics.rules.logic.induction.conditions.classification;

import adaa.analytics.rules.logic.induction.ConditionEvaluation;
import adaa.analytics.rules.logic.induction.InductionParameters;
import adaa.analytics.rules.logic.induction.conditions.AbstractConditionsInducer;
import adaa.analytics.rules.logic.induction.conditions.helpers.TotalPosNeg;
import adaa.analytics.rules.logic.representation.*;
import com.rapidminer.example.Attribute;
import com.rapidminer.example.ExampleSet;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.stream.Collectors;

public class IntervalConditionsInducer extends AbstractConditionsInducer {

    private List<Attribute> numericalAttributes;

    public IntervalConditionsInducer(ExecutorService pool, InductionParameters params, Map<Attribute, Map<Double, IntegerBitSet>> precalculatedCoverings, Map<Double, IntegerBitSet> P, Map<Double, IntegerBitSet> N) {
        super(pool, params, precalculatedCoverings, P, N);
    }

    public IntervalConditionsInducer(IntervalConditionsInducer reference) {
        super(reference.pool, reference.params, reference.precalculatedCoverings, reference.P, reference.N);
        this.numericalAttributes = reference.numericalAttributes;
    }

    @Override
    public List<Future> preprocess(ExampleSet trainSet, List<Attribute> numericalAttributes, List<Attribute> nominalAttributes) {
        this.numericalAttributes = numericalAttributes;
        return new ArrayList<>();
    }

    @Override
    public List<Future<Void>> induceConditions(Rule rule, ExampleSet trainSet, double apriori_prec, Set<Integer> uncoveredPositives, Set<Integer> coveredByRule, Map<String, Map<Double, TotalPosNeg>> totals) {
        if (trainSet.getAttributes().getWeight() != null)
            throw new IllegalArgumentException("Interval attributes conditions are not supported for weighted datasets");

        List<Future<Void>> futures = new ArrayList<>();
        double P = rule.getWeighted_P();
        double N = rule.getWeighted_N();
        double classId = ((SingletonSet) rule.getConsequence().getValueSet()).getValue();

        for (Attribute attr : this.numericalAttributes) {
            Future f = pool.submit(() -> {
                Map<Double, TotalPosNeg> attrTotals = totals.get(attr.getName());
                Double[] keys = attrTotals.keySet().stream()
                        .filter(e -> !e.isNaN())
                        .collect(Collectors.toList())
                        .toArray(new Double[attrTotals.size()]);
                Arrays.sort(keys);
                ConditionEvaluation currentBest = new ConditionEvaluation(trainSet);
                for (int i = 1; i < keys.length - 2; i++) {
                    double leftValue = keys[i];

                    IntegerBitSet examplesCoveredByInterval = new IntegerBitSet(trainSet.size());
                    examplesCoveredByInterval.addAll(precalculatedCoverings.get(attr).get(leftValue));

                    Double nextValue = keys[i + 1];
                    examplesCoveredByInterval.addAll(precalculatedCoverings.get(attr).get(nextValue));

                    for (int j = i + 2; j < keys.length - 1; j++) {
                        double rightValue = keys[j];
                        double movedLeftValue = keys[i - 1] + ((keys[i] - keys[i - 1]) / 2);


                        double p = examplesCoveredByInterval.calculateIntersectionSize(rule.getCoveredPositives());
                        int toCover_p = examplesCoveredByInterval.calculateIntersectionSize((IntegerBitSet) coveredByRule, (IntegerBitSet) uncoveredPositives);
                        double n = examplesCoveredByInterval.calculateIntersectionSize((IntegerBitSet) coveredByRule) - p;

                        double precision = p / (p + n);

                        if (precision > apriori_prec) {
                            double quality = params.getInductionMeasure().calculate(p, n, P, N);

                            Interval interval = new Interval(movedLeftValue, rightValue, true, false);
                            interval.setIsRealInterval(true);
                            ElementaryCondition candidateCondition = new ElementaryCondition(attr.getName(), interval);

                            if (checkCandidate(candidateCondition, classId, P, toCover_p)) {
                                ConditionEvaluation candidate = new ConditionEvaluation();
                                candidate.quality = quality;
                                candidate.covered = p;
                                candidate.condition = candidateCondition;
                                candidate.coveredExamples = examplesCoveredByInterval.clone();
                                addCondition(candidate);

                                if (candidate.compareTo(currentBest) > 0) {
                                    Logger.log("\tCurrent best: " + candidate + " (p=" + p + ", n=" + n + ", new_p=" + (double) toCover_p + ", quality=" + quality + "\n", Level.FINEST);
                                    currentBest.quality = quality;
                                    currentBest.covered = p;
                                    currentBest.condition = candidateCondition;
                                    currentBest.coveredExamples = examplesCoveredByInterval.clone();
                                }
                            }
                        }
                        if (params.areNegatedConditionsEnabled()) {
                            induceNegatedCondition(trainSet, apriori_prec, (IntegerBitSet) uncoveredPositives, P, N, classId, attr, currentBest, examplesCoveredByInterval, rightValue, movedLeftValue);
                        }
                        examplesCoveredByInterval.addAll(precalculatedCoverings.get(attr).get(rightValue));
                    }
                }
                setBestCondition(currentBest);
            });
            futures.add(f);
        }
        return futures;
    }

    private void induceNegatedCondition(ExampleSet trainSet, double apriori_prec, IntegerBitSet uncoveredPositives, double P, double N, double classId, Attribute attr, ConditionEvaluation currentBest, IntegerBitSet examplesCoveredByInterval, double rightValue, double movedLeftValue) {
        IntegerBitSet negatedIntervalBitSet = examplesCoveredByInterval.clone();
        negatedIntervalBitSet.negate();
        /*
        Removing empty values
        Precalculated covering does not take into account empty values (they are simply not counted). That
        is why negating is not enough to calculate negated conditions coverage as it will potentially count
        empty values. Solution is to negate and later manually remove all empty values.
         */
        IntegerBitSet emptyValuesCovered = precalculatedCoverings.get(attr).get(Double.NaN);
        if (emptyValuesCovered != null && emptyValuesCovered.size() > 0) {
            negatedIntervalBitSet.removeAll(emptyValuesCovered);
        }

        double negatedToCover_p = negatedIntervalBitSet.calculateIntersectionSize(uncoveredPositives);
        double negated_p = negatedIntervalBitSet.calculateIntersectionSize(this.P.get(classId));
        double negated_n = negatedIntervalBitSet.calculateIntersectionSize(this.N.get(classId));
        double negatedPrecision = negated_p / (negated_p + negated_n);

        if (negatedPrecision > apriori_prec) {
            double quality = params.getInductionMeasure().calculate(negated_p, negated_n, P, N);

            Interval interval = new Interval(movedLeftValue, rightValue, true, false);
            interval.setIsRealInterval(true);
            interval.negate();
            ElementaryCondition candidateCondition = new ElementaryCondition(attr.getName(), interval, true);


            if (checkCandidate(candidateCondition, classId, P, negatedToCover_p)) {
                ConditionEvaluation candidate = new ConditionEvaluation();
                candidate.quality = quality;
                candidate.covered = negated_p;
                candidate.condition = candidateCondition;
                candidate.coveredExamples = negatedIntervalBitSet;

                addCondition(candidate);

                if (candidate.compareTo(currentBest) > 0) {
                    Logger.log("\tCurrent best: " + candidate + " (p=" + negated_p + ", n=" + negated_n + ", new_p=" + (double) negatedToCover_p + ", quality=" + quality + "\n", Level.FINEST);
                    currentBest.quality = quality;
                    currentBest.covered = negated_p;
                    currentBest.condition = candidateCondition;
                    currentBest.coveredExamples = negatedIntervalBitSet;
                }
            }
        }
    }

    @Override
    public void onBestConditionSelected(ConditionEvaluation conditionEvaluation) {
    }

    @Override
    public AbstractConditionsInducer clone() {
        return new IntervalConditionsInducer(this);
    }
}
