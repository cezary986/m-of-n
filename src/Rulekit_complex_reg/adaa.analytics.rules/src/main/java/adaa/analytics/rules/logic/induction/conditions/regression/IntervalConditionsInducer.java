package adaa.analytics.rules.logic.induction.conditions.regression;

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

        for (Attribute attr : this.numericalAttributes) {
            Future f = pool.submit(() -> {
                Map<Double, TotalPosNeg> attrTotals = totals.get(attr.getName());
                List<Double> keysList = attrTotals.keySet().stream()
                        .filter(e -> !e.isNaN())
                        .filter((e) -> e != null)
                        .collect(Collectors.toList());
                Double[] keys = keysList.toArray(new Double[keysList.size()]);
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

                        Interval interval = new Interval(movedLeftValue, rightValue, true, false);
                        interval.setIsRealInterval(true);
                        ElementaryCondition candidateCondition = new ElementaryCondition(attr.getName(), interval);

                        ConditionEvaluation newBest = RegressionConditionEvaluator.checkCandidate(
                                trainSet, rule, candidateCondition, uncoveredPositives, currentBest, params
                        );
                        if (newBest != null) {
                            Logger.log("\tCurrent best: " + candidateCondition + " (quality=" + newBest.quality + "\n", Level.FINEST);
                            currentBest.quality = newBest.quality;
                            currentBest.covered = newBest.covered;
                            currentBest.condition = candidateCondition;
                            currentBest.coveredExamples = examplesCoveredByInterval.clone();
                            addCondition(newBest);
                            setBestCondition(currentBest);
                        }
                        if (params.areNegatedConditionsEnabled()) {
                            induceNegatedCondition(trainSet, rule, (IntegerBitSet) uncoveredPositives, attr, currentBest, examplesCoveredByInterval, rightValue, movedLeftValue);
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

    private void induceNegatedCondition(ExampleSet trainSet, Rule rule, IntegerBitSet uncoveredPositives, Attribute attr, ConditionEvaluation currentBest, IntegerBitSet examplesCoveredByInterval, double rightValue, double movedLeftValue) {
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

        Interval interval = new Interval(movedLeftValue, rightValue, true, false);
        interval.setIsRealInterval(true);
        interval.negate();
        ElementaryCondition candidateCondition = new ElementaryCondition(attr.getName(), interval, true);

        ConditionEvaluation newBest = RegressionConditionEvaluator.checkCandidate(
                trainSet, rule, candidateCondition, uncoveredPositives, currentBest, params
        );
        if (newBest != null) {
            Logger.log("\tCurrent best: " + candidateCondition + " (quality=" + newBest.quality + "\n", Level.FINEST);
            currentBest.quality = newBest.quality;
            currentBest.covered = newBest.covered;
            currentBest.condition = candidateCondition;
            currentBest.coveredExamples = negatedIntervalBitSet;
            addCondition(newBest);
            setBestCondition(currentBest);
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
