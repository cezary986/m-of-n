package adaa.analytics.rules.logic.induction.conditions.regression;

import adaa.analytics.rules.logic.induction.ConditionEvaluation;
import adaa.analytics.rules.logic.induction.InductionParameters;
import adaa.analytics.rules.logic.induction.conditions.AbstractConditionsInducer;
import adaa.analytics.rules.logic.induction.conditions.helpers.TotalPosNeg;
import adaa.analytics.rules.logic.representation.*;
import com.rapidminer.example.Attribute;
import com.rapidminer.example.Example;
import com.rapidminer.example.ExampleSet;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.stream.Collectors;

public class NumericalAttributesConditionsInducer extends AbstractConditionsInducer {

    Set<AttributesCondition> allowedAttributesConditions;
    Map<AttributesCondition, IntegerBitSet> attributeConditionsCoverageGlobal;

    public NumericalAttributesConditionsInducer(ExecutorService pool, InductionParameters params, Map<Attribute, Map<Double, IntegerBitSet>> precalculatedCoverings, Map<Double, IntegerBitSet> P, Map<Double, IntegerBitSet> N) {
        super(pool, params, precalculatedCoverings, P, N);

        allowedAttributesConditions = new HashSet<>();
        attributeConditionsCoverageGlobal = new HashMap<>();
    }

    public NumericalAttributesConditionsInducer(NumericalAttributesConditionsInducer reference) {
        super(reference.pool, reference.params, reference.precalculatedCoverings, reference.P, reference.N);

        allowedAttributesConditions = new HashSet<>();
        allowedAttributesConditions.addAll(reference.allowedAttributesConditions);
        attributeConditionsCoverageGlobal = new HashMap<>();
        attributeConditionsCoverageGlobal.putAll(reference.attributeConditionsCoverageGlobal);
    }

    @Override
    public List<Future> preprocess(ExampleSet trainSet, List<Attribute> numericalAttributes, List<Attribute> nominalAttributes) {
        List<Future> futures = new ArrayList<>();
        Future f = pool.submit(() -> {
            for (int i = 0; i < numericalAttributes.size() - 1; i++) {
                for (int j = i + 1; j < numericalAttributes.size(); j++) {
                    for (Example example : trainSet) {
                        Attribute attr1 = numericalAttributes.get(i);
                        Attribute attr2 = numericalAttributes.get(j);
                        double attr1Value = example.getValue(attr1);
                        double attr2Value = example.getValue(attr2);

                        if (!(Double.isNaN(attr1Value) || Double.isNaN(attr2Value))) {
                            AttributesCondition key;
                            if (attr1Value > attr2Value) {
                                key = new AttributesCondition(attr1, attr2, AttributesCondition.Operators.GREATER);
                            } else {
                                if (attr1Value == attr2Value)
                                    key = new AttributesCondition(attr1, attr2, AttributesCondition.Operators.EQUAL);
                                else
                                    key = new AttributesCondition(attr1, attr2, AttributesCondition.Operators.LOWER);
                            }
                            if (!attributeConditionsCoverageGlobal.containsKey(key)) {
                                IntegerBitSet covered = new IntegerBitSet(trainSet.size());
                                key.evaluate(trainSet, covered);
                                attributeConditionsCoverageGlobal.put(key, covered);
                            }
                        }
                    }
                }
            }
            allowedAttributesConditions.addAll(attributeConditionsCoverageGlobal.keySet());
        });
        futures.add(f);
        return futures;
    }

    @Override
    public List<Future<Void>> induceConditions(Rule rule, ExampleSet trainSet, double apriori_prec, Set<Integer> uncoveredPositives, Set<Integer> coveredByRule, Map<String, Map<Double, TotalPosNeg>> totals) {
        if (trainSet.getAttributes().getWeight() != null)
            throw new IllegalArgumentException("Numerical attributes conditions are not supported for weighted datasets");

        List<Future<Void>> futures = new ArrayList<>();

        ConditionEvaluation currentBest = new ConditionEvaluation();

        Future future = pool.submit(() -> {
            for (AttributesCondition candidateCondition : allowedAttributesConditions) {
                IntegerBitSet globalCoverage = attributeConditionsCoverageGlobal.get(candidateCondition);

                ConditionEvaluation newBest = RegressionConditionEvaluator.checkCandidate(
                        trainSet, rule, candidateCondition, uncoveredPositives, currentBest, params
                );
                if (newBest != null) {
                    Logger.log("\tCurrent best: " + candidateCondition + " (quality=" + newBest.quality + "\n", Level.FINEST);
                    currentBest.quality = newBest.quality;
                    currentBest.covered = newBest.covered;
                    currentBest.condition = candidateCondition;
                    currentBest.coveredExamples = globalCoverage;
                    addCondition(newBest);
                    setBestCondition(currentBest);
                }
            }
            setBestCondition(currentBest);
        });
        futures.add(future);
        return futures;
    }

    @Override
    public synchronized void onBestConditionSelected(ConditionEvaluation conditionEvaluation) {
        List<ConditionBase> selected = new ArrayList<>();
        if (conditionEvaluation.condition instanceof CompoundCondition)
            selected.addAll(((CompoundCondition) conditionEvaluation.condition).getSubconditions());
        else
            selected.add(conditionEvaluation.condition);
        List<ConditionBase> inducedCondition = getConditions().stream().map((e) -> e.condition).collect(Collectors.toList());
        for (ConditionBase condition : selected) {
            if (inducedCondition.contains(condition)) {
                AttributesCondition bestCondition = (AttributesCondition) condition;
                allowedAttributesConditions.remove(bestCondition);
            }
        }

    }

    @Override
    public AbstractConditionsInducer clone() {
        return new NumericalAttributesConditionsInducer(this);
    }
}
