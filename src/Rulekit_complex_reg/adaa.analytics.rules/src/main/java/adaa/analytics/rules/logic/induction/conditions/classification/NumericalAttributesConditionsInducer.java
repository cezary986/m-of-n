package adaa.analytics.rules.logic.induction.conditions.classification;

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

        double P = rule.getWeighted_P();
        double N = rule.getWeighted_N();
        double classId = ((SingletonSet) rule.getConsequence().getValueSet()).getValue();

        List<Future<Void>> futures = new ArrayList<>();

        ConditionEvaluation currentBest = new ConditionEvaluation();

        Future future = pool.submit(() -> {
            for (AttributesCondition candidateCondition : allowedAttributesConditions) {
                IntegerBitSet globalCoverage = attributeConditionsCoverageGlobal.get(candidateCondition);

                double p = globalCoverage.calculateIntersectionSize(rule.getCoveredPositives());
                int toCover_p = globalCoverage.calculateIntersectionSize((IntegerBitSet) coveredByRule, (IntegerBitSet) uncoveredPositives);
                double n = globalCoverage.calculateIntersectionSize((IntegerBitSet) coveredByRule) - p;

                double prec = p / (p + n);
                if (prec > apriori_prec) {
                    // evaluate equality condition a = v
                    double quality = params.getInductionMeasure().calculate(p, n, P, N);

                    if (checkCandidate(candidateCondition, classId, P, toCover_p)) {
                        ConditionEvaluation candidate = new ConditionEvaluation();
                        candidate.quality = quality;
                        candidate.covered = p;
                        candidate.condition = candidateCondition;
                        candidate.coveredExamples = globalCoverage;
                        addCondition(candidate);

                        if (candidate.compareTo(currentBest) > 0) {
                            Logger.log("\tCurrent best: " + candidateCondition + " (p=" + p + ", n=" + n + ", new_p=" + (double) toCover_p + ", quality=" + quality + "\n", Level.FINEST);
                            currentBest.quality = quality;
                            currentBest.covered = p;
                            currentBest.condition = candidateCondition;
                            currentBest.coveredExamples = globalCoverage;
                        }
                    }
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
