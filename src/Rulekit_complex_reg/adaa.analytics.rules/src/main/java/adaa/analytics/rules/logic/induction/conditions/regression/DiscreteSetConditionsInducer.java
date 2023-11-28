package adaa.analytics.rules.logic.induction.conditions.regression;

import adaa.analytics.rules.logic.induction.ConditionEvaluation;
import adaa.analytics.rules.logic.induction.Covering;
import adaa.analytics.rules.logic.induction.InductionParameters;
import adaa.analytics.rules.logic.induction.conditions.AbstractConditionsInducer;
import adaa.analytics.rules.logic.induction.conditions.helpers.TotalPosNeg;
import adaa.analytics.rules.logic.representation.*;
import com.rapidminer.example.Attribute;
import com.rapidminer.example.Example;
import com.rapidminer.example.ExampleSet;
import com.rapidminer.example.table.DataRow;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.logging.Level;

public class DiscreteSetConditionsInducer extends adaa.analytics.rules.logic.induction.conditions.classification.DiscreteSetConditionsInducer {

    public DiscreteSetConditionsInducer(ExecutorService pool, InductionParameters params, Map<Attribute, Map<Double, IntegerBitSet>> precalculatedCoverings, Map<Double, IntegerBitSet> P, Map<Double, IntegerBitSet> N) {
        super(pool, params, precalculatedCoverings, P, N);

        this.precalculatedSubsetsCoverings = new HashMap<>();
    }

    public DiscreteSetConditionsInducer(DiscreteSetConditionsInducer reference) {
        super(reference.pool, reference.params, reference.precalculatedCoverings, reference.P, reference.N);

        this.precalculatedCoverings = reference.precalculatedCoverings;
        this.precalculatedSubsetsCoverings = reference.precalculatedSubsetsCoverings;
        this.allowedAttributes = new HashSet<>(reference.allowedAttributes);
    }

    @Override
    public List<Future<Void>> induceConditions(Rule rule, ExampleSet trainSet, double apriori_prec, Set<Integer> uncoveredPositives, Set<Integer> coveredByRule, Map<String, Map<Double, TotalPosNeg>> totals) {
        if (trainSet.getAttributes().getWeight() != null)
            throw new IllegalArgumentException("Discrete Set conditions are not supported for weighted datasets");

        List<Future<Void>> futures = new ArrayList<>();
        for (Attribute attr : allowedAttributes) {
            Future f = pool.submit(() -> {
                ConditionEvaluation currentBest = new ConditionEvaluation(trainSet);
                for (int i = 0; i < precalculatedSubsetsCoverings.get(attr).size(); ++i) {
                    NominalValuesSubset nominalValuesSubset = precalculatedSubsetsCoverings.get(attr).get(i);
                    IntegerBitSet conditionCovered = nominalValuesSubset.getCoveredExamples();
                    ElementaryCondition candidate = new ElementaryCondition(
                            attr.getName(),
                            new DiscreteSet(nominalValuesSubset.getSubset(), attr.getMapping().getValues())
                    );
                    if (candidate.toString().equals("disease_type = {0, 1, 2}")) {
                        int a = 0;

                    }

                    ConditionEvaluation newBest = RegressionConditionEvaluator.checkCandidate(
                            trainSet, rule, candidate, uncoveredPositives, currentBest, params
                    );
                    if (newBest != null) {
                        Logger.log("\tCurrent best: " + candidate + " (quality=" + newBest.quality + "\n", Level.FINEST);
                        currentBest.quality = newBest.quality;
                        currentBest.covered = newBest.covered;
                        currentBest.condition = candidate;
                        currentBest.coveredExamples = conditionCovered;
                        addCondition(newBest);
                        setBestCondition(currentBest);
                    }
                }
            });
            futures.add(f);
        }
        return futures;
    }

    @Override
    public AbstractConditionsInducer clone() {
        return new DiscreteSetConditionsInducer(this);
    }
}
