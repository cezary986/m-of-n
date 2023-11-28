package adaa.analytics.rules.logic.induction.conditions;

import adaa.analytics.rules.logic.induction.ConditionEvaluation;
import adaa.analytics.rules.logic.induction.Covering;
import adaa.analytics.rules.logic.induction.InductionParameters;
import adaa.analytics.rules.logic.induction.conditions.helpers.TotalPosNeg;
import adaa.analytics.rules.logic.representation.ConditionBase;
import adaa.analytics.rules.logic.representation.IntegerBitSet;
import adaa.analytics.rules.logic.representation.Rule;
import com.rapidminer.example.Attribute;
import com.rapidminer.example.ExampleSet;

import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;


public abstract class AbstractConditionsInducer implements Cloneable {

    protected ExecutorService pool;
    protected InductionParameters params;
    /**
     * Map of precalculated coverings (time optimization).
     * For each attribute there is a set of distinctive values. For each value there is a bit vector of examples covered.
     */
    protected Map<Attribute, Map<Double, IntegerBitSet>> precalculatedCoverings;
    protected Map<Double, IntegerBitSet> P;
    protected Map<Double, IntegerBitSet> N;

    private ConcurrentLinkedQueue<ConditionEvaluation> conditions;
    public ConcurrentLinkedQueue<ConditionEvaluation> bestConditions;

    public AbstractConditionsInducer(
            ExecutorService pool,
            InductionParameters params,
            Map<Attribute, Map<Double, IntegerBitSet>> precalculatedCoverings,
            Map<Double, IntegerBitSet> P,
            Map<Double, IntegerBitSet> N
    ) {
        this.pool = pool;
        this.params = params;
        this.precalculatedCoverings = precalculatedCoverings;
        this.P = P;
        this.N = N;

        this.conditions = new ConcurrentLinkedQueue<>();
        this.bestConditions = new ConcurrentLinkedQueue<>();
    }

    /**
     * Perform induction preprocessing,
     *
     * @param trainSet
     * @return
     */
    public abstract List<Future> preprocess(ExampleSet trainSet,
                                            List<Attribute> numericalAttributes,
                                            List<Attribute> nominalAttributes);

    protected abstract List<Future<Void>> induceConditions(
            Rule rule,
            ExampleSet trainSet,
            double apriori_prec,
            Set<Integer> uncoveredPositives,
            Set<Integer> coveredByRule,
            Map<String, Map<Double, TotalPosNeg>> totals
    );

    public List<Future<Void>> induce(
            Rule rule,
            ExampleSet trainSet,
            double apriori_prec,
            Set<Integer> uncoveredPositives,
            Set<Integer> coveredByRule,
            Map<String, Map<Double, TotalPosNeg>> totals
    ) {
        this.conditions.clear();
        this.bestConditions.clear();
        return induceConditions(rule, trainSet, apriori_prec, uncoveredPositives, coveredByRule, totals);
    }

    protected synchronized void addCondition(ConditionEvaluation condition) {
        ConditionEvaluation copy = new ConditionEvaluation();
        copy.condition = condition.condition;
        copy.covered = condition.covered;
        if (condition.covering != null)
            copy.covering = new Covering(condition.covering);
        copy.quality = condition.quality;
        if (condition.coveredExamples != null) {
            copy.coveredExamples = condition.coveredExamples.clone();
        }
        conditions.add(copy);
    }

    protected synchronized void setBestCondition(ConditionEvaluation condition) {
        if (condition == null || condition.condition == null)
            return;
        ConditionEvaluation copy = new ConditionEvaluation();
        copy.condition = condition.condition;
        copy.covered = condition.covered;
        if (condition.covering != null)
            copy.covering = new Covering(condition.covering);
        copy.quality = condition.quality;
        if (condition.coveredExamples != null) {
            copy.coveredExamples = condition.coveredExamples.clone();
        }
        bestConditions.add(copy);
    }

    /***
     * Checks if candidate condition fulfills coverage requirement.
     *
     * @param cnd Candidate condition.
     * @param classId Class identifier.
     * @param newlyCoveredPositives Number of newly covered positive examples after addition of the condition.
     * @return
     */
    protected boolean checkCandidate(ConditionBase cnd, double classId, double totalPositives, double newlyCoveredPositives) {
        double adjustedMinCov = Math.min(
                params.getMinimumCovered(),
                Math.max(1.0, 0.2 * totalPositives));

        if (newlyCoveredPositives >= adjustedMinCov) {
            return true;
        } else {
            return false;
        }
    }

    /**
     * Method called when globally best new condition was selected. It could be used to eliminate
     * certain possibilities for further induction to eliminate risk of duplicate conditions in rule.
     *
     * @param conditionEvaluation
     */
    public abstract void onBestConditionSelected(ConditionEvaluation conditionEvaluation);

    public ConcurrentLinkedQueue<ConditionEvaluation> getConditions() {
        return this.conditions;
    }

    public ConditionEvaluation getBestCondition() {
        return this.bestConditions.stream()
                .max(ConditionEvaluation.getComparator())
                .orElse(this.bestConditions.peek());
    }

    /**
     * Every condition inducer class should implement method clone. Default java Clonable.clone method won't do here.
     * This method should clone object in such a way that ensures that this new copy could be safely used for induction
     * in another thread for another rule without conflicts.
     *
     * @return object clone
     */
    public abstract AbstractConditionsInducer clone();
}
