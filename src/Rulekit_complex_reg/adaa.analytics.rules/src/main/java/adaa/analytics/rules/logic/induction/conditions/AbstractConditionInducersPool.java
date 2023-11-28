package adaa.analytics.rules.logic.induction.conditions;

import adaa.analytics.rules.logic.induction.ConditionEvaluation;
import adaa.analytics.rules.logic.induction.conditions.helpers.TotalPosNeg;
import adaa.analytics.rules.logic.induction.conditions.regression.IntervalConditionsInducer;
import adaa.analytics.rules.logic.induction.conditions.regression.NumericalAttributesConditionsInducer;
import adaa.analytics.rules.logic.induction.conditions.regression.PlainConditionsInducer;
import adaa.analytics.rules.logic.representation.Rule;
import com.rapidminer.example.Attribute;
import com.rapidminer.example.ExampleSet;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Future;

public abstract class AbstractConditionInducersPool {

    protected List<AbstractConditionsInducer> inducers;

    public AbstractConditionInducersPool() {
        this.inducers = new ArrayList<>();
    }

    public AbstractConditionInducersPool(
            AbstractConditionInducersPool objectToClone
    ) {
        this.inducers = new ArrayList<>();
        for (AbstractConditionsInducer inducer : objectToClone.inducers) {
            this.inducers.add(inducer.clone());
        }
    }

    abstract public List<Future> preprocess(
            ExampleSet trainSet,
            List<Attribute> numericalAttributes,
            List<Attribute> nominalAttributes
    );

    abstract public List<Future<Void>> induceConditions(
            Rule rule,
            ExampleSet trainSet,
            double apriori_prec,
            Set<Integer> uncoveredPositives,
            Set<Integer> coveredByRule,
            Map<String, Map<Double, TotalPosNeg>> totals
    );

    public void onBestConditionSelected(ConditionEvaluation selectedCondition) {
        for (AbstractConditionsInducer inducer : this.inducers) {
            inducer.onBestConditionSelected(selectedCondition);
        }
    }

    /**
     * Returns list of all induced conditions - all induced conditions from all inducers
     *
     * @return list of all induced conditions
     */

    public List<ConditionEvaluation> getInducedConditions() {
        List<ConditionEvaluation> conditions = new ArrayList<>();
        for (AbstractConditionsInducer inducer : inducers)
            conditions.addAll(inducer.getConditions());
        return conditions;
    }


    public List<ConditionEvaluation> getNumericalInducedConditions() {

        List<ConditionEvaluation> conditions = new ArrayList<>();
        for (AbstractConditionsInducer inducer : inducers) {
            if (inducer instanceof PlainConditionsInducer || inducer instanceof NumericalAttributesConditionsInducer || inducer instanceof IntervalConditionsInducer) {
                conditions.addAll(inducer.getConditions());
            }
        }
        return conditions;
    }

    /**
     * Returns list of best induced conditions - single best condition per inducer.
     *
     * @return list of best induced conditions
     */
    public List<ConditionEvaluation> getBestInducedConditions() {
        List<ConditionEvaluation> conditions = new ArrayList<>();
        for (AbstractConditionsInducer inducer : inducers) {
            ConditionEvaluation bestCondition = inducer.getBestCondition();
            if (bestCondition != null)
                conditions.add(bestCondition);
        }
        return conditions;
    }

    public List<ConditionEvaluation> getAllConditions() {
        List<ConditionEvaluation> conditions = new ArrayList<>();
        for (AbstractConditionsInducer inducer : inducers) {
            conditions.addAll(inducer.bestConditions);
        }
        return conditions;
    }

    abstract public AbstractConditionInducersPool clone();
}
