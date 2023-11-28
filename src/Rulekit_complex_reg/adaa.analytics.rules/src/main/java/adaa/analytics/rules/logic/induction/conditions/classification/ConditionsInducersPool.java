package adaa.analytics.rules.logic.induction.conditions.classification;

import adaa.analytics.rules.logic.induction.ConditionEvaluation;
import adaa.analytics.rules.logic.induction.InductionParameters;
import adaa.analytics.rules.logic.induction.conditions.AbstractConditionInducersPool;
import adaa.analytics.rules.logic.induction.conditions.AbstractConditionsInducer;
import adaa.analytics.rules.logic.induction.conditions.helpers.TotalPosNeg;
import adaa.analytics.rules.logic.representation.IntegerBitSet;
import adaa.analytics.rules.logic.representation.Rule;
import com.rapidminer.example.Attribute;
import com.rapidminer.example.ExampleSet;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;


public class ConditionsInducersPool extends AbstractConditionInducersPool {

    public ConditionsInducersPool(
            ExecutorService pool,
            InductionParameters params,
            Map<Attribute, Map<Double, IntegerBitSet>> precalculatedCoverings,
            Map<Double, IntegerBitSet> P,
            Map<Double, IntegerBitSet> N
    ) {
        super();

        this.inducers.add(new PlainConditionsInducer(pool, params, precalculatedCoverings, P, N));
        if (params.areAttributesConditionsEnabled())
            this.inducers.add(new NumericalAttributesConditionsInducer(pool, params, precalculatedCoverings, P, N));
        if (params.areNominalAttributesConditionsEnabled())
            this.inducers.add(new NominalAttributesConditionsInducer(pool, params, precalculatedCoverings, P, N));
        if (params.areDiscreteSetConditionsEnabled())
            this.inducers.add(new DiscreteSetConditionsInducer(pool, params, precalculatedCoverings, P, N));
        if (params.areAttributesIntervalsEnabled())
            this.inducers.add(new IntervalConditionsInducer(pool, params, precalculatedCoverings, P, N));
    }

    public ConditionsInducersPool(
            AbstractConditionInducersPool ref
    ) {
        super(ref);
    }

    @Override
    public List<Future> preprocess(
            ExampleSet trainSet,
            List<Attribute> numericalAttributes,
            List<Attribute> nominalAttributes
    ) {
        List<Future> futures = new ArrayList<>();
        for (AbstractConditionsInducer inducer : inducers) {
            futures.addAll(inducer.preprocess(trainSet, numericalAttributes, nominalAttributes));
        }
        return futures;
    }

    @Override
    public List<Future<Void>> induceConditions(
            Rule rule,
            ExampleSet trainSet,
            double apriori_prec,
            Set<Integer> uncoveredPositives,
            Set<Integer> coveredByRule,
            Map<String, Map<Double, TotalPosNeg>> totals
    ) {
        List<Future<Void>> futures = new ArrayList<>();
        for (AbstractConditionsInducer inducer : this.inducers) {
            futures.addAll(inducer.induce(rule, trainSet, apriori_prec, uncoveredPositives, coveredByRule, totals));
        }
        return futures;
    }

    @Override
    public AbstractConditionInducersPool clone() {
        return new ConditionsInducersPool(this);
    }
}
