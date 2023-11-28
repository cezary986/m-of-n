package adaa.analytics.rules.logic.induction.conditions;

import adaa.analytics.rules.logic.induction.ConditionEvaluation;
import adaa.analytics.rules.logic.induction.InductionParameters;
import adaa.analytics.rules.logic.representation.IntegerBitSet;
import com.rapidminer.example.Attribute;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;

public abstract class AbstractConditionsPostInducer extends AbstractConditionsInducer {

    protected List<ConditionEvaluation> candidates;

    public AbstractConditionsPostInducer(ExecutorService pool, InductionParameters params, Map<Attribute, Map<Double, IntegerBitSet>> precalculatedCoverings, Map<Double, IntegerBitSet> P, Map<Double, IntegerBitSet> N) {
        super(pool, params, precalculatedCoverings, P, N);
    }

    public void setCandidates(List<ConditionEvaluation> candidates) {
        this.candidates = candidates;
    }
}
