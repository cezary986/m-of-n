package adaa.analytics.rules.logic.induction.conditions.classification;

import adaa.analytics.rules.logic.induction.AttributeComparator;
import adaa.analytics.rules.logic.induction.ConditionEvaluation;
import adaa.analytics.rules.logic.induction.InductionParameters;
import adaa.analytics.rules.logic.induction.conditions.AbstractConditionsInducer;
import adaa.analytics.rules.logic.induction.conditions.helpers.TotalPosNeg;
import adaa.analytics.rules.logic.representation.*;
import com.rapidminer.example.Attribute;
import com.rapidminer.example.ExampleSet;
import com.rapidminer.example.table.DataRow;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.stream.Collectors;

/**
 * Class for inducing plain traditional conditions
 */
public class PlainConditionsInducer extends AbstractConditionsInducer {

    private Set<Attribute> allowedAttributes;
    private Map<Attribute, Map<Double, IntegerBitSet>> precalculatedNegatedCoverings;

    public PlainConditionsInducer(
            ExecutorService pool,
            InductionParameters params,
            Map<Attribute, Map<Double, IntegerBitSet>> precalculatedCoverings,
            Map<Double, IntegerBitSet> P,
            Map<Double, IntegerBitSet> N
    ) {
        super(pool, params, precalculatedCoverings, P, N);

        this.allowedAttributes = new TreeSet<>(new AttributeComparator());
        this.precalculatedNegatedCoverings = new HashMap<>();
    }

    public PlainConditionsInducer(PlainConditionsInducer reference) {
        super(reference.pool, reference.params, reference.precalculatedCoverings, reference.P, reference.N);

        this.allowedAttributes = new HashSet<>(reference.allowedAttributes);
        this.precalculatedNegatedCoverings = new HashMap<>(reference.precalculatedNegatedCoverings);
    }

    @Override
    public List<Future> preprocess(ExampleSet trainSet, List<Attribute> numericalAttributes, List<Attribute> nominalAttributes) {
        List<Future> futures = new ArrayList<>();

        // iterate over all allowed decision attributes
        for (Attribute attr : trainSet.getAttributes()) {
            allowedAttributes.add(attr);
            Future f = pool.submit(() -> {
                if (attr.isNominal()) {
                    Map<Double, IntegerBitSet> negatedAttributeCovering = new TreeMap<Double, IntegerBitSet>();
                    // prepare bit vectors
                    for (int val = 0; val != attr.getMapping().size(); ++val) {
                        negatedAttributeCovering.put((double) val, new IntegerBitSet(trainSet.size()));
                    }
                    Map<Double, IntegerBitSet> attributeCovering = precalculatedCoverings.get(attr);
                    for (Double val : attributeCovering.keySet()) {
                        for (Double other_val : attributeCovering.keySet()) {
                                /*
                                Covering of condition attr != <value> is same as covering of condition:
                                (attr = <other_value> OR attr = <other_value2> OR ...)
                                 */
                            if (!Double.isNaN(val) && !Double.isNaN(other_val) && val != other_val) {
                                negatedAttributeCovering.get(val).addAll(attributeCovering.get(other_val));
                            }
                        }
                    }
                    synchronized (this) {
                        precalculatedNegatedCoverings.put(attr, negatedAttributeCovering);
                    }

                }
            });
            futures.add(f);
        }
        return futures;
    }

    private void induceNumericalCondition(
            Attribute attr,
            Rule rule,
            ExampleSet trainSet,
            double apriori_prec,
            Map<String, Map<Double, TotalPosNeg>> totals,
            Set<Integer> uncoveredPositives,
            Set<Integer> coveredByRule,
            ConditionEvaluation currentBest
    ) {
        double P = rule.getWeighted_P();
        double N = rule.getWeighted_N();
        double classId = ((SingletonSet) rule.getConsequence().getValueSet()).getValue();

        // statistics from all points
        double left_p = 0;
        double left_n = 0;
        double right_p = 0;
        double right_n = 0;

        // statistics from points yet to cover
        int toCover_right_p = 0;
        int toCover_left_p = 0;
        Attribute weightAttr = trainSet.getAttributes().getWeight();
        Set<Integer> positives = rule.getCoveredPositives();

        // get all distinctive values of attribute
        for (int id : coveredByRule) {
            DataRow dr = trainSet.getExample(id).getDataRow();
            double val = dr.get(attr);

            // exclude missing values from keypoints
            if (Double.isNaN(val)) {
                continue;
            }
            double w = (weightAttr != null) ? dr.get(weightAttr) : 1.0;

            // put to proper bin depending of class label
            if (positives.contains(id)) {
                right_p += w;
                if (uncoveredPositives.contains(id)) {
                    ++toCover_right_p;
                }
            } else {
                right_n += w;
            }
        }

        Map<Double, TotalPosNeg> attrTotals = totals.get(attr.getName());
        Double[] keys = attrTotals.keySet().toArray(new Double[attrTotals.size()]);
        Arrays.sort(keys);


        // check all possible midpoints (number of distinctive attribute values - 1)
        // if only one attribute value - ignore it
        for (int keyId = 0; keyId < keys.length - 1; ++keyId) {
            double key = keys[keyId];

            double next = keys[keyId + 1];
            double midpoint = (key + next) / 2;

            TotalPosNeg tot = attrTotals.get(key);
            left_p += tot.p;
            right_p -= tot.p;
            left_n += tot.n;
            right_n -= tot.n;
            toCover_left_p += tot.toCover_p;
            toCover_right_p -= tot.toCover_p;

            TotalPosNeg totNext = attrTotals.get(next);
            if ((tot.n == 0 && totNext.n == 0) || (tot.p == 0 && totNext.p == 0)) {
                continue;
            }

            // calculate precisions
            double left_prec = left_p / (left_p + left_n);
            double right_prec = right_p / (right_p + right_n);

            // evaluate left-side condition: a in (-inf, v)
            if (left_prec > apriori_prec) {
                double quality = params.getInductionMeasure().calculate(left_p, left_n, P, N);

                ElementaryCondition candidate = new ElementaryCondition(attr.getName(), Interval.create_le(midpoint));
                if (checkCandidate(candidate, classId, P, toCover_left_p)) {
                    ConditionEvaluation evaluation = new ConditionEvaluation();
                    evaluation.quality = quality;
                    evaluation.covered = left_p;
                    evaluation.condition = candidate;
                    evaluation.coveredExamples = calculateCoveredExamplesForNumericalCondition(
                            trainSet, candidate, attr
                    );
                    addCondition(evaluation);

                    if (evaluation.compareTo(currentBest) > 0) {
                        Logger.log("\tCurrent best: " + candidate + " (p=" + left_p + ", n=" + left_n + ", new_p=" + (double) toCover_left_p + ", quality=" + quality + "\n", Level.FINEST);
                        currentBest.quality = quality;
                        currentBest.covered = left_p;
                        currentBest.condition = candidate;
                        currentBest.coveredExamples = evaluation.coveredExamples;
                    }
                }
            }

            // evaluate right-side condition: a in <v, inf)
            if (right_prec > apriori_prec) {
                double quality = params.getInductionMeasure().calculate(right_p, right_n, P, N);

                ElementaryCondition candidate = new ElementaryCondition(attr.getName(), Interval.create_geq(midpoint));
                if (checkCandidate(candidate, classId, P, toCover_right_p)) {
                    ConditionEvaluation evaluation = new ConditionEvaluation();
                    evaluation.quality = quality;
                    evaluation.covered = right_p;
                    evaluation.condition = candidate;
                    evaluation.coveredExamples = calculateCoveredExamplesForNumericalCondition(
                            trainSet, candidate, attr
                    );

                    addCondition(evaluation);

                    if (evaluation.compareTo(currentBest) > 0) {
                        Logger.log("\tCurrent best: " + candidate + " (p=" + right_p + ", n=" + right_n + ", new_p=" + (double) toCover_right_p + ", quality=" + quality + "\n", Level.FINEST);
                        currentBest.quality = quality;
                        currentBest.covered = right_p;
                        currentBest.condition = candidate;
                        currentBest.coveredExamples = evaluation.coveredExamples;
                    }
                }
            }
        }
    }

    protected IntegerBitSet calculateCoveredExamplesForNumericalCondition(
            ExampleSet trainSet,
            ElementaryCondition condition, Attribute attribute
    ) {
        IntegerBitSet result = new IntegerBitSet(trainSet.size());
        Interval interval = (Interval) condition.getValueSet();

        Map<Double, IntegerBitSet> attrCoverings = precalculatedCoverings.get(attribute);
        for (Double value : attrCoverings.keySet()) {
            if (interval.contains(value))
                result.addAll(attrCoverings.get(value));
        }
        return result;
    }

    private void induceNominalCondition(
            Attribute attr,
            Rule rule,
            ExampleSet trainSet,
            double apriori_prec,
            Set<Integer> uncoveredPositives,
            Set<Integer> coveredByRule,
            ConditionEvaluation currentBest
    ) {
        // weighted case - no precalculated converings
        if (precalculatedCoverings == null) {
            induceWeightedNominalCondition(
                    attr, rule, trainSet, uncoveredPositives, coveredByRule, currentBest
            );
        } else {
            induceUnweightedNominalCondition(
                    attr, rule, trainSet, apriori_prec, uncoveredPositives, coveredByRule, currentBest
            );
        }
    }

    private void induceWeightedNominalCondition(
            Attribute attr,
            Rule rule,
            ExampleSet trainSet,
            Set<Integer> uncoveredPositives,
            Set<Integer> coveredByRule,
            ConditionEvaluation currentBest
    ) {
        Attribute weightAttr = trainSet.getAttributes().getWeight();
        IntegerBitSet positives = rule.getCoveredPositives();
        double P = rule.getWeighted_P();
        double N = rule.getWeighted_N();
        double classId = ((SingletonSet) rule.getConsequence().getValueSet()).getValue();

        // sum of positive and negative weights for all values
        double[] p = new double[attr.getMapping().size()];
        double[] n = new double[attr.getMapping().size()];

        int[] toCover_p = new int[attr.getMapping().size()];

        // get all distinctive values of attribute
        for (int id : coveredByRule) {
            DataRow dr = trainSet.getExample(id).getDataRow();
            double value = dr.get(attr);

            // omit missing values
            if (Double.isNaN(value)) {
                continue;
            }

            int castedValue = (int) value;
            double w = (weightAttr != null) ? dr.get(weightAttr) : 1.0;

            if (positives.contains(id)) {
                p[castedValue] += w;
                if (uncoveredPositives.contains(id)) {
                    ++toCover_p[castedValue];
                }

            } else {
                n[castedValue] += w;
            }
        }

        // try all possible conditions
        for (int i = 0; i < attr.getMapping().size(); ++i) {
            // evaluate equality condition a = v
            double quality = params.getInductionMeasure().calculate(p[i], n[i], P, N);

            ElementaryCondition candidate =
                    new ElementaryCondition(attr.getName(), new SingletonSet(i, attr.getMapping().getValues()));
            if (checkCandidate(candidate, classId, P, toCover_p[i])) {
                ConditionEvaluation evaluation = new ConditionEvaluation();
                evaluation.quality = quality;
                evaluation.covered = p[i];
                evaluation.condition = candidate;
                IntegerBitSet conditionCovered = precalculatedCoverings.get(attr).get(attr.getMapping().mapIndex(i));
                evaluation.coveredExamples = conditionCovered;

                addCondition(evaluation);

                if (evaluation.compareTo(currentBest) > 0) {
                    Logger.log("\tCurrent best: " + candidate + " (p=" + p[i] + ", n=" + n[i] + ", new_p=" + (double) toCover_p[i] + ", quality=" + quality + "\n", Level.FINEST);
                    currentBest.quality = quality;
                    currentBest.covered = p[i];
                    currentBest.condition = candidate;
                    currentBest.coveredExamples = evaluation.coveredExamples;
                }
            }
        }
    }

    private void induceUnweightedNominalCondition(
            Attribute attr,
            Rule rule,
            ExampleSet trainSet,
            double apriori_prec,
            Set<Integer> uncoveredPositives,
            Set<Integer> coveredByRule,
            ConditionEvaluation currentBest
    ) {
        double P = rule.getWeighted_P();
        double N = rule.getWeighted_N();
        double classId = ((SingletonSet) rule.getConsequence().getValueSet()).getValue();

        for (int i = 0; i < attr.getMapping().size(); ++i) {

            IntegerBitSet conditionCovered = precalculatedCoverings.get(attr).get((double) i);
            double p = conditionCovered.calculateIntersectionSize(rule.getCoveredPositives());
            int toCover_p = conditionCovered.calculateIntersectionSize((IntegerBitSet) coveredByRule, (IntegerBitSet) uncoveredPositives);
            double n = conditionCovered.calculateIntersectionSize((IntegerBitSet) coveredByRule) - p;

            double prec = p / (p + n);

            if (prec > apriori_prec) {
                // evaluate equality condition a = v
                double quality = params.getInductionMeasure().calculate(p, n, P, N);
                ElementaryCondition candidate =
                        new ElementaryCondition(attr.getName(), new SingletonSet(i, attr.getMapping().getValues()));
                if (checkCandidate(candidate, classId, P, toCover_p)) {
                    ConditionEvaluation evaluation = new ConditionEvaluation();
                    evaluation.quality = quality;
                    evaluation.covered = p;
                    evaluation.condition = candidate;
                    evaluation.coveredExamples = conditionCovered;

                    addCondition(evaluation);

                    if (evaluation.compareTo(currentBest) > 0) {
                        Logger.log("\tCurrent best: " + candidate + " (p=" + p + ", n=" + n + ", new_p=" + (double) toCover_p + ", quality=" + quality + "\n", Level.FINEST);
                        currentBest.quality = quality;
                        currentBest.covered = p;
                        currentBest.condition = candidate;
                        currentBest.coveredExamples = conditionCovered;
                    }
                }
            }
            if (params.areNegatedConditionsEnabled()) {
                induceUnweightedNegatedNominalCondition(
                        attr, rule, i, classId, apriori_prec, coveredByRule, uncoveredPositives, currentBest);
            }
        }
    }

    private void induceUnweightedNegatedNominalCondition(
            Attribute attr,
            Rule rule,
            int i,
            double classId,
            double apriori_prec,
            Set<Integer> coveredByRule,
            Set<Integer> uncoveredPositives,
            ConditionEvaluation currentBest
    ) {
        double P = rule.getWeighted_P();
        double N = rule.getWeighted_N();
        IntegerBitSet negatedConditionCovered = precalculatedNegatedCoverings.get(attr).get((double) i);
        double p = negatedConditionCovered.calculateIntersectionSize(rule.getCoveredPositives());
        double toCover_p = negatedConditionCovered.calculateIntersectionSize((IntegerBitSet) coveredByRule, (IntegerBitSet) uncoveredPositives);
        double n = negatedConditionCovered.calculateIntersectionSize((IntegerBitSet) coveredByRule) - p;
        double prec = p / (p + n);

        if (prec > apriori_prec) {
            // evaluate equality condition a = v
            double quality = params.getInductionMeasure().calculate(p, n, P, N);

            ElementaryCondition candidate =
                    new ElementaryCondition(attr.getName(), new SingletonSet(i, attr.getMapping().getValues()), true);
            if (checkCandidate(candidate, classId, P, toCover_p)) {
                ConditionEvaluation evaluation = new ConditionEvaluation();
                evaluation.quality = quality;
                evaluation.covered = p;
                evaluation.condition = candidate;
                evaluation.coveredExamples = negatedConditionCovered;

                addCondition(evaluation);

                if (evaluation.compareTo(currentBest) > 0) {
                    Logger.log("\tCurrent best: " + candidate + " (p=" + p + ", n=" + n + ", new_p=" + (double) toCover_p + ", quality=" + quality + "\n", Level.FINEST);
                    currentBest.quality = quality;
                    currentBest.covered = p;
                    currentBest.condition = candidate;
                    currentBest.coveredExamples = evaluation.coveredExamples.clone();
                }
            }
        }
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

        // iterate over all allowed decision attributes
        for (Attribute attr : allowedAttributes) {
            ConditionEvaluation currentBest = new ConditionEvaluation();

            // consider attributes in parallel
            Future future = pool.submit(() -> {
                // check if attribute is numerical or nominal
                if (attr.isNumerical()) {
                    induceNumericalCondition(
                            attr,
                            rule,
                            trainSet,
                            apriori_prec,
                            totals,
                            uncoveredPositives,
                            coveredByRule,
                            currentBest
                    );
                } else {
                    induceNominalCondition(
                            attr,
                            rule,
                            trainSet,
                            apriori_prec,
                            uncoveredPositives,
                            coveredByRule,
                            currentBest
                    );
                }
                setBestCondition(currentBest);
            });

            futures.add(future);
        }
        return futures;
    }

    @Override
    public void onBestConditionSelected(ConditionEvaluation conditionEvaluation) {
        List<ConditionBase> selected = new ArrayList<>();
        if (conditionEvaluation.condition instanceof CompoundCondition)
            selected.addAll(((CompoundCondition) conditionEvaluation.condition).getSubconditions());
        else
            selected.add(conditionEvaluation.condition);

        List<ConditionBase> inducedCondition = getConditions().stream().map((e) -> e.condition).collect(Collectors.toList());
        for (ConditionBase condition : selected) {
            IValueSet valueSet;
            if (condition instanceof ElementaryCondition) {
                valueSet = ((ElementaryCondition) condition).getValueSet();
            } else {
                valueSet = new SingletonSet(0, new ArrayList<>());
            }
            if (inducedCondition.contains(condition) || (valueSet instanceof DiscreteSet)) {
                for (Attribute attr : allowedAttributes) {
                    if (attr.getName().equals(((ElementaryCondition) condition).getAttribute()) && attr.isNominal()) {
                        allowedAttributes.remove(attr);
                        return;
                    }
                }
            }
        }
    }

    @Override
    public AbstractConditionsInducer clone() {
        return new PlainConditionsInducer(this);
    }
}
