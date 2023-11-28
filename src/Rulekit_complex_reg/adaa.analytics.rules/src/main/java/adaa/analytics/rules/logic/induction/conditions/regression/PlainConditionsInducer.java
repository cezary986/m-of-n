package adaa.analytics.rules.logic.induction.conditions.regression;

import adaa.analytics.rules.logic.induction.AttributeComparator;
import adaa.analytics.rules.logic.induction.ConditionEvaluation;
import adaa.analytics.rules.logic.induction.InductionParameters;
import adaa.analytics.rules.logic.induction.conditions.AbstractConditionsInducer;
import adaa.analytics.rules.logic.induction.conditions.helpers.TotalPosNeg;
import adaa.analytics.rules.logic.representation.*;
import com.rapidminer.example.Attribute;
import com.rapidminer.example.ExampleSet;
import com.rapidminer.example.table.DataRow;
import jdk.nashorn.internal.runtime.regexp.joni.exception.ValueException;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.stream.Collectors;

/**
 * Class for inducing plain traditional conditions
 */
public class PlainConditionsInducer extends adaa.analytics.rules.logic.induction.conditions.classification.PlainConditionsInducer {

    private Set<Attribute> allowedAttributes;
    private Map<Attribute, Map<Double, IntegerBitSet>> precalculatedNegatedCoverings;

    public PlainConditionsInducer(
            ExecutorService pool,
            InductionParameters params,
            Map<Attribute, Map<Double, IntegerBitSet>> precalculatedCoverings
    ) {
        super(pool, params, precalculatedCoverings, null, null);

        this.allowedAttributes = new TreeSet<>(new AttributeComparator());
        this.precalculatedNegatedCoverings = new HashMap<>();
    }

    public PlainConditionsInducer(PlainConditionsInducer reference) {
        super(reference.pool, reference.params, reference.precalculatedCoverings, null, null);

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

    private void induceNumericalConditionUsingMeanBasedRegression(
            Attribute attr,
            Rule rule,
            ExampleSet trainSet,
            Set<Integer> uncoveredPositives,
            Set<Integer> covered,
            ConditionEvaluation currentBest
    ) {
        SortedExampleSetEx set = (SortedExampleSetEx) trainSet;
        ConditionEvaluation best = new ConditionEvaluation();

        IntegerBitSet mask = set.nonMissingVals.get(attr);
        IntegerBitSet localCov = new IntegerBitSet(set.size());
        IntegerBitSet localCovNew = new IntegerBitSet(set.size());

        localCov.setAll(covered);
        localCov.retainAll(mask);
        localCovNew.setAll(localCov);
        localCovNew.retainAll(uncoveredPositives);

        IntegerBitSet [] covs = new IntegerBitSet[2];
        // zero na warunek < od wartośc
        covs[0] = new IntegerBitSet(trainSet.size());
        // zero na warunek > od wartośc
        covs[1] = localCov;

        // initial values for left hand side and right hand side conditions
        class Stats{
            double sum_w = 0;
            double sum_new_w = 0;
            double sum_y = 0;
            double sum_y2 = 0;
            double mean_y = 0 ;
            double stddev_y = 0;
        }

        Stats[] stats = new Stats[2];
        stats[0] = new Stats();
        stats[1] = new Stats();

        // get indices array
        Integer [] ids = new Integer[localCov.size()];
        int i = 0;
        for (int id : localCov) {
            ids[i++] = id;
            double y = set.labelsWeighted[id];
            stats[1].sum_y += y;
            stats[1].sum_y2 += y * y;
            stats[1].sum_w += set.weights[id];
        }

        stats[1].sum_new_w = localCovNew.size();


        // sort ids array according to the attribute value
        Arrays.sort(ids, Comparator.comparingDouble(a -> trainSet.getExample(a).getValue(attr)));

        // iterate over examples in increasing attribute value order
        double prev_val = Double.MAX_VALUE;
        for (i = 0; i < ids.length; ++i) {
            int id = ids[i];
            double val = trainSet.getExample(id).getValue(attr);

            if (Double.isNaN(val)) {
                continue;
            }

            // we moved to another value - verify midpoint
            if (val > prev_val) {
                double midpoint = (val + prev_val) / 2;

                // evaluate both conditions
                for (int c = 0; c < 2; ++c) {
                    stats[c].mean_y = stats[c].sum_y / stats[c].sum_w;
                    double mean_y2 = stats[c].sum_y2 / stats[c].sum_w;
                    stats[c].stddev_y = Math.sqrt(mean_y2 - stats[c].mean_y * stats[c].mean_y); // VX = E(X^2) - (EX)^2

                    // binary search to get elements inside epsilon
                    // assumption: double value preceeding/following one being search appears at most once
                    int lo = Arrays.binarySearch(set.labels, Math.nextDown(stats[c].mean_y - stats[c].stddev_y));
                    if (lo < 0) {
                        lo = -(lo + 1); // if element not found, extract id of the next larger: ret = (-(insertion point) - 1)
                    } else { lo += 1;} // if element found move to next one (first inside a range)

                    int hi = Arrays.binarySearch(set.labels, Math.nextUp(stats[c].mean_y + stats[c].stddev_y));
                    if (hi < 0) { hi = -(hi + 1); // if element not found, extract id of the next larger: ret = (-(insertion point) - 1)
                    } // if element found - do nothing (first after the range)

                    double P = set.totalWeightsBefore[hi] - set.totalWeightsBefore[lo];
                    double N = set.totalWeightsBefore[set.size()] - P;
                    double n = stats[c].sum_w;
                    double p = 0;
                    double new_n = stats[c].sum_new_w;
                    double new_p = 0;

                    // iterate over elements from the entire set
                    for (int j = lo; j < hi; ++j) {
                        if (covs[c].contains(j)) {
                            double wj = set.weights[j];
                            n -= wj;
                            p += wj;
                            if (uncoveredPositives.contains(j)) {
                                new_n -= wj;
                                new_p += wj;
                            }
                        }
                    }

                    double quality = params.getInductionMeasure().calculate(p, n, P, N);

                    if (quality > best.quality || (quality == best.quality && (new_p + new_n) > best.covered)) {

                        ElementaryCondition candidateLeft = new ElementaryCondition(attr.getName(),
                                (c == 0) ? Interval.create_le(midpoint) : Interval.create_geq(midpoint));

                        ConditionEvaluation newBest = RegressionConditionEvaluator.checkCandidate(best, candidateLeft, params, p, n, new_p, new_n, P, N);
                        if (newBest != null) {
                            Logger.log("\t\tCurrent best: " + candidateLeft + " (p=" + p + ", n=" + n +
                                    ", new_p=" + (double) new_p + ", new_n="+  new_n +
                                    ", P=" + P + ", N=" + N +
                                    ", mean_y=" + stats[c].mean_y + ", mean_y2=" + mean_y2 + ", stddev_y=" + stats[c].stddev_y +
                                    ", quality=" + quality + "\n", Level.FINEST);
                            currentBest.quality = newBest.quality;
                            currentBest.covered = newBest.covered;
                            currentBest.condition = candidateLeft;
                            currentBest.coveredExamples = calculateCoveredExamplesForNumericalCondition(
                                    trainSet, candidateLeft, attr
                            );
                            addCondition(currentBest);
                        }
                    }
                }
            }

            // update stats
            double y = set.labelsWeighted[id];
            double w = set.weights[id];
            stats[0].sum_y += y;
            stats[0].sum_y2 += y * y;
            stats[0].sum_w += w;
            covs[0].add(id);

            stats[1].sum_y -= y;
            stats[1].sum_y2 -= y*y;
            stats[1].sum_w -= w;
            covs[1].remove(id);

            if (uncoveredPositives.contains(id)) {
                stats[0].sum_new_w += w;
                stats[1].sum_new_w -= w;
            }

            prev_val = val;
        }
    }

    private void induceNumericalCondition(
            Attribute attr,
            Rule rule,
            ExampleSet trainSet,
            Map<String, Map<Double, TotalPosNeg>> totals,
            Set<Integer> uncoveredPositives,
            ConditionEvaluation currentBest
    ) {
        Map<Double, TotalPosNeg> attrTotals = totals.get(attr.getName());
        Double[] keys = attrTotals.keySet().toArray(new Double[attrTotals.size()]);
        Arrays.sort(keys);


        // check all possible midpoints (number of distinctive attribute values - 1)
        // if only one attribute value - ignore it
        for (int keyId = 0; keyId < keys.length - 1; ++keyId) {
            double key = keys[keyId];

            double next = keys[keyId + 1];
            double midpoint = (key + next) / 2;

            ElementaryCondition candidateLeft = new ElementaryCondition(attr.getName(), Interval.create_le(midpoint));

            ConditionEvaluation newBest = RegressionConditionEvaluator.checkCandidate(
                    trainSet, rule, candidateLeft, uncoveredPositives, currentBest, params
            );

            // evaluate left-side condition: a in <v, inf)
            if (newBest != null) {
                Logger.log("\tCurrent best: " + candidateLeft + " (quality=" + newBest.quality + "\n", Level.FINEST);
                currentBest.quality = newBest.quality;
                currentBest.covered = newBest.covered;
                currentBest.condition = candidateLeft;
                currentBest.coveredExamples = calculateCoveredExamplesForNumericalCondition(
                        trainSet, candidateLeft, attr
                );
                addCondition(currentBest);
            }

            // evaluate right-side condition: a in <v, inf)
            ElementaryCondition candidateRight = new ElementaryCondition(attr.getName(), Interval.create_geq(midpoint));
            newBest = RegressionConditionEvaluator.checkCandidate(
                    trainSet, rule, candidateRight, uncoveredPositives, currentBest, params
            );
            if (newBest != null) {
                Logger.log("\tCurrent best: " + candidateRight + " (quality=" + newBest.quality + "\n", Level.FINEST);
                currentBest.quality = newBest.quality;
                currentBest.covered = newBest.covered;
                currentBest.condition = candidateRight;
                currentBest.coveredExamples = calculateCoveredExamplesForNumericalCondition(
                        trainSet, candidateRight, attr
                );
                addCondition(currentBest);
            }
        }
    }

    private void induceNominalCondition(
            Attribute attr,
            Rule rule,
            ExampleSet trainSet,
            Set<Integer> uncoveredPositives,
            Set<Integer> coveredByRule,
            ConditionEvaluation currentBest
    ) {
        // weighted case - no precalculated converings
        if (precalculatedCoverings == null) {
            throw new ValueException("Weighted datasets not supported to regression with complex conditions");
        } else {
            induceUnweightedNominalCondition(
                    attr, rule, trainSet, uncoveredPositives, coveredByRule, currentBest
            );
        }
    }

    private void induceUnweightedNominalCondition(
            Attribute attr,
            Rule rule,
            ExampleSet trainSet,
            Set<Integer> uncoveredPositives,
            Set<Integer> coveredByRule,
            ConditionEvaluation currentBest
    ) {
        for (int i = 0; i < attr.getMapping().size(); ++i) {

            IntegerBitSet conditionCovered = precalculatedCoverings.get(attr).get((double) i);


            // evaluate equality condition a = v
            ElementaryCondition candidate =
                    new ElementaryCondition(attr.getName(), new SingletonSet(i, attr.getMapping().getValues()));

            ConditionEvaluation newBest = RegressionConditionEvaluator.checkCandidate(
                    trainSet, rule, candidate, uncoveredPositives, currentBest, params
            );
            if (newBest != null) {
                Logger.log("\tCurrent best: " + candidate + " (quality=" + newBest.quality + "\n", Level.FINEST);
                currentBest.quality = newBest.quality;
                currentBest.covered = newBest.covered;
                currentBest.condition = candidate;
                currentBest.coveredExamples = conditionCovered;
                addCondition(currentBest);
            }

            if (params.areNegatedConditionsEnabled()) {
                induceUnweightedNegatedNominalCondition(
                        trainSet, attr, rule, i, coveredByRule, currentBest);
            }
        }
    }

    private void induceUnweightedNegatedNominalCondition(
            ExampleSet trainSet,
            Attribute attr,
            Rule rule,
            int i,
            Set<Integer> uncoveredPositives,
            ConditionEvaluation currentBest
    ) {
        IntegerBitSet negatedConditionCovered = precalculatedNegatedCoverings.get(attr).get((double) i);

        ElementaryCondition candidate =
                new ElementaryCondition(attr.getName(), new SingletonSet(i, attr.getMapping().getValues()), true);

        ConditionEvaluation newBest = RegressionConditionEvaluator.checkCandidate(
                trainSet, rule, candidate, uncoveredPositives, currentBest, params
        );
        if (newBest != null) {
            Logger.log("\tCurrent best: " + candidate + " (quality=" + newBest.quality + "\n", Level.FINEST);
            currentBest.quality = newBest.quality;
            currentBest.covered = newBest.covered;
            currentBest.condition = candidate;
            currentBest.coveredExamples = negatedConditionCovered;
            addCondition(currentBest);
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
                    if (params.isMeanBasedRegression()) {
                        induceNumericalConditionUsingMeanBasedRegression(
                                attr,
                                rule,
                                trainSet,
                                uncoveredPositives,
                                coveredByRule,
                                currentBest
                        );
                    } else {
                        induceNumericalCondition(
                                attr,
                                rule,
                                trainSet,
                                totals,
                                uncoveredPositives,
                                currentBest
                        );
                    }
                } else {
                    induceNominalCondition(
                            attr,
                            rule,
                            trainSet,
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
