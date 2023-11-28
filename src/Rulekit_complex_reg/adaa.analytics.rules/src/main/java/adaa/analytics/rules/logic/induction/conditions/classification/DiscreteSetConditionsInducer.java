package adaa.analytics.rules.logic.induction.conditions.classification;

import adaa.analytics.rules.logic.induction.ConditionEvaluation;
import adaa.analytics.rules.logic.induction.InductionParameters;
import adaa.analytics.rules.logic.induction.conditions.AbstractConditionsInducer;
import adaa.analytics.rules.logic.induction.conditions.helpers.TotalPosNeg;
import adaa.analytics.rules.logic.representation.*;
import com.rapidminer.example.Attribute;
import com.rapidminer.example.ExampleSet;
import com.rapidminer.example.Example;
import com.rapidminer.example.table.DataRow;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.logging.Level;

public class DiscreteSetConditionsInducer extends AbstractConditionsInducer {

    protected class NominalValuesSubset {
        protected Set<Double> subset;
        protected IntegerBitSet coveredExamples;

        public NominalValuesSubset(Set<Double> set, IntegerBitSet coveredExamples) {
            this.subset = set;
            this.coveredExamples = coveredExamples;
        }

        public Set<Double> getSubset() {
            return subset;
        }

        public void setSubset(Set<Double> subset) {
            this.subset = subset;
        }

        public IntegerBitSet getCoveredExamples() {
            return coveredExamples;
        }

        public void setCoveredExamples(IntegerBitSet coveredExamples) {
            this.coveredExamples = coveredExamples;
        }

        public void addCoveredExamples(IntegerBitSet toAdd) {
            coveredExamples.addAll(toAdd);
        }

        public boolean equals(Object obj) {
            if (obj == this) {
                return true;
            } else {
                NominalValuesSubset ref = (obj instanceof NominalValuesSubset) ? (NominalValuesSubset) obj : null;
                if (ref != null) {
                    return ((NominalValuesSubset) obj).subset.equals(this.subset) && ((NominalValuesSubset) obj).coveredExamples.equals(this.coveredExamples);
                } else {
                    return false;
                }
            }
        }
    }

    protected Set<Attribute> allowedAttributes;
    protected Map<Attribute, List<NominalValuesSubset>> precalculatedSubsetsCoverings;

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
    public List<Future> preprocess(ExampleSet trainSet, List<Attribute> numericalAttributes, List<Attribute> nominalAttributes) {
        this.allowedAttributes = new HashSet<>(nominalAttributes);
        List<Future> futures = new ArrayList<>();
        for (Attribute attr : nominalAttributes) {
            Future future = pool.submit(() -> {
                Map<Double, IntegerBitSet> attributeCovering = new TreeMap<>();
                List<NominalValuesSubset> nominalValuesSubsetList = new ArrayList<>();
                List<Double> mappingValues = new ArrayList<>();

                for (double val = 0; val < attr.getMapping().size(); val++) {
                    mappingValues.add(val);
                    attributeCovering.put(val, new IntegerBitSet(trainSet.size()));
                }
                // get all distinctive values of attribute
                int id = 0;
                for (Example e : trainSet) {
                    DataRow dr = e.getDataRow();
                    double value = dr.get(attr);

                    // omit missing values
                    if (!Double.isNaN(value)) {
                        attributeCovering.get(value).add(id);
                    }
                    ++id;
                }

                List<Set<Double>> valuesSubsetsList = getValuesSubsets(mappingValues);

                for (Set<Double> subset : valuesSubsetsList) {
                    if (subset.size() > 1) {
                        IntegerBitSet coveredExamples = new IntegerBitSet(trainSet.size());
                        for (Double val : subset) {
                            coveredExamples.addAll(attributeCovering.get(val));
                        }
                        NominalValuesSubset nominalValuesSubset = new NominalValuesSubset(subset, coveredExamples);
                        nominalValuesSubsetList.add(nominalValuesSubset);
                    }
                }

                synchronized (this) {
                    precalculatedSubsetsCoverings.put(attr, nominalValuesSubsetList);
                }
            });
            futures.add(future);
        }

        return futures;
    }

    public static List<Set<Double>> getValuesSubsets(List<Double> superSet) {
        List<Set<Double>> res = new ArrayList<>();
        getSubsetsHelper(superSet, 2, 0, new HashSet<Double>(), res);
        getSubsetsHelper(superSet, 3, 0, new HashSet<Double>(), res);
        return res;
    }

    private static void getSubsetsHelper(List<Double> superSet, int k, int idx, Set<Double> current, List<Set<Double>> solution) {
        //successful stop clause
        if (current.size() == k) {
            solution.add(new HashSet<>(current));
            return;
        }
        //unseccessful stop clause
        if (idx == superSet.size()) return;
        Double x = superSet.get(idx);
        current.add(x);
        //"guess" x is in the subset
        getSubsetsHelper(superSet, k, idx + 1, current, solution);
        current.remove(x);
        //"guess" x is not in the subset
        getSubsetsHelper(superSet, k, idx + 1, current, solution);
    }

    @Override
    public List<Future<Void>> induceConditions(Rule rule, ExampleSet trainSet, double apriori_prec, Set<Integer> uncoveredPositives, Set<Integer> coveredByRule, Map<String, Map<Double, TotalPosNeg>> totals) {
        if (trainSet.getAttributes().getWeight() != null)
            throw new IllegalArgumentException("Discrete Set conditions are not supported for weighted datasets");

        List<Future<Void>> futures = new ArrayList<>();

        double P = rule.getWeighted_P();
        double N = rule.getWeighted_N();
        double classId = ((SingletonSet) rule.getConsequence().getValueSet()).getValue();

        for (Attribute attr : allowedAttributes) {
            Future f = pool.submit(() -> {
                ConditionEvaluation currentBest = new ConditionEvaluation(trainSet);
                for (int i = 0; i < precalculatedSubsetsCoverings.get(attr).size(); ++i) {
                    NominalValuesSubset nominalValuesSubset = precalculatedSubsetsCoverings.get(attr).get(i);
                    IntegerBitSet conditionCovered = nominalValuesSubset.getCoveredExamples();
                    double p = conditionCovered.calculateIntersectionSize(rule.getCoveredPositives());
                    int toCover_p = conditionCovered.calculateIntersectionSize((IntegerBitSet) coveredByRule, (IntegerBitSet) uncoveredPositives);
                    double n = conditionCovered.calculateIntersectionSize((IntegerBitSet) coveredByRule) - p;

                    double prec = p / (p + n);

                    if (prec > apriori_prec) {
                        // evaluate equality condition a = v
                        double quality = params.getInductionMeasure().calculate(p, n, P, N);
                        ElementaryCondition candidateCondition = new ElementaryCondition(
                                attr.getName(),
                                new DiscreteSet(nominalValuesSubset.getSubset(), attr.getMapping().getValues())
                        );

                        if (checkCandidate(candidateCondition, classId, P, toCover_p)) {
                            ConditionEvaluation candidate = new ConditionEvaluation();
                            candidate.quality = quality;
                            candidate.covered = p;
                            candidate.condition = candidateCondition;
                            candidate.coveredExamples = conditionCovered;
                            addCondition(candidate);

                            if (candidate.compareTo(currentBest) > 0) {
                                Logger.log("\tCurrent best: " + candidate + " (p=" + p + ", n=" + n + ", new_p=" + (double) toCover_p + ", quality=" + quality + "\n", Level.FINEST);
                                currentBest.quality = quality;
                                currentBest.covered = p;
                                currentBest.condition = candidateCondition;
                                currentBest.coveredExamples = conditionCovered;
                            }
                        }
                    }
                }
                setBestCondition(currentBest);
            });
            futures.add(f);
        }
        return futures;
    }

    @Override
    public void onBestConditionSelected(ConditionEvaluation conditionEvaluation) {
        if (conditionEvaluation.condition instanceof ElementaryCondition) {
            if ((((ElementaryCondition) (conditionEvaluation.condition)).getValueSet() instanceof SingletonSet) ||
                    (((ElementaryCondition) (conditionEvaluation.condition)).getValueSet() instanceof DiscreteSet)) {
                for (Attribute attr : allowedAttributes) {
                    if (attr.getName().equals(((ElementaryCondition) (conditionEvaluation.condition)).getAttribute())) {
                        allowedAttributes.remove(attr);
                        return;
                    }
                }
            }
        }
    }

    @Override
    public AbstractConditionsInducer clone() {
        return new DiscreteSetConditionsInducer(this);
    }
}
