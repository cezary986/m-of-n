package adaa.analytics.rules.logic.induction.conditions.classification;

import adaa.analytics.rules.logic.induction.ConditionEvaluation;
import adaa.analytics.rules.logic.induction.InductionParameters;
import adaa.analytics.rules.logic.induction.conditions.AbstractConditionsInducer;
import adaa.analytics.rules.logic.induction.conditions.helpers.TotalPosNeg;
import adaa.analytics.rules.logic.representation.*;
import com.rapidminer.example.Attribute;
import com.rapidminer.example.Attributes;
import com.rapidminer.example.ExampleSet;
import org.apache.commons.lang.builder.HashCodeBuilder;
import org.apache.commons.math3.util.Combinations;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.stream.Collectors;

public class NominalAttributesConditionsInducer extends AbstractConditionsInducer {

    private class NominalAttributeConditionKey {

        protected Set<String> attributes;
        protected boolean negated;

        NominalAttributeConditionKey(Set<String> attributes, boolean negated) {
            this.attributes = attributes;
            this.negated = negated;
        }

        public Set<String> getAttributes() {
            return attributes;
        }

        public void setAttributes(Set<String> attributes) {
            this.attributes = attributes;
        }

        public void setAttribute(String attribute) {
            this.attributes.add(attribute);
        }

        public boolean equals(Object obj) {
            if (obj == this) {
                return true;
            } else {
                NominalAttributeConditionKey ref = (obj instanceof NominalAttributeConditionKey) ? (NominalAttributeConditionKey) obj : null;
                if (ref != null) {
                    return ((NominalAttributeConditionKey) obj).attributes.equals(this.attributes) && (((NominalAttributeConditionKey) obj).negated == this.negated);
                } else {
                    return false;
                }
            }
        }

        @Override
        public int hashCode() {
            return new HashCodeBuilder().append(attributes).append(negated).toHashCode();
        }

    }

    protected Map<NominalAttributeConditionKey, IntegerBitSet> precalculatedNominalAttributesConditionsCoverings;
    protected Set<NominalAttributeConditionKey> allowedNominalAttributesConditions;

    public NominalAttributesConditionsInducer(ExecutorService pool, InductionParameters params, Map<Attribute, Map<Double, IntegerBitSet>> precalculatedCoverings, Map<Double, IntegerBitSet> P, Map<Double, IntegerBitSet> N) {
        super(pool, params, precalculatedCoverings, P, N);

        allowedNominalAttributesConditions = new HashSet<>();
        precalculatedNominalAttributesConditionsCoverings = new HashMap<>();
    }

    public NominalAttributesConditionsInducer(NominalAttributesConditionsInducer reference) {
        super(reference.pool, reference.params, reference.precalculatedCoverings, reference.P, reference.N);

        allowedNominalAttributesConditions = new HashSet<>();
        allowedNominalAttributesConditions.addAll(reference.allowedNominalAttributesConditions);
        precalculatedNominalAttributesConditionsCoverings = reference.precalculatedNominalAttributesConditionsCoverings;
    }

    @Override
    public List<Future> preprocess(ExampleSet trainSet, List<Attribute> numericalAttributes, List<Attribute> nominalAttributes) {
        Attributes attributes = trainSet.getAttributes();
        List<Set<String>> nominalAttributeConditionsSets = getNominalAttributeConditionsSets(nominalAttributes);

        List<Future> futures = new ArrayList<>();
        Future f = pool.submit(() -> {
            for (Set<String> attributesSet : nominalAttributeConditionsSets) {

                List<String> attributesList = new ArrayList(attributesSet);
                String attrName = attributesList.get(0);
                List<String> allValues = attributes.get(attrName).getMapping().getValues();
                IntegerBitSet coveredExamplesOfValue = new IntegerBitSet(trainSet.size());
                IntegerBitSet coveredExamples = new IntegerBitSet(trainSet.size());
                boolean first_attr;

                for (String value : allValues) {
                    first_attr = true;
                    for (String attr : attributesList) {
                        Attribute attribute = attributes.get(attr);
                        if (first_attr) {
                            coveredExamplesOfValue = precalculatedCoverings.get(attribute)
                                    .get((double) attribute.getMapping().getIndex(value));
                            first_attr = false;
                        } else {
                            coveredExamplesOfValue = coveredExamplesOfValue.getIntersection(precalculatedCoverings.get(attribute)
                                    .get((double) attribute.getMapping().getIndex(value)));
                        }

                    }
                    coveredExamples.addAll(coveredExamplesOfValue);
                }

                NominalAttributeConditionKey nominalAttributeConditionKey = new NominalAttributeConditionKey(attributesSet, false);
                precalculatedNominalAttributesConditionsCoverings.put(nominalAttributeConditionKey, coveredExamples);

                if (params.areNegatedConditionsEnabled() && attributesList.size() == 2){ //&& allValues.size() == 2) {
                    NominalAttributeConditionKey nominalAttributeConditionKeyNegated = new NominalAttributeConditionKey(attributesSet, true);
                    IntegerBitSet coveredExamplesNegated = coveredExamples.clone();
                    coveredExamplesNegated.negate();
                    /*
                    Removing empty values
                    Precalculated covering does not take into account empty values (they are simply not counted). That
                    is why negating is not enough to calculate negated conditions coverage as it will potentially count
                    empty values. Solution is to negate and later manually remove all empty values for each attribute.
                     */
                    for (String attr : attributesList) {
                        Attribute attribute = attributes.get(attr);
                        IntegerBitSet emptyValuesCovered = precalculatedCoverings.get(attribute).get(Double.NaN);
                        coveredExamplesNegated.removeAll(emptyValuesCovered);
                    }
                    //negacja może być tylko dla przypadku 2 atrybutów binarnych a!=b
                    precalculatedNominalAttributesConditionsCoverings.put(nominalAttributeConditionKeyNegated, coveredExamplesNegated);
                }
            }
            allowedNominalAttributesConditions.addAll(precalculatedNominalAttributesConditionsCoverings.keySet());
        });
        futures.add(f);
        return futures;
    }

    private List<Set<String>> getNominalAttributeConditionsSets(List<Attribute> nominalAttributes) {
        //split the attributes into those with the same values
        Map<List<String>, Set<String>> valuesWithAttributes = new HashMap<>();
        List<String> tmpValuesList = new ArrayList<>();
        Set<String> tmpAttributesList = new HashSet<>();
        Set<Set<String>> attributesSubsets;
        for (Attribute attr : nominalAttributes) {

            tmpValuesList = new ArrayList<>(attr.getMapping().getValues());
            Collections.sort(tmpValuesList);
            if (valuesWithAttributes.keySet().contains(tmpValuesList)) {
                valuesWithAttributes.get(tmpValuesList).add(attr.getName());
            } else {
                tmpAttributesList = new HashSet<>();
                tmpAttributesList.add(attr.getName());
                valuesWithAttributes.put(tmpValuesList, tmpAttributesList);
            }
        }
        List<Set<String>> listOfSets = new ArrayList();
        int maxNumberOfAttrsInNominalAttributeCondition = 3;
        for (Set<String> attributesSet : valuesWithAttributes.values()) {
            attributesSubsets = getAttributesSubsets(attributesSet);
            for (Set<String> subSet : attributesSubsets) {
                if (subSet.size() > 1 && subSet.size() <= maxNumberOfAttrsInNominalAttributeCondition) {
                    listOfSets.add(subSet);
                }
            }

        }
        return listOfSets;
    }

    public Set<Set<String>> getAttributesSubsets(Set<String> set) {
        String[] setValues = set.toArray(new String[set.size()]);

        Set<Set<String>> subSets = new HashSet<>();
        for (int k = set.size(); k > 0; k--) {
            Iterator<int[]> itr = new Combinations(set.size(), k).iterator();
            while (itr.hasNext()) {
                Set<String> subset = new HashSet<>();
                int[] indexes = itr.next();
                if (indexes.length > 0) {
                    for (int index : indexes) {
                        subset.add(setValues[index]);
                    }
                    subSets.add(subset);
                }
            }
        }
        return subSets;
    }


    @Override
    public List<Future<Void>> induceConditions(Rule rule, ExampleSet trainSet, double apriori_prec, Set<Integer> uncoveredPositives, Set<Integer> coveredByRule, Map<String, Map<Double, TotalPosNeg>> totals) {
        if (trainSet.getAttributes().getWeight() != null)
            throw new IllegalArgumentException("Nominal attributes conditions are not supported for weighted datasets");

        List<Future<Void>> futures = new ArrayList<>();

        double P = rule.getWeighted_P();
        double N = rule.getWeighted_N();
        double classId = ((SingletonSet) rule.getConsequence().getValueSet()).getValue();


        for (NominalAttributeConditionKey nominalAttributeConditionKey : allowedNominalAttributesConditions) {

            // consider attributes in parallel
            Future future = pool.submit(() -> {

                ConditionEvaluation bestNominalAttributesCondition = new ConditionEvaluation();

                IntegerBitSet conditionCovered = precalculatedNominalAttributesConditionsCoverings.get(nominalAttributeConditionKey);
                double p = conditionCovered.calculateIntersectionSize(rule.getCoveredPositives());
                int toCover_p = conditionCovered.calculateIntersectionSize((IntegerBitSet) coveredByRule, (IntegerBitSet) uncoveredPositives);
                double n = conditionCovered.calculateIntersectionSize((IntegerBitSet) coveredByRule) - p;

                double prec = p / (p + n);

                if (prec > apriori_prec) {
                    // evaluate equality condition a = v
                    double quality = params.getInductionMeasure().calculate(p, n, P, N);
                    NominalAttributesCondition candidateCondition =
                            new NominalAttributesCondition(nominalAttributeConditionKey.getAttributes(), nominalAttributeConditionKey.negated);
                    if (checkCandidate(candidateCondition, classId, P, toCover_p)) {
                        ConditionEvaluation candidate = new ConditionEvaluation();
                        candidate.quality = quality;
                        candidate.covered = p;
                        candidate.condition = candidateCondition;
                        candidate.coveredExamples = conditionCovered;
                        addCondition(candidate);

                        if (candidate.compareTo(bestNominalAttributesCondition) > 0) {
                            Logger.log("\tCurrent best: " + candidateCondition + " (p=" + p + ", n=" + n + ", new_p=" + (double) toCover_p + ", quality=" + quality + "\n", Level.FINEST);
                            bestNominalAttributesCondition.quality = quality;
                            bestNominalAttributesCondition.covered = p;
                            bestNominalAttributesCondition.condition = candidateCondition;
                            bestNominalAttributesCondition.coveredExamples = conditionCovered;
                        }
                    }
                }
                setBestCondition(bestNominalAttributesCondition);
            });
            futures.add(future);
        }
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
                NominalAttributesCondition bestCondition = (NominalAttributesCondition) condition;
                NominalAttributeConditionKey bestKey = new NominalAttributeConditionKey(bestCondition.getAttributes(), bestCondition.isNegated());
                allowedNominalAttributesConditions.remove(bestKey);
            }
        }
    }

    @Override
    public AbstractConditionsInducer clone() {
        return new NominalAttributesConditionsInducer(this);
    }
}
