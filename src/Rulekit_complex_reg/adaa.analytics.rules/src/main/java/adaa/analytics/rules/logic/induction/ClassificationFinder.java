/*******************************************************************************
 * Copyright (C) 2019 RuleKit Development Team
 *
 * This program is free software: you can redistribute it and/or modify it under the terms of the
 * GNU Affero General Public License as published by the Free Software Foundation, either version 3
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
 * even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 *  Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License along with this program.
 * If not, see http://www.gnu.org/licenses/.
 ******************************************************************************/
package adaa.analytics.rules.logic.induction;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.stream.Collectors;

import adaa.analytics.rules.logic.induction.conditions.AbstractConditionInducersPool;
import adaa.analytics.rules.logic.induction.conditions.classification.ConditionsInducersPool;
import adaa.analytics.rules.logic.induction.conditions.classification.alternatives.InnerAlternativesInducer;
import adaa.analytics.rules.logic.induction.conditions.helpers.TotalPosNeg;
import adaa.analytics.rules.logic.representation.*;

import com.rapidminer.example.Attribute;
import com.rapidminer.example.Example;
import com.rapidminer.example.ExampleSet;
import com.rapidminer.example.table.DataRow;


/**
 * Class for growing and pruning classification rules.
 *
 * @author Adam Gudys
 */
public class ClassificationFinder extends AbstractFinder {
    /**
     * Map of precalculated coverings (time optimization).
     * For each attribute there is a set of distinctive values. For each value there is a bit vector of examples covered.
     */
    protected Map<Attribute, Map<Double, IntegerBitSet>> precalculatedCoverings;
    protected Map<Double, IntegerBitSet> P;
    protected Map<Double, IntegerBitSet> N;

    protected boolean discreteSetConditions;
    protected boolean negatedConditions;
    protected boolean attributesIntervals;
    protected boolean attributesConditions;
    protected boolean nominalAttributesConditions;

    // TODO parametrize
    private int beamSize = 3;

    public HashMap<String, HashSet<String>> attributesMappings;
    public ConcurrentHashMap<String, HashSet<String>> rulesAttributes = new ConcurrentHashMap<>();
    public ConcurrentHashMap<String, Boolean> hasMofNFlags = new ConcurrentHashMap<>();

    /**
     * Initializes induction parameters.
     *
     * @param params Induction parameters.
     */
    public ClassificationFinder(InductionParameters params) {
        super(params);
        MissingValuesHandler.ignore = params.isIgnoreMissing();

        discreteSetConditions = params.areDiscreteSetConditionsEnabled();
        negatedConditions = params.areNegatedConditionsEnabled();
        attributesIntervals = params.areAttributesIntervalsEnabled();
        attributesConditions = params.areAttributesConditionsEnabled();
        nominalAttributesConditions = params.areNominalAttributesConditionsEnabled();

        this.mainConditionsInducer = null;
    }

    /**
     * If example set is unweighted, method precalculates conditions coverings and stores
     * them as bit vectors in @see precalculatedCoverings field.
     *
     * @param trainSet Training set.
     */
    public void preprocess(ExampleSet trainSet) {

        // do nothing for weighted datasets
        if (trainSet.getAttributes().getWeight() != null) {
            return;
        }

        precalculatedCoverings = new HashMap<>();

        List<Attribute> numericalAttributes = new ArrayList<>();
        List<Attribute> nominalAttributes = new ArrayList<>();
        List<Future> futures = new ArrayList<>();

        // iterate over all allowed decision attributes
        for (Attribute attr : trainSet.getAttributes()) {
            Future f = pool.submit(() -> {
                Map<Double, IntegerBitSet> attributeCovering = new TreeMap<Double, IntegerBitSet>();

                // check if attribute is nominal
                if (attr.isNominal()) {
                    // prepare bit vectors
                    for (int val = 0; val != attr.getMapping().size(); ++val) {
                        attributeCovering.put((double) val, new IntegerBitSet(trainSet.size()));
                    }
//                  // special bin for empty values
                    attributeCovering.put(Double.NaN, new IntegerBitSet(trainSet.size()));
                    // get all distinctive values of attribute
                    int id = 0;
                    for (Example e : trainSet) {
                        DataRow dr = e.getDataRow();
                        double value = dr.get(attr);

                        if (!Double.isNaN(value)) {
                            attributeCovering.get(value).add(id);
                        } else {
                            attributeCovering.get(Double.NaN).add(id);
                        }
                        ++id;
                    }
                    synchronized (this) {
                        nominalAttributes.add(attr);
                    }
                } else {
                    // prepare bit vectors
                    // get all distinctive values of attribute
                    int id = 0;
                    attributeCovering.put(Double.NaN, new IntegerBitSet(trainSet.size()));
                    for (Example e : trainSet) {
                        DataRow dr = e.getDataRow();
                        double value = dr.get(attr);
                        if (!attributeCovering.containsKey(value)) {
                            attributeCovering.put(value, new IntegerBitSet(trainSet.size()));
                        }

                        if (!Double.isNaN(value)) {
                            attributeCovering.get(value).add(id);
                        } else {
                            // special bin for empty values
                            attributeCovering.get(Double.NaN).add(id);
                        }
                        ++id;
                    }
                    synchronized (this) {
                        numericalAttributes.add(attr);
                    }
                }

                synchronized (this) {
                    precalculatedCoverings.put(attr, attributeCovering);
                }
            });

            futures.add(f);
        }

        try {
            for (Future f : futures) {
                f.get();
            }
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }

        this.P = new HashMap<>();
        this.N = new HashMap<>();
        int id = 0;
        IntegerBitSet all = new IntegerBitSet(trainSet.size());
        for (Example example : trainSet) {
            double labelValue = example.getLabel();
            this.P.computeIfAbsent(labelValue, (k) -> new IntegerBitSet(trainSet.size()));
            this.P.get(labelValue).add(id);
            all.add(id);
            id++;
        }
        for (Map.Entry<Double, IntegerBitSet> entry : this.P.entrySet()) {
            IntegerBitSet tmp = this.P.get(entry.getKey());
            IntegerBitSet N = all.clone();
            N.removeAll(tmp);
            this.N.put(entry.getKey(), N);
        }

        futures.clear();

        mainConditionsInducer = new ConditionsInducersPool(pool, params, precalculatedCoverings, P, N);
        futures.addAll(mainConditionsInducer.preprocess(trainSet, numericalAttributes, nominalAttributes));
        try {
            for (Future f : futures) {
                f.get();
            }
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
    }


    /**
     * Adds elementary conditions to the rule premise until termination conditions are fulfilled.
     *
     * @param rule      Rule to be grown.
     * @param dataset   Training set.
     * @param uncovered Set of positive examples yet uncovered by the model.
     * @return Number of conditions added.
     */
    public int grow(
            final Rule rule,
            final ExampleSet dataset,
            final Set<Integer> uncovered) {

        Logger.log("ClassificationFinder.grow()\n", Level.FINE);

        int initialConditionsCount = rule.getPremise().getSubconditions().size();
        this.rulesAttributes.put(((ClassificationRule) rule).getUuid(), new HashSet<>());
        //HashSet<Integer> covered = new HashSet<Integer>();
        Set<Integer> covered = new IntegerBitSet(dataset.size());
        covered.addAll(rule.getCoveredPositives());
        covered.addAll(rule.getCoveredNegatives());

        // bit vectors for faster operations on coverings
        IntegerBitSet conditionCovered = new IntegerBitSet(dataset.size());

        // add conditions to rule
        boolean carryOn = true;
        Rule currentRule = new ClassificationRule();
        currentRule.copyFrom(rule);

        // each rule growing creates a clone of inducersPool objects containing copy precalculated statistics to ensure
        // thread safety
        AbstractConditionInducersPool conditionsInducer = mainConditionsInducer.clone();

        do {
            ConditionBase condition = induceCondition(rule, dataset, uncovered, covered, null, conditionsInducer);

            if (condition != null) {
                if (params.getSelectBestCandidate()) {
                    carryOn = tryAddCondition(currentRule, rule, condition, dataset, covered, conditionCovered);
                } else {
                    carryOn = tryAddCondition(rule, null, condition, dataset, covered, conditionCovered);
                }

                if (params.getMaxGrowingConditions() > 0) {
                    if (rule.getPremise().getSubconditions().size() - initialConditionsCount >=
                            params.getMaxGrowingConditions() * dataset.getAttributes().size()) {
                        carryOn = false;
                    }
                }

            } else {
                carryOn = false;
            }

        } while (carryOn);

        // if rule has been successfully grown
        int addedConditionsCount = rule.getPremise().getSubconditions().size() - initialConditionsCount;
        rule.setInducedContitionsCount(addedConditionsCount);
        return addedConditionsCount;
    }


    /**
     * Removes irrelevant conditions from the rule using hill-climbing strategy.
     *
     * @param rule     Rule to be pruned.
     * @param trainSet Training set.
     * @return Updated covering object.
     */
    public void prune(final Rule rule, final ExampleSet trainSet, final Set<Integer> uncovered) {
        Logger.log("ClassificationFinder.prune()\n", Level.FINE);

        // check preconditions
        if (rule.getWeighted_p() == Double.NaN || rule.getWeighted_p() == Double.NaN ||
                rule.getWeighted_P() == Double.NaN || rule.getWeighted_N() == Double.NaN) {
            throw new IllegalArgumentException();
        }

        int examplesCount = trainSet.size();
        int conditionsCount = rule.getPremise().getSubconditions().size();
        int maskLength = (trainSet.size() + Long.SIZE - 1) / Long.SIZE;
        long[] masks = new long[conditionsCount * maskLength];
        long[] labelMask = new long[maskLength];

        double P = rule.getWeighted_P();
        double N = rule.getWeighted_N();

        int[] conditionsPerExample = new int[trainSet.size()];

        for (int i = 0; i < trainSet.size(); ++i) {
            Example e = trainSet.getExample(i);
            int wordId = i / Long.SIZE;
            int wordOffset = i % Long.SIZE;

            if (rule.getConsequence().evaluate(e)) {
                labelMask[wordId] |= 1L << wordOffset;
            }

            for (int m = 0; m < conditionsCount; ++m) {
                ConditionBase cnd = rule.getPremise().getSubconditions().get(m);
                if (cnd.evaluate(e)) {
                    masks[m * maskLength + wordId] |= 1L << wordOffset;
                    ++conditionsPerExample[i];
                }
            }
        }

        IntegerBitSet removedConditions = new IntegerBitSet(conditionsCount);
        int conditionsLeft = rule.getPremise().getSubconditions().size();

        ContingencyTable ct = new ContingencyTable();
        rule.covers(trainSet, ct);
        double initialQuality = params.getPruningMeasure().calculate(trainSet, ct);
        boolean continueClimbing = true;
        boolean weighting = (trainSet.getAttributes().getWeight() != null);

        while (continueClimbing) {
            int toRemove = -1;
            double bestQuality = Double.NEGATIVE_INFINITY;
            final int conditionsLeft_final = conditionsLeft;
            final int[] conditionsPerExample_final = conditionsPerExample;

            List<Future<Double>> futures = new ArrayList<Future<Double>>(conditionsCount);

            // distribute over threads
            for (int cid = 0; cid < conditionsCount; ++cid) {
                final int fcid = cid;
                Future<Double> f = pool.submit(() -> {

                    ConditionBase cnd = rule.getPremise().getSubconditions().get(fcid);
                    // ignore already removed conditions
                    if (removedConditions.contains(fcid)) {
                        return Double.NEGATIVE_INFINITY;
                    }

                    // consider only prunable conditions
                    if (!cnd.isPrunable()) {
                        return Double.NEGATIVE_INFINITY;
                    }

                    double p = 0;
                    double n = 0;

                    // iterate over all words
                    int id = 0;
                    for (int wordId = 0; wordId < maskLength; ++wordId) {

                        long word = masks[fcid * maskLength + wordId];
                        long filteredWord = 0;

                        for (int wordOffset = 0; wordOffset < Long.SIZE && id < examplesCount; ++wordOffset, ++id) {
                            // an example is covered by rule after condition removal in two cases:
                            // - it is covered by all conditions prior the removal
                            // - it is covered by all conditions except the one being removed

                            if ((conditionsPerExample_final[id] == conditionsLeft_final) ||
                                    ((conditionsPerExample_final[id] == conditionsLeft_final - 1) && (word & (1L << wordOffset)) == 0)) {
                                filteredWord |= 1L << wordOffset;
                            }
                        }

                        long labelWord = labelMask[wordId];
                        long posWord = filteredWord & labelWord;
                        long negWord = filteredWord & ~labelWord;

                        if (weighting) {
                            // weighted - iterate over bits and sum weights
                            for (int wordOffset = 0; wordOffset < Long.SIZE; ++wordOffset) {
                                if ((posWord & (1L << wordOffset)) != 0) {
                                    p += trainSet.getExample(wordId * Long.SIZE + wordOffset).getWeight();
                                } else if ((negWord & (1L << wordOffset)) != 0) {
                                    n += trainSet.getExample(wordId * Long.SIZE + wordOffset).getWeight();
                                }
                            }
                        } else {

                            // not weighted - bit operations
                            p += Long.bitCount(posWord);
                            n += Long.bitCount(negWord);
                        }
                    }

                    double q = params.getPruningMeasure().calculate(p, n, P, N);
                    return q;
                });

                futures.add(f);
            }

            // gather results for conditions
            for (int cid = 0; cid < futures.size(); ++cid) {
                Future<Double> f = futures.get(cid);
                try {
                    Double q = f.get();
                    if (q > bestQuality) {
                        bestQuality = q;
                        toRemove = cid;
                    }

                } catch (InterruptedException | ExecutionException e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
            }

            // if there is something to remove
            if (bestQuality >= initialQuality) {
                initialQuality = bestQuality;
                removedConditions.add(toRemove);
                --conditionsLeft;

                // decrease counters for examples covered by removed condition
                int id = 0;
                for (int wordId = 0; wordId < maskLength; ++wordId) {
                    long word = masks[toRemove * maskLength + wordId];
                    for (int wordOffset = 0; wordOffset < Long.SIZE && id < examplesCount; ++wordOffset, ++id) {

                        if ((word & (1L << wordOffset)) != 0) {
                            --conditionsPerExample[id];
                        }
                    }
                }

                if (conditionsLeft == 1) {
                    continueClimbing = false;
                }
            } else {
                continueClimbing = false;
            }
        }

        CompoundCondition prunedPremise = new CompoundCondition();

        for (int cid = 0; cid < conditionsCount; ++cid) {
            if (!removedConditions.contains(cid)) {
                prunedPremise.addSubcondition(rule.getPremise().getSubconditions().get(cid));
            }
        }

        rule.setPremise(prunedPremise);

        ct = new ContingencyTable();
        IntegerBitSet positives = new IntegerBitSet(trainSet.size());
        IntegerBitSet negatives = new IntegerBitSet(trainSet.size());

        rule.covers(trainSet, ct, positives, negatives);

        rule.setWeighted_p(ct.weighted_p);
        rule.setWeighted_n(ct.weighted_n);
        rule.setCoveredPositives(positives);
        rule.setCoveredNegatives(negatives);

        rule.updateWeightAndPValue(trainSet, ct, params.getVotingMeasure());

    }

    private Map<String, Map<Double, TotalPosNeg>> calculateTotals(
            ExampleSet trainSet,
            Attribute weightAttr,
            Set<Integer> coveredPositives,
            Set<Integer> uncoveredPositives,
            Set<Integer> coveredByRule
    ) {
        Map<String, Map<Double, TotalPosNeg>> totals = new TreeMap<>();

        for (Attribute attr : trainSet.getAttributes()) {
            if (attr.isNumerical()) {
                Map<Double, TotalPosNeg> attrTotal = new HashMap<>();
                // get all distinctive values of attribute
                for (int id : coveredByRule) {
                    DataRow dr = trainSet.getExample(id).getDataRow();
                    double val = dr.get(attr);

                    // exclude missing values from keypoints
                    if (Double.isNaN(val)) {
                        continue;
                    }

                    TotalPosNeg tot = attrTotal.computeIfAbsent(val, (k) -> new TotalPosNeg());
                    double w = (weightAttr != null) ? dr.get(weightAttr) : 1.0;

                    // put to proper bin depending of class label
                    if (coveredPositives.contains(id)) {
                        tot.p += w;
                        if (uncoveredPositives.contains(id)) {
                            ++tot.toCover_p;
                        }
                    } else {
                        tot.n += w;
                    }
                }
                totals.put(attr.getName(), attrTotal);
            }
        }
        return totals;
    }

    /**
     * Induces an elementary condition.
     *
     * @param rule               Current rule.
     * @param trainSet           Training set.
     * @param uncoveredPositives Set of positive examples uncovered by the model.
     * @param coveredByRule      Set of examples covered by the rule being grown.
     * @param allowedAttributes  Set of attributes that may be used during induction.
     * @param extraParams        Additional parameters.
     * @return Induced elementary condition.
     */
    @Override
    protected ConditionBase induceCondition(
            Rule rule,
            ExampleSet trainSet,
            Set<Integer> uncoveredPositives,
            Set<Integer> coveredByRule,
            Set<Attribute> allowedAttributes,
            Object... extraParams) {

        ConditionsInducersPool conditionsInducer =  (ConditionsInducersPool) extraParams[0];
        List<ConditionEvaluation> best;
        double P = rule.getWeighted_P();
        double N = rule.getWeighted_N();
        double apriori_prec = params.isControlAprioriPrecision()
                ? P / (P + N)
                : Double.MIN_VALUE;


        Map<String, Map<Double, TotalPosNeg>> totals = calculateTotals(trainSet, trainSet.getAttributes().getWeight(),
                rule.getCoveredPositives(), uncoveredPositives, coveredByRule);

        List<Future<Void>> futures = conditionsInducer.induceConditions(
                rule, trainSet, apriori_prec, uncoveredPositives, coveredByRule, totals
        );

        try {
            for (Future<Void> f : futures) {
                f.get();
            }
        } catch (InterruptedException | ExecutionException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        ConditionEvaluation selectedCondition = null;
        List<ConditionEvaluation> selectedConditions;
        if (this.attributesMappings != null) {
            // M-of-Nc
            HashSet<String> attributesAlreadyInRule = this.rulesAttributes.get(((ClassificationRule) rule).getUuid());
            best = conditionsInducer.getAllConditions();
            Collections.sort(best, Collections.reverseOrder());
            sortBestConditionBasedOnType(best);
            selectedConditions = best.stream().limit(this.beamSize).collect(Collectors.toList());
            int bestConditionIndex = 0;
            while (bestConditionIndex < best.size()) {
                selectedCondition = best.get(bestConditionIndex);
                if (!this.hasMofNFlags.containsKey(((ClassificationRule) rule).getUuid()) && !selectedCondition.toString().contains("-of-")) {
                    break;
                } else {
                    String conditionAttribute = (String) selectedCondition.condition.getAttributes().toArray()[0];
                    HashSet<String> mappedAttributes = this.attributesMappings.get(conditionAttribute);
                    if (mappedAttributes != null) {
                        boolean skipCondition = false;
                        for (String mappedAttr : mappedAttributes) {
                            if (attributesAlreadyInRule.contains(mappedAttr)) {
                                skipCondition = true;
                                break;
                            }
                        }
                        if (!skipCondition) {
                            break;
                        } else {
                            selectedCondition = null;
                        }
                    } else {
                        break;
                    }
                    bestConditionIndex ++;
                }
            }
        } else {
            best = conditionsInducer.getBestInducedConditions();
            selectedConditions = findNBestConditions(best);
            selectedCondition = selectedConditions.size() > 0 ? selectedConditions.get(0) : null;
        }


        // post induction
        if (selectedCondition != null && params.isInnerAlternativesEnabled()) {
            List<ConditionEvaluation> candidates = conditionsInducer.getNumericalInducedConditions(); // select only plain, numerical attr conditions and intervals
            candidates.remove(selectedCondition);
            List<ConditionEvaluation> postInducedConditions = postConditionInducing(
                    rule, trainSet, uncoveredPositives, coveredByRule, candidates, selectedCondition, totals, apriori_prec
            );
            if (postInducedConditions.size() > 0) {
                selectedConditions.addAll(postInducedConditions);
                selectedConditions = findNBestConditions(selectedConditions);
                selectedCondition = selectedConditions.size() > 0 ? selectedConditions.get(0) : null;
            }
        }
        if (selectedCondition != null && selectedCondition.condition != null) {
            conditionsInducer.onBestConditionSelected(selectedCondition);
        }
        if (selectedCondition != null)  {
            if (rule.getPremise().getSubconditions().contains(selectedCondition.condition))
                return null;
            else {
                if (this.attributesMappings != null) {
                    String conditionAttribute = (String) selectedCondition.condition.getAttributes().toArray()[0];
                    HashSet<String> mappedAttributes = this.attributesMappings.get(conditionAttribute);
                    if (mappedAttributes != null) {
                        this.rulesAttributes.get(rule.getUuid()).addAll(
                                mappedAttributes
                        );
                    } else{
                        this.rulesAttributes.get(rule.getUuid()).add(conditionAttribute);
                    }
                }
                if (selectedCondition.condition.toString().contains("-of-"))
                    this.hasMofNFlags.put(rule.getUuid(), true);
                return selectedCondition.condition;
            }
        }
        else
            return null;
    }

    private List<ConditionEvaluation> postConditionInducing(
            Rule rule,
            ExampleSet trainSet,
            Set<Integer> uncoveredPositives,
            Set<Integer> coveredByRule,
            List<ConditionEvaluation> candidates,
            ConditionEvaluation startingPoint,
            Map<String, Map<Double, TotalPosNeg>> totals,
            double apriori_prec
    ) {
        if (trainSet.getAttributes().get((String) startingPoint.condition.getAttributes().toArray()[0]).isNumerical()) {
            InnerAlternativesInducer innerAlternativesInducer = new InnerAlternativesInducer(
                    pool, params, precalculatedCoverings, this.P, this.N
            );

            innerAlternativesInducer.setStartingPoints(Collections.singletonList(startingPoint));
            List<ConditionEvaluation> tmp = new ArrayList<>(candidates);
            innerAlternativesInducer.setCandidates(tmp);
            List<Future<Void>> futures = innerAlternativesInducer
                    .induce(rule, trainSet, apriori_prec, uncoveredPositives, coveredByRule, totals);
            List<ConditionEvaluation> conditions = new ArrayList<>();
            try {
                for (Future<Void> f : futures)
                    f.get();
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
            conditions.addAll(innerAlternativesInducer.getConditions());
            return conditions;
        } else {
            return new ArrayList<>();
        }
    }

    /***
     * Makes an attempt to add the condition to the rule.
     *
     * @param currentRule Rule to be updated.
     * @param bestRule Best rule found up to now. Use null value if not needed.
     * @param condition Condition to be added.
     * @param trainSet Training set.
     * @param covered Set of examples covered by the rules.
     * @param conditionCovered Bit vector of examples covered by the condition.
     * @return Flag indicating whether condition has been added successfully.
     */
    public boolean tryAddCondition(
            final Rule currentRule,
            final Rule bestRule,
            final ConditionBase condition,
            final ExampleSet trainSet,
            final Set<Integer> covered,
            final IntegerBitSet conditionCovered) {

        boolean carryOn = true;
        boolean add = false;
        ContingencyTable ct = new ContingencyTable();


        if (condition != null) {
            conditionCovered.clear();
            condition.evaluate(trainSet, conditionCovered);

            // calculate  quality before addition
            ct.weighted_P = currentRule.getWeighted_P();
            ct.weighted_N = currentRule.getWeighted_N();
            ct.weighted_p = currentRule.getWeighted_p();
            ct.weighted_n = currentRule.getWeighted_n();

            double qualityBefore = params.getInductionMeasure().calculate(trainSet, ct);

            if (trainSet.getAttributes().getWeight() != null) {
                // calculate weights

            } else {
                ct.weighted_p = currentRule.getCoveredPositives().calculateIntersectionSize(conditionCovered);
                ct.weighted_n = currentRule.getCoveredNegatives().calculateIntersectionSize(conditionCovered);
            }

            // analyse stopping criteria
            double adjustedMinCov = Math.min(
                    params.getMinimumCovered(),
                    Math.max(1.0, 0.2 * ct.weighted_P));

            if (ct.weighted_p < adjustedMinCov) {
                if (currentRule.getPremise().getSubconditions().size() == 0) {
                    // special case of empty rule - add condition anyway
                    //		add = true;
                }
                carryOn = false;
            } else {
                // exact rule
                if (ct.weighted_n == 0) {
                    carryOn = false;
                }
                add = true;
            }

            // update coverage if condition was added
            if (add) {

                // recalculate quality
                double qualityAfter = params.getInductionMeasure().calculate(trainSet, ct);

                if (bestRule != null) {
                    if (qualityAfter > qualityBefore) {
                        // quality increase
                        double bestQuality = params.getInductionMeasure().calculate(
                                bestRule.getWeighted_p(), bestRule.getWeighted_n(), bestRule.getWeighted_P(), bestRule.getWeighted_N());

                        if (bestRule.getPremise() != currentRule.getPremise() && qualityAfter > bestQuality) {
                            // if current is better then previous best and has different premise
                            bestRule.copyFrom(currentRule);
                        }
                    } else {
                        // quality drop - local maximum found
                        if (currentRule.getPremise() == bestRule.getPremise()) {
                            // store current state in best rule
                            bestRule.copyFrom(currentRule);

                            // fork rules - make deep copy of selected components
                            currentRule.setPremise(new CompoundCondition());
                            currentRule.getPremise().getSubconditions().addAll(bestRule.getPremise().getSubconditions());
                            currentRule.setCoveredPositives(bestRule.getCoveredPositives().clone());
                            currentRule.setCoveredNegatives(bestRule.getCoveredNegatives().clone());
                        }
                    }
                }

                currentRule.getPremise().getSubconditions().add(condition);

                covered.retainAll(conditionCovered);
                currentRule.getCoveredPositives().retainAll(conditionCovered);
                currentRule.getCoveredNegatives().retainAll(conditionCovered);

                currentRule.setWeighted_p(ct.weighted_p);
                currentRule.setWeighted_n(ct.weighted_n);

                currentRule.updateWeightAndPValue(trainSet, ct, params.getVotingMeasure());

                Logger.log("Condition " + currentRule.getPremise().getSubconditions().size() + " added: "
                        + currentRule.toString() + " " + currentRule.printStats() + "\n", Level.FINER);
            }
        } else {
            carryOn = false;
        }

        // best is current and has not been updated from the beginning
        if (carryOn == false && bestRule != null) {
            double bestQuality = params.getInductionMeasure().calculate(
                    bestRule.getWeighted_p(), bestRule.getWeighted_n(), bestRule.getWeighted_P(), bestRule.getWeighted_N());
            double currentQuality = params.getInductionMeasure().calculate(
                    currentRule.getWeighted_p(), currentRule.getWeighted_n(), currentRule.getWeighted_P(), currentRule.getWeighted_N());

            if (currentQuality > bestQuality) {
                bestRule.copyFrom(currentRule);
            }
        }

        return carryOn;
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
}


