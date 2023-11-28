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

import adaa.analytics.rules.logic.induction.conditions.regression.ConditionsInducersPool;
import adaa.analytics.rules.logic.induction.conditions.helpers.TotalPosNeg;
import adaa.analytics.rules.logic.induction.conditions.regression.alternatives.InnerAlternativesInducer;
import adaa.analytics.rules.logic.representation.*;

import com.rapidminer.example.Attribute;
import com.rapidminer.example.Example;
import com.rapidminer.example.ExampleSet;
import com.rapidminer.example.table.DataRow;

import java.security.InvalidParameterException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.stream.Collectors;

/**
 * Algorithm for growing and pruning regression rules.
 * @author Adam
 *
 */
public class RegressionFinder extends AbstractFinder {
	protected boolean discreteSetConditions;
	protected boolean negatedConditions;
	protected boolean attributesIntervals;
	protected boolean attributesConditions;
	protected boolean nominalAttributesConditions;

	private int beamSize = 3;

	public HashMap<String, HashSet<String>> attributesMappings;
	public ConcurrentHashMap<String, HashSet<String>> rulesAttributes = new ConcurrentHashMap<>();
	public ConcurrentHashMap<String, Boolean> hasMofNFlags = new ConcurrentHashMap<>();


	protected Map<Attribute, Map<Double, IntegerBitSet>> precalculatedCoverings;

	public RegressionFinder(final InductionParameters params) {
		super(params);
		discreteSetConditions = false; //params.areDiscreteSetConditionsEnabled();
		negatedConditions = false; //params.areNegatedConditionsEnabled();
		attributesIntervals = false; //params.areAttributesIntervalsEnabled();
		attributesConditions = false; //params.areAttributesConditionsEnabled();
		nominalAttributesConditions = false;//params.areNominalAttributesConditionsEnabled();

		beamSize = params.getInnerAlternativesSearchBeamSize();
		this.mainConditionsInducer = null;

		RegressionRule.setUseMean(params.isMeanBasedRegression());
	}

	private Map<String, Map<Double, TotalPosNeg>> calculateTotals(
			ExampleSet trainSet,
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
					attrTotal.computeIfAbsent(val, (k) -> new TotalPosNeg());
				}
				totals.put(attr.getName(), attrTotal);
			}
		}
		return totals;
	}
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

		mainConditionsInducer = new ConditionsInducersPool(pool, params, precalculatedCoverings, null, null);
		futures.addAll(mainConditionsInducer.preprocess(trainSet, numericalAttributes, nominalAttributes));
		try {
			for (Future f : futures) {
				f.get();
			}
		} catch (InterruptedException | ExecutionException e) {
			e.printStackTrace();
		}
	}

	@Override
	protected ConditionBase induceCondition(
		final Rule rule,
		final ExampleSet trainSet,
		final Set<Integer> uncovered, 
		final Set<Integer> covered, 
		final Set<Attribute> allowedAttributes,
		Object... extraParams) {
			if (!this.rulesAttributes.containsKey(rule.getUuid())) {
				this.rulesAttributes.put(rule.getUuid(), new HashSet<>());
			}
			// each rule growing creates a clone of inducersPool objects containing copy precalculated statistics to ensure
			// thread safety
			ConditionsInducersPool conditionsInducer = (ConditionsInducersPool) extraParams[0];
			Map<String, Map<Double, TotalPosNeg>> totals = calculateTotals(trainSet, covered);
			List<Future<Void>> futures = conditionsInducer.induceConditions(rule, trainSet, 0.0, uncovered, covered, totals);

			try {
				for (Future<Void> f : futures) {
					f.get();
				}
			} catch (InterruptedException | ExecutionException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			List<ConditionEvaluation> best;
			ConditionEvaluation selectedCondition = null;
			List<ConditionEvaluation> selectedConditions;
			if (this.attributesMappings != null) {
				// M-of-Nc
				HashSet<String> attributesAlreadyInRule = this.rulesAttributes.get(rule.getUuid());
				best = conditionsInducer.getAllConditions();
				Collections.sort(best, Collections.reverseOrder());
				sortBestConditionBasedOnType(best);
				selectedConditions = best.stream().limit(this.beamSize).collect(Collectors.toList());
				int bestConditionIndex = 0;
				while (bestConditionIndex < best.size()) {
					selectedCondition = best.get(bestConditionIndex);
					if (!this.hasMofNFlags.containsKey(rule.getUuid()) && !selectedCondition.toString().contains("-of-")) {
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
						rule, trainSet, uncovered, covered, candidates, selectedCondition, totals, 0
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
				if (rule.getPremise().getSubconditions().contains(selectedCondition.condition) || selectedCondition.coveredExamples.size() == 0)
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
					if (rule.getPremise().getSubconditions().contains(selectedCondition.condition)) {
						return null;
					}
					if (selectedCondition.condition.toString().contains("-of-"))
						this.hasMofNFlags.put(rule.getUuid(), true);
					this.eval = selectedCondition;
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
					pool, params, precalculatedCoverings, null, null
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

	protected boolean checkCandidate(
			ExampleSet dataset,
			Rule rule,
			ConditionBase candidate,
			Set<Integer> uncovered,
			ConditionEvaluation currentBest) {

		try {
			Logger.log("Evaluating candidate: " + candidate, Level.FINEST);

			CompoundCondition newPremise = new CompoundCondition();
			newPremise.getSubconditions().addAll(rule.getPremise().getSubconditions());
			newPremise.addSubcondition(candidate);

			Rule newRule = (Rule) rule.clone();
			newRule.setPremise(newPremise);


			Covering cov = new Covering();
			newRule.covers(dataset, cov, cov.positives, cov.negatives);

			double new_p = 0, new_n = 0;

			if (dataset.getAttributes().getWeight() == null) {
				// unweighted examples
				new_p = SetHelper.intersectionSize(uncovered, cov.positives);
				new_n =	SetHelper.intersectionSize(uncovered, cov.negatives);
			} else {
				// calculate weights of newly covered examples
				for (int id : cov.positives) {
					new_p += uncovered.contains(id) ? dataset.getExample(id).getWeight() : 0;
				}
				for (int id : cov.negatives) {
					new_n += uncovered.contains(id) ? dataset.getExample(id).getWeight() : 0;
				}
			}

			if (checkCoverage(cov.weighted_p, cov.weighted_n, new_p, new_n, rule.getWeighted_P(), rule.getWeighted_N())) {
				double quality = params.getInductionMeasure().calculate(dataset, cov);
				
				Logger.log(", q=" + quality, Level.FINEST);

				if (quality > currentBest.quality || (quality == currentBest.quality && new_p + new_n > currentBest.covered)) {
					currentBest.quality = quality;
					currentBest.condition = candidate;
					currentBest.covered = new_p + new_n;
					currentBest.covering = cov;
					Logger.log(", approved!\n", Level.FINEST);
					//rule.setWeight(quality);
					return true;
				}
			}

			Logger.log("\n", Level.FINEST);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return false;
	}


	boolean checkCoverage(double p, double n, double new_p, double new_n, double P, double N) {
		return ((new_p + new_n) >= params.getAbsoluteMinimumCovered(P + N)) &&
				((p + n) >= params.getAbsoluteMinimumCoveredAll(P + N));
	}
}
