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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.logging.Level;
import java.util.stream.Collectors;

import adaa.analytics.rules.logic.induction.conditions.AbstractConditionInducersPool;
import adaa.analytics.rules.logic.induction.conditions.classification.ConditionsInducersPool;
import adaa.analytics.rules.logic.induction.conditions.regression.RegressionConditionEvaluator;
import adaa.analytics.rules.logic.representation.*;

import com.rapidminer.example.Attribute;
import com.rapidminer.example.ExampleSet;
import jdk.nashorn.internal.runtime.regexp.joni.exception.ValueException;

/**
 * Abstract base class for growing and pruning procedures for all types of rules (classification, regression, survival).
 * 
 * @author Adam Gudys
 * 
 */
public abstract class AbstractFinder implements AutoCloseable {
	
	/**
	 * Rule induction parameters.
	 */
	protected final InductionParameters params;
	
	/**
	 * Number of threads to be used by the induction algorithm.
	 */
	protected int threadCount;
    
	/**
	 * Thread pool to be used by the algorithm.
	 */
	protected ExecutorService pool;
	public ConditionEvaluation eval;

	protected int beamSize = 3;
	public HashMap<String, HashSet<String>> attributesMappings;
	public ConcurrentHashMap<String, HashSet<String>> rulesAttributes = new ConcurrentHashMap<>();
	public ConcurrentHashMap<String, Boolean> hasMofNFlags = new ConcurrentHashMap<>();
	protected AbstractConditionInducersPool mainConditionsInducer;
	
	/**
	 * Initializes induction parameters and thread pool.
	 *
	 * @param params Induction parameters.
	 */
	public AbstractFinder(final InductionParameters params) {
		this.params = params;
		
		threadCount = Runtime.getRuntime().availableProcessors();
		pool = Executors.newFixedThreadPool(threadCount);
	}

	@Override
	public void close() {
		pool.shutdown();
	}

	/**
	 * Adds elementary conditions to the rule premise until termination conditions are fulfilled.
	 * 
	 * @param rule Rule to be grown.
	 * @param dataset Training set.
	 * @param uncovered Collection of examples yet uncovered by the model (positive examples in the classification problems).
	 * @return Number of conditions added.
	 */
	public int grow(
		final Rule rule,
		final ExampleSet dataset,
		final Set<Integer> uncovered) {

		Logger.log("AbstractFinder.grow()\n", Level.FINE);
		int initialConditionsCount = rule.getPremise().getSubconditions().size();
		
		// get current covering
		Covering covering = new Covering();
		rule.covers(dataset, covering, covering.positives, covering.negatives);
		IntegerBitSet covered = new IntegerBitSet(dataset.size());
		covered.addAll(covering.positives);
		covered.addAll(covering.negatives);
//		covered.addAll(rule.getCoveredPositives());
//		covered.addAll(rule.getCoveredNegatives());
		
		// add conditions to rule
		boolean carryOn = true;

		AbstractConditionInducersPool conditionsInducer = mainConditionsInducer.clone();

		do {
			ConditionBase condition = induceCondition(
					rule, dataset, uncovered, covered, null, conditionsInducer);

			if (condition != null) {
				rule.getPremise().addSubcondition(condition);

				Covering lastCov = new Covering();
				lastCov.negatives = new IntegerBitSet(dataset.size());
				lastCov.negatives.addAll(covering.negatives);

				lastCov.positives = new IntegerBitSet(dataset.size());
				lastCov.positives.addAll(covering.positives);

				lastCov.weighted_p = covering.weighted_p;
				lastCov.weighted_n = covering.weighted_n;
				lastCov.weighted_N = covering.weighted_N;
				lastCov.weighted_P = covering.weighted_P;


				covering = new Covering();
				rule.covers(dataset, covering, covering.positives, covering.negatives);
				covered.clear();
				covered.addAll(covering.positives);
				covered.addAll(covering.negatives);

				rule.setCoveringInformation(covering);
				rule.getCoveredPositives().setAll(covering.positives);
				rule.getCoveredNegatives().setAll(covering.negatives);

				rule.setCoveredNegatives(new IntegerBitSet(dataset.size()));
				rule.getCoveredNegatives().setAll(covering.negatives);

				if (covering.weighted_p == 0) {
					RegressionRule newRule = new RegressionRule(new CompoundCondition(), rule.getConsequence());
					newRule.getPremise().addSubcondition(condition);
					IntegerBitSet condCovered = new IntegerBitSet(dataset.size());
					newRule.getPremise().evaluate(dataset, condCovered);

					rule.getPremise().removeSubcondition(condition);
					covering = new Covering();
					rule.covers(dataset, covering, covering.positives, covering.negatives);
					covered.clear();
					covered.addAll(covering.positives);
					covered.addAll(covering.negatives);

					rule.getPremise().addSubcondition(condition);
					covering = new Covering();
					rule.covers(dataset, covering, covering.positives, covering.negatives);
					covered.clear();
					covered.addAll(covering.positives);
					covered.addAll(covering.negatives);

					throw new ValueException("p = 0 after adding conditions");
				}

				rule.updateWeightAndPValue(dataset, covering, params.getVotingMeasure());
				
				Logger.log("Condition " + rule.getPremise().getSubconditions().size() + " added: " 
						+ rule.toString() + "\n", Level.FINER);
				
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
	 * Removes irrelevant conditions from rule using hill-climbing strategy. 
	 * 
	 * @param rule Rule to be pruned.
	 * @param trainSet Training set.
	 * @param uncovered Collection of examples yet uncovered by the model (positive examples in the classification problems).
	 * @return Covering of the rule after pruning.
	 */
	public void prune(
			final Rule rule,
			final ExampleSet trainSet,
			final Set<Integer> uncovered) {
		
		Logger.log("AbstractFinder.prune()\n", Level.FINE);
		
		// check preconditions
		if (rule.getWeighted_p() == Double.NaN || rule.getWeighted_p() == Double.NaN ||
			rule.getWeighted_P() == Double.NaN || rule.getWeighted_N() == Double.NaN) {
			throw new IllegalArgumentException();
		}

		Covering covering = new Covering();
		rule.covers(trainSet, covering, covering.positives, covering.negatives);

		double initialQuality = params.getPruningMeasure().calculate(trainSet, covering);
		boolean continueClimbing = true;
		
		while (continueClimbing) {
			ConditionBase toRemove = null;
			double bestQuality = Double.NEGATIVE_INFINITY;
			
			for (ConditionBase cnd : rule.getPremise().getSubconditions()) {
				// consider only prunable conditions
				if (!cnd.isPrunable()) {
					continue;
				}
				
				// disable subcondition to calculate measure
				cnd.setDisabled(true);
				covering = new Covering();
				rule.covers(trainSet, covering, covering.positives, covering.negatives);
				cnd.setDisabled(false);
				
				double q = params.getPruningMeasure().calculate(trainSet, covering);
				
				if (q > bestQuality) {
					bestQuality = q;
					toRemove = cnd;
				}
			}
			
			// if there is something to remove
			if (bestQuality >= initialQuality) {
				initialQuality = bestQuality;
				rule.getPremise().removeSubcondition(toRemove);
				// stop climbing when only single condition remains
				continueClimbing = rule.getPremise().getSubconditions().size() > 1;
				Logger.log("Condition removed: " + rule + "\n", Level.FINER);
			} else {
				continueClimbing = false;
			}
		}

		covering = new Covering();
		rule.covers(trainSet, covering, covering.positives, covering.negatives);
		rule.setCoveringInformation(covering);

		rule.getCoveredPositives().addAll(covering.positives);
		rule.getCoveredNegatives().addAll(covering.negatives);

		rule.updateWeightAndPValue(trainSet, covering, params.getVotingMeasure());
	}

	/**
	 * Postprocesses a rule.
	 *
	 * @param rule Rule to be postprocessed.
	 * @param dataset Training set.
	 *
	 */
	public void postprocess(
		final Rule rule,
		final ExampleSet dataset) {
	}

	/**
	 * Abstract method representing all procedures which induce an elementary condition.
	 * 
	 * @param rule Current rule.
	 * @param trainSet Training set.
	 * @param uncoveredByRuleset Set of examples uncovered by the model.
	 * @param coveredByRule Set of examples covered by the rule being grown.
	 * @param allowedAttributes Set of attributes that may be used during induction.
	 * @param extraParams Additional parameters.
	 * @return Induced elementary condition.
	 */
	protected abstract ConditionBase induceCondition(
		final Rule rule,
		final ExampleSet trainSet,
		final Set<Integer> uncoveredByRuleset,
		final Set<Integer> coveredByRule, 
		final Set<Attribute> allowedAttributes,
		Object... extraParams);
	
	/**
	 * Maps a set of attribute names to a set of attributes.
	 * 
	 * @param names Set of attribute names.
	 * @param dataset Training set.
	 * @return Set of attributes.
	 */
	protected Set<Attribute> names2attributes(Set<String> names, ExampleSet dataset) {
		Set<Attribute> out = new HashSet<Attribute>();
		for (String s : names) {
			out.add(dataset.getAttributes().get(s));
		}
		return out;
	}

	protected List<ConditionEvaluation> findNBestConditions(List<ConditionEvaluation> list) {
		Collections.sort(list, Collections.reverseOrder());
		sortBestConditionBasedOnType(list);
		if (list.size() <= this.beamSize)
			return list;
		else
			return list.stream().limit(this.beamSize).collect(Collectors.toList());
	}

	protected void sortBestConditionBasedOnType(List<ConditionEvaluation> list) {
		if (list.size() < 2)
			return;

		double bestQuality = list.get(0).quality;
		// get all conditions with same quality
		List<ConditionEvaluation> bestConditions = new ArrayList<>();
		bestConditions.add(list.get(0));
		for (int i = 1; i < list.size(); i++) {
			double quality = list.get(i).quality;
			if (quality != bestQuality)
				break;
			else
				bestConditions.add(list.get(i));
		}
		// sort according to conditions type
		class ConditionSortWrapper {
			public ConditionEvaluation conditionEvaluation;
			public int sortWeight;
			public ConditionSortWrapper(ConditionEvaluation conditionEvaluation) {
				this.conditionEvaluation = conditionEvaluation;
				this.sortWeight = deduceConditionSortingWeight(conditionEvaluation);
			}
		}
		List<ConditionSortWrapper> sortedConditions = new ArrayList<>();
		for (ConditionEvaluation e : bestConditions)
			sortedConditions.add(new ConditionSortWrapper(e));
		Collections.sort(
				sortedConditions,
				(o1, o2) -> o2.sortWeight - o1.sortWeight
		);
		for (int i = 0; i < sortedConditions.size(); i++)
			list.set(i, sortedConditions.get(i).conditionEvaluation);
	}

	protected int deduceConditionSortingWeight(ConditionEvaluation conditionEvaluation) {
		String conditionString = conditionEvaluation.condition.toString();
		if (conditionString.contains("-of-") || conditionString.contains("OR")) {
			// M-of-N conditon or inner alternative condition
			return 0;
		}
		if (conditionString.contains("=")) {
			conditionString = conditionString.split("=")[1];
			if (conditionString.contains("{")) {
				if (conditionString.contains(","))
					// discrete set conditions
					return 2;
				else
					// plain nominal conditions
					return  1;
			}
			if (conditionString.contains("<") || conditionString.contains("(") || conditionString.contains(">") || conditionString.contains(")")) {
				// plain numerical condition or interval condition
				return 1;
			} else  {
				// attribute condition attr1 = attr2
				return 2;
			}
		} else {
			// attribute condition attr1 >/< attr2
			return 2;
		}
	}
}
