package adaa.analytics.rules.logic.induction.conditions.regression.alternatives;

import adaa.analytics.rules.logic.induction.ConditionEvaluation;
import adaa.analytics.rules.logic.induction.InductionParameters;
import adaa.analytics.rules.logic.induction.beamsearch.BeamSearch;
import adaa.analytics.rules.logic.induction.beamsearch.SearchResult;
import adaa.analytics.rules.logic.induction.conditions.helpers.TotalPosNeg;
import adaa.analytics.rules.logic.induction.conditions.regression.RegressionConditionEvaluator;
import adaa.analytics.rules.logic.representation.*;
import com.rapidminer.example.Attribute;
import com.rapidminer.example.ExampleSet;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

public class InnerAlternativesInducer extends adaa.analytics.rules.logic.induction.conditions.classification.alternatives.InnerAlternativesInducer {

    private ExampleSet trainExampleSet;

    public InnerAlternativesInducer(ExecutorService pool, InductionParameters params, Map<Attribute, Map<Double, IntegerBitSet>> precalculatedCoverings, Map<Double, IntegerBitSet> P, Map<Double, IntegerBitSet> N) {
        super(pool, params, precalculatedCoverings, P, N);
    }

    @Override
    protected void initializeBeamSearch(
            Rule rule,
            double apriori_prec,
            IntegerBitSet coveredByRule,
            IntegerBitSet uncoveredPositives
    ) {
        this.beamSearch = new BeamSearch<ConditionEvaluation>(this.beamSize) {

            @Override
            protected Comparable evaluateSolution(ConditionEvaluation solution, int indexInBeam) {
                ConditionEvaluation evaluatedSolution = RegressionConditionEvaluator.checkCandidate(
                        getTrainExampleSet(), rule, solution.condition, uncoveredPositives, null, params
                );
                if (evaluatedSolution != null) {
                    evaluatedSolution.coveredExamples = solution.coveredExamples;
                }
                return evaluatedSolution;
            }

            @Override
            public List<ConditionEvaluation> generateNewSolutions(SearchResult<ConditionEvaluation> solution, int indexInBeam) {
                List<ConditionEvaluation> newSolutions = new ArrayList<>();
                for (ConditionEvaluation candidate : beamCandidates[indexInBeam]) {
                    ElementaryCondition candidateCondition = (ElementaryCondition) candidate.condition;
                    CompoundCondition alternative = cloneAlternative((CompoundCondition) solution.value.condition);

                    // ignore candidates having intersection with any of conditions in alternative
                    boolean intersectWithAnyCondition = false;
                    if (candidate.condition instanceof ElementaryCondition) {
                        // check intersections only for elementary conditions - not for numerical attr conditions
                        List<ElementaryCondition> alternativeConditions = alternative.getSubconditions()
                                .stream()
                                .filter((e) -> e instanceof ElementaryCondition)
                                .map((e) -> ((ElementaryCondition) e))
                                .collect(Collectors.toList());
                        for (ElementaryCondition condition : alternativeConditions) {
                            if (condition.getValueSet().intersects(candidateCondition.getValueSet())) {
                                intersectWithAnyCondition = true;
                                break;
                            }
                        }
                    }

                    if (!intersectWithAnyCondition) {
                        alternative.addSubcondition(candidate.condition);
                        ConditionEvaluation newSolution = new ConditionEvaluation();
                        newSolution.condition = alternative;

                        newSolution.coveredExamples = solution.value.coveredExamples.getSum(candidate.coveredExamples);
                        newSolutions.add(newSolution);
                    }
                }
                return newSolutions;
            }

            @Override
            protected void performIteration() {
                super.performIteration();
                // adding more conditions to alternative reduce the search space
                List<SearchResult<ConditionEvaluation>> currentSolutions = getCurrentSolutions();
                for (int i = 0; i < currentSolutions.size(); i++) {
                    List<ConditionBase> solutionSubconditions = ((CompoundCondition) currentSolutions.get(i).value.condition).getSubconditions();
                    ConditionEvaluation tmp = new ConditionEvaluation();
                    tmp.condition = solutionSubconditions.get(solutionSubconditions.size() - 1);
                    beamCandidates[i].remove(tmp);
                }
            }
        };
        // we want to get N best solutions from all iterations no matter the length of the alternative
        this.beamSearch.setSearchForGlobalBest(true);
    }

    @Override
    public ConcurrentLinkedQueue<ConditionEvaluation> getConditions() {
        ConcurrentLinkedQueue queue = new ConcurrentLinkedQueue();
        if (this.beamSearch != null) {
            List<ConditionEvaluation> conditions = this.beamSearch.getGlobalBestSolutions().stream().map((e) -> e.value).collect(Collectors.toList());
            queue.addAll(conditions);
        }
        return queue;
    }

    public ConditionEvaluation getBestCondition() {
        List<ConditionEvaluation> conditions = this.beamSearch.getGlobalBestSolutions().stream().map((e) -> (ConditionEvaluation) e.evaluation).collect(Collectors.toList());
        return conditions.size() > 0 ? conditions.get(0) : null;
    }

    private ExampleSet getTrainExampleSet() {
        return this.trainExampleSet;
    }

    @Override
    protected List<Future<Void>> induceConditions(Rule rule, ExampleSet trainSet, double apriori_prec, Set<Integer> uncoveredPositives, Set<Integer> coveredByRule, Map<String, Map<Double, TotalPosNeg>> totals) {
        this.trainExampleSet = trainSet;
        return super.induceConditions(rule, trainSet, apriori_prec, uncoveredPositives, coveredByRule, totals);
    }
}
