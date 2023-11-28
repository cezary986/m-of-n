package adaa.analytics.rules.logic.induction.conditions.classification.alternatives;

import adaa.analytics.rules.logic.induction.ConditionEvaluation;
import adaa.analytics.rules.logic.induction.InductionParameters;
import adaa.analytics.rules.logic.induction.beamsearch.BeamSearch;
import adaa.analytics.rules.logic.induction.beamsearch.SearchResult;
import adaa.analytics.rules.logic.induction.conditions.AbstractConditionsPostInducer;
import adaa.analytics.rules.logic.induction.conditions.helpers.TotalPosNeg;
import adaa.analytics.rules.logic.quality.IQualityMeasure;
import adaa.analytics.rules.logic.representation.*;
import com.rapidminer.example.Attribute;
import com.rapidminer.example.ExampleSet;

import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

public class InnerAlternativesInducer extends AbstractConditionsPostInducer {

    protected List<ConditionEvaluation> alternatives;
    protected List<ConditionEvaluation> startingPoints;
    protected BeamSearch<ConditionEvaluation> beamSearch;
    protected List<ConditionEvaluation>[] beamCandidates;
    protected double minimumCovered;
    protected int beamSize;
    protected int maxIterations;
    protected IQualityMeasure qualityMeasure;
    /**
     * if set to true alternatives will be induces using conditions based on single attribute
     */
    protected boolean limitToOneAttribute = true;


    public InnerAlternativesInducer(ExecutorService pool, InductionParameters params, Map<Attribute, Map<Double, IntegerBitSet>> precalculatedCoverings, Map<Double, IntegerBitSet> P, Map<Double, IntegerBitSet> N) {
        super(pool, params, precalculatedCoverings, P, N);

        this.beamSize = params.getInnerAlternativesSearchBeamSize();
        this.maxIterations = params.getInnerAlternativesMaxSearchIterations();
        this.qualityMeasure = params.getInductionMeasure();
        this.minimumCovered = params.getMinimumCovered();
        this.alternatives = new ArrayList<>();
    }

    public InnerAlternativesInducer(InnerAlternativesInducer reference) {
        super(reference.pool, reference.params, reference.precalculatedCoverings, reference.P, reference.N);
    }

    /**
     * Set induced elementary conditions to start growing from. Single
     * elements alternatives will be produces from them and later will be grown
     * to induce best possible alternatives
     *
     * @param startingPoints elementary conditions to start from
     */
    public void setStartingPoints(List<ConditionEvaluation> startingPoints) {
        this.initializeAlternatives(startingPoints);
    }

    private void initializeAlternatives(List<ConditionEvaluation> startingPoints) {
        this.startingPoints = startingPoints.stream().map((ConditionEvaluation s) -> {
            // produce single element alternative from starting elementary conditions
            // each such alternative it equal to this condition
            ConditionEvaluation startingPoint = new ConditionEvaluation();
            startingPoint.coveredExamples = s.coveredExamples;
            startingPoint.covered = s.covered;
            startingPoint.covering = s.covering;
            startingPoint.quality = s.quality;

            CompoundCondition alternative = new CompoundCondition();
            alternative.setLogicalOperator(LogicalOperator.ALTERNATIVE);
            alternative.addSubcondition(s.condition);
            startingPoint.condition = alternative;
            return startingPoint;
        }).collect(Collectors.toList());
        alternatives = new ArrayList<>();
        alternatives.addAll(this.startingPoints);
    }

    protected void initializeBeamSearch(
            Rule rule,
            double apriori_prec,
            IntegerBitSet coveredByRule,
            IntegerBitSet uncoveredPositives
    ) {
        double classId = ((SingletonSet) rule.getConsequence().getValueSet()).getValue();

        this.beamSearch = new BeamSearch<ConditionEvaluation>(this.beamSize) {

            @Override
            protected Comparable evaluateSolution(ConditionEvaluation solution, int indexInBeam) {
                double p = solution.coveredExamples.calculateIntersectionSize(rule.getCoveredPositives());
                int toCover_p = solution.coveredExamples.calculateIntersectionSize(coveredByRule, uncoveredPositives);
                double n = solution.coveredExamples.calculateIntersectionSize(coveredByRule) - p;
                double precision = p / (p + n);

                if (precision < apriori_prec)
                    return null; // ignore solution
                if (!checkCandidate(solution.condition, classId, rule.getWeighted_P(), toCover_p))
                    return null; // ignore solution

                solution.quality = qualityMeasure.calculate(p, n, rule.getWeighted_P(), rule.getWeighted_N());
                solution.covered = p;
                // no need to specify evaluation logic cause solution itself is Comparable
                return solution;
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

    protected CompoundCondition cloneAlternative(CompoundCondition alternative) {
        CompoundCondition clone = new CompoundCondition();
        clone.setLogicalOperator(LogicalOperator.ALTERNATIVE);
        clone.setDisabled(alternative.isDisabled());
        alternative.getSubconditions().forEach(clone::addSubcondition);
        return clone;
    }

    private List<Future<Void>> grow(ConditionEvaluation currentBestQuality) {
        List<Future<Void>> futures = new ArrayList<>();
        Future future = pool.submit(() -> {
            List<SearchResult<ConditionEvaluation>> solutions = beamSearch.searchUntil(
                    this.alternatives,
                    () -> {
                        boolean allSolutionsWorseThanStart = beamSearch.getCurrentSolutions().stream()
                                .allMatch((s) -> s.evaluation != null && s.evaluation.compareTo(currentBestQuality) <= 0);
                        return (allSolutionsWorseThanStart && beamSearch.getIterationsCount() > 1) || this.beamCandidates[0].size() == 0;
                    },
                    this.maxIterations
            );
            this.alternatives = solutions.stream()
                    .map((e) -> e.value)
                    .filter(c -> (((CompoundCondition) c.condition).getSubconditions().size() > 1))
                    .collect(Collectors.toList());
            if (alternatives.size() > 0)
                addCondition(alternatives.get(0));
        });
        futures.add(future);
        return futures;
    }

    @Override
    public void setCandidates(List<ConditionEvaluation> candidates) {
        if (limitToOneAttribute) {
            String startingPointAttribute = this.startingPoints.get(0).condition.getAttributes().toArray()[0].toString();
            candidates = candidates.stream()
                    .filter((e) -> e.condition.getAttributes().size() == 1 && e.condition.getAttributes()
                            .contains(startingPointAttribute)).collect(Collectors.toList());
        }
        // don't use negated conditions in inner alternatives
        candidates = candidates.stream()
                .filter((e) -> !e.condition.isNegated())
                .collect(Collectors.toList());
        super.setCandidates(candidates);
        beamCandidates = new List[this.beamSize];
        for (int i = 0; i < this.beamSize; i++) {
            beamCandidates[i] = new ArrayList(candidates);
            beamCandidates[i].remove(startingPoints.get(0));
        }
    }

    @Override
    public List<Future> preprocess(ExampleSet trainSet, List<Attribute> numericalAttributes, List<Attribute> nominalAttributes) {
        return new ArrayList<>();
    }

    /**
     * Try inducing N best alternatives from starting points given poll of
     * candidates conditions to append.
     *
     * @return N best induces alternatives
     */
    @Override
    protected List<Future<Void>> induceConditions(Rule rule, ExampleSet trainSet, double apriori_prec, Set<Integer> uncoveredPositives, Set<Integer> coveredByRule, Map<String, Map<Double, TotalPosNeg>> totals) {
        Attribute startingPointAttribute = trainSet
                .getAttributes().get((String) this.startingPoints.get(0).condition.getAttributes().toArray()[0]);
        // for nominal attributes inner alternatives for same conditions are equal to discrete set conditions
        if (limitToOneAttribute && startingPointAttribute.isNominal())
            return new ArrayList<>();
        if (candidates.size() == 0)
            return new ArrayList<>();
        if (this.beamSearch == null)
            this.initializeBeamSearch(rule, apriori_prec, (IntegerBitSet) coveredByRule, (IntegerBitSet) uncoveredPositives);
        // select currently best condition's quality
        ConditionEvaluation currentBest = startingPoints.stream().sorted().collect(Collectors.toList()).get(0);
        // try to grow alternatives
        return grow(currentBest);
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
        List<ConditionEvaluation> conditions = this.beamSearch.getGlobalBestSolutions().stream().map((e) -> e.value).collect(Collectors.toList());
        return conditions.size() > 0 ? conditions.get(0) : null;
    }

    @Override
    public void onBestConditionSelected(ConditionEvaluation conditionEvaluation) {

    }

    @Override
    public InnerAlternativesInducer clone() {
        return new InnerAlternativesInducer(this);
    }
}
