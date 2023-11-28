package adaa.analytics.rules.logic.induction.beamsearch;

import adaa.analytics.rules.logic.induction.ConditionEvaluation;
import adaa.analytics.rules.logic.representation.ElementaryCondition;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Class implementing generic Beam search.
 *
 * @param <T> type of solution
 */
public abstract class BeamSearch<T> {

    public int getIterationsCount() {
        return iterationsCount;
    }

    protected int getBeamSize() {
        return beamSize;
    }

    public List<SearchResult<T>> getCurrentSolutions() {
        return currentSolutions;
    }

    private int beamSize;
    private List<SearchResult<T>> currentSolutions;

    private List<SearchResult<T>> globalBestSolutions;
    private int iterationsCount;
    private boolean searchForGlobalBest;

    public BeamSearch(int beamSize) {
        this.beamSize = beamSize;
        this.searchForGlobalBest = false;
    }

    public List<SearchResult<T>> getGlobalBestSolutions() {
        return globalBestSolutions;
    }

    /**
     * Sets a special flag which if set to true will result in searching globally
     * best solutions. Globally best means best N solutions founds in any iteration.
     * Otherwise best N solutions from last iteration will be found.
     *
     * @param searchForGlobalBest flag value
     */
    public void setSearchForGlobalBest(boolean searchForGlobalBest) {
        this.searchForGlobalBest = searchForGlobalBest;
    }

    /**
     * Method evaluating given solution, it is used to select best solutions
     *
     * @param solution solution
     * @return comparable quality of solution
     */
    protected abstract Comparable evaluateSolution(T solution, int indexInBeam);

    /**
     * Method generating new solutions from given solution, used in each iteration.
     *
     * @return new solutions for next iterations
     */
    public abstract List<T> generateNewSolutions(SearchResult<T> result, int indexInBeam);

    private List<SearchResult<T>> mapStartingPointsToSearchResults(List<T> startingPoints) {
        int i = 0;
        List<SearchResult<T>> results = new ArrayList<>();
        for (T startingPoint : startingPoints) {
            results.add(new SearchResult<>(startingPoint, this.evaluateSolution(startingPoint, i)));
            i++;
        }
       return results;
    }

    private List<SearchResult<T>> solutionsFactory() {
        List<SearchResult<T>> solutions = new ArrayList<>();
        int i = 0;
        for (SearchResult<T> parentSolution : getCurrentSolutions()) {
            List<SearchResult<T>> tmpSolutions = this.mapStartingPointsToSearchResults(
                    this.generateNewSolutions(parentSolution, i)
            );
            solutions.forEach((e) -> {
                e.parentSolution = parentSolution;
            });
            solutions.addAll(tmpSolutions);
            i++;
        }
        return solutions;
    }

    private List<SearchResult<T>> findNBestSolutions(List<SearchResult<T>> solutions) {
        List<SearchResult<T>> topN = solutions.stream().sorted(Comparator.reverseOrder()).limit(this.beamSize).collect(Collectors.toList());
        return topN;
    }

    protected void performIteration() {
        List<SearchResult<T>> newSolutions = solutionsFactory();
        currentSolutions = findNBestSolutions(newSolutions);
        if (searchForGlobalBest) {
            globalBestSolutions.addAll(currentSolutions);
            globalBestSolutions = findNBestSolutions(globalBestSolutions);
        }
        iterationsCount++;
    }

    private List<SearchResult<T>> getFoundSolutions() {
        if (searchForGlobalBest)
            return globalBestSolutions;
        else
            return currentSolutions;
    }

    private void initializeSearch() {
        iterationsCount = 0;
        currentSolutions = null;
        if (searchForGlobalBest)
            globalBestSolutions = new ArrayList<>();
        else
            globalBestSolutions = null;
    }

    /**
     * Perform beam search for given number of iteration, returning solutions
     *
     * @param startingPoints solutions to start from
     * @param iterations     number of iterations
     * @return solutions found
     */
    public List<SearchResult<T>> search(List<T> startingPoints, int iterations) {
        initializeSearch();
        currentSolutions = mapStartingPointsToSearchResults(startingPoints);
        for (int i = 0; i < iterations; i++) {
            performIteration();
        }

        return getFoundSolutions();
    }

    /**
     * Perform beam search for given number of iteration, returning solutions
     *
     * @param startingPoints solutions to start from
     * @param stopCriterion  criterion telling when stop
     * @return solutions found
     */
    public List<SearchResult<T>> searchUntil(List<T> startingPoints, BeamSearchStopCriterion stopCriterion, int maxIterations) {
        initializeSearch();
        currentSolutions = mapStartingPointsToSearchResults(startingPoints);
        while (!stopCriterion.isSatisfied() && this.iterationsCount < maxIterations) {
            performIteration();
        }
        return getFoundSolutions();
    }
}