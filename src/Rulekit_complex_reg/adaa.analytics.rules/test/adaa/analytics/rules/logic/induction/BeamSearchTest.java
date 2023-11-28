package adaa.analytics.rules.logic.induction;

import adaa.analytics.rules.logic.induction.beamsearch.BeamSearch;
import adaa.analytics.rules.logic.induction.beamsearch.SearchResult;
import com.rapidminer.tools.container.Pair;
import junit.framework.TestCase;
import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.apache.commons.math3.distribution.ChiSquaredDistribution;
public class BeamSearchTest extends TestCase {

    @Test
    public void testSearchUntil() {
        int BEAM_SIZE = 3;
        double STOP_VALUE = 10000.0;

        BeamSearch<Integer> beamSearch = new BeamSearch<Integer>(BEAM_SIZE) {

            @Override
            protected Double evaluateSolution(Integer solution, int indexInBeam) {
                return (double) solution;
            }

            @Override
            public List<Integer> generateNewSolutions(SearchResult<Integer> solution, int indexInBeam) {
                List<Integer> newSolutions = new ArrayList<>();
                Random rn = new Random();
                for (int i = 0; i < this.getBeamSize(); i++)
                    newSolutions.add(solution.value + rn.nextInt(30));
                return newSolutions;
            }
        };
        beamSearch.setSearchForGlobalBest(true);

        List<Integer> startingPoints = new ArrayList<>();
        startingPoints.add(1);
        startingPoints.add(2);
        startingPoints.add(-300);


        List<SearchResult<Integer>> solutions = beamSearch.searchUntil(startingPoints, () -> {
            // run until one of the solution is bigger than STOP_VALUE
            return beamSearch.getCurrentSolutions().stream().anyMatch((s) -> s.evaluation.compareTo(STOP_VALUE) > 0);
        }, 100000);

        Assert.assertTrue("Best solution should match criterion", solutions.get(0).value > STOP_VALUE);
        Assert.assertTrue("Fist solution in the list should be the best one", solutions.get(0).value > solutions.get(solutions.size() - 1).value);

        solutions = beamSearch.searchUntil(startingPoints, () -> {
            // run until one of the solution is bigger than STOP_VALUE
            return beamSearch.getCurrentSolutions().stream().anyMatch((s) -> s.evaluation.compareTo(STOP_VALUE) > 0);
        }, 1);

        Assert.assertTrue("Solution should not much criterion", solutions.get(0).value > STOP_VALUE);
    }


    private double foo(
            double expectedDev, double sampleDev, int samplesize
    ) {
        double factor = sampleDev / expectedDev;
        double T = ((double)(samplesize - 1)) * (factor * factor);

        ChiSquaredDistribution chi = new ChiSquaredDistribution(samplesize - 1);

        return chi.cumulativeProbability(T);
    }

    @Test
    public void testChi() {
        double a = foo(2.0, 12.4, 4);
        a = foo(3.1, 2.4, 10);
        a = foo(6.8, 22.0, 3);

        int i = 0;
    }

}