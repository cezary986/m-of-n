package adaa.analytics.rules.logic.induction.conditions.regression;
import adaa.analytics.rules.logic.induction.ConditionEvaluation;
import adaa.analytics.rules.logic.induction.Covering;
import adaa.analytics.rules.logic.induction.InductionParameters;
import adaa.analytics.rules.logic.induction.SetHelper;
import adaa.analytics.rules.logic.representation.*;
import com.rapidminer.example.ExampleSet;
import java.util.Set;
import java.util.logging.Level;

public abstract class  RegressionConditionEvaluator {

    public static ConditionEvaluation checkCandidate(
            ConditionEvaluation currentBest,
            ConditionBase candidate,
            InductionParameters params,
            double p,
            double n,
            double new_p,
            double new_n,
            double P,
            double N
    ) {
        if (RegressionConditionEvaluator.checkCoverage(params, p, n, new_p, new_n, P, N)) {
            double quality = params.getInductionMeasure().calculate(p, n, P, N);

            ConditionEvaluation evaluation = new ConditionEvaluation();
            evaluation.quality = quality;
            evaluation.covered =  new_p + new_n;
            evaluation.condition = candidate;

            Logger.log(", q=" + quality, Level.FINEST);

            if (currentBest != null && evaluation.compareTo(currentBest) > 0) {
                currentBest.quality = quality;
                currentBest.condition = candidate;
                currentBest.covered = new_p + new_n;
                currentBest.covering = new Covering(p, n, P, N);
                Logger.log(", approved!\n", Level.FINEST);
                //rule.setWeight(quality);
            }
            return evaluation;
        }
        return null;
    }


    public static ConditionEvaluation checkCandidate(
            ExampleSet dataset,
            Rule rule,
            ConditionBase candidate,
            Set<Integer> uncovered,
            ConditionEvaluation currentBest,
            InductionParameters params
    ) {

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

            if (cov.weighted_p == cov.weighted_P) {
                return  null;
            }

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

            if (new_p == 0) {
                return  null;
            }

            if (RegressionConditionEvaluator.checkCoverage(params, cov.weighted_p, cov.weighted_n, new_p, new_n, rule.getWeighted_P(), rule.getWeighted_N())) {
                double quality = params.getInductionMeasure().calculate(dataset, cov);

                ConditionEvaluation evaluation = new ConditionEvaluation();
                evaluation.quality = quality;
                evaluation.covered =  new_p + new_n;
                evaluation.condition = candidate;

                Logger.log(", q=" + quality, Level.FINEST);

                if (currentBest != null && evaluation.compareTo(currentBest) > 0) {
                    currentBest.quality = quality;
                    currentBest.condition = candidate;
                    currentBest.covered = new_p + new_n;
                    currentBest.covering = cov;
                    Logger.log(", approved!\n", Level.FINEST);
                    //rule.setWeight(quality);
                }
                return evaluation;
            }

            Logger.log("\n", Level.FINEST);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    private static boolean checkCoverage(InductionParameters params, double p, double n, double new_p, double new_n, double P, double N) {
        return ((new_p + new_n) >= params.getAbsoluteMinimumCovered(P + N)) &&
                ((p + n) >= params.getAbsoluteMinimumCoveredAll(P + N));
    }
}
