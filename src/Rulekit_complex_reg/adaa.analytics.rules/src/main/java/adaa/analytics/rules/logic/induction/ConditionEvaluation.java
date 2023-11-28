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

import adaa.analytics.rules.logic.representation.ConditionBase;
import adaa.analytics.rules.logic.representation.IntegerBitSet;
import adaa.analytics.rules.logic.representation.NominalAttributesCondition;
import com.rapidminer.example.ExampleSet;
import org.jetbrains.annotations.NotNull;

import java.util.Comparator;

/**
 * Helper class for storing information about evaluated condition.
 *
 * @author Adam Gudys
 */
public class ConditionEvaluation implements Comparable<ConditionEvaluation> {
    public ConditionBase condition = null;
    public Covering covering = null;
    public IntegerBitSet coveredExamples = null;
    public double quality = -Double.MAX_VALUE;
    public double covered = 0;

    public ConditionEvaluation() {
    }

    private double getQuality() {
        return this.quality;
    }

    private double getCovered() {
        return this.covered;
    }

    private int getAttributesCount() {
        return condition.getAttributes().size();
    }

    private String getConditionName() {
        return this.condition.toString();
    }

    public ConditionEvaluation(ExampleSet exampleSet) {
        this.coveredExamples = new IntegerBitSet(exampleSet.size());
    }

    public static Comparator<ConditionEvaluation> getComparator() {
        return ConditionEvaluation::compareTo;
    }

    @Override
    public int compareTo(@NotNull ConditionEvaluation o) {
        if (Double.isNaN(this.quality)) {
            this.quality = Double.NEGATIVE_INFINITY;
        }
        if (Double.isNaN(o.quality)) {
            o.quality = Double.NEGATIVE_INFINITY;
        }
        Comparator<ConditionEvaluation> comparator = Comparator
                .comparing(ConditionEvaluation::getQuality)
                .thenComparing(ConditionEvaluation::getCovered)
                .thenComparing(ConditionEvaluation::getAttributesCount)
                .thenComparing(ConditionEvaluation::getConditionName, Comparator.reverseOrder());
        return comparator.compare(this, o);
    }

    @Override
    public boolean equals(Object obj) {
        ConditionBase condition = obj instanceof ConditionBase ? (ConditionBase) obj : ((ConditionEvaluation) obj).condition;
        return condition.toString().equals(condition.toString());
    }

    /**
     * Calculates object hash code.
     *
     * @return Hash code.
     */
    @Override
    public int hashCode() {
        return condition.hashCode();
    }
}
