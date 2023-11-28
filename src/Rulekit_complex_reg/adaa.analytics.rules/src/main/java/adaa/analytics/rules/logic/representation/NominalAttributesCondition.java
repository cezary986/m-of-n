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
package adaa.analytics.rules.logic.representation;

import com.rapidminer.example.Attribute;
import com.rapidminer.example.Example;
import com.rapidminer.example.ExampleSet;
import com.rapidminer.example.table.DataRow;

import java.util.*;

/**
 * Represents an elementary condition (built upon single attribute and value set).
 *
 * @author Adam Gudys
 */
public class NominalAttributesCondition extends ConditionBase {

    /**
     * Serialization id.
     */
    private static final long serialVersionUID = -7502367853403510782L;

    /**
     * Set of attributes the condition is built upon.
     */
    protected Set<String> attributes;

    /**
     * Flag indicating if condition (in particular, the value set) is adjustable.
     */
    protected boolean adjustable = false;

    /**
     * Gets {@link #attributes}.
     */
    public Set<String> getAttributes() {
        return attributes;
    }


    /**
     * Gets {@link #adjustable}.
     */
    public boolean isAdjustable() {
        return adjustable;
    }

    /**
     * Sets {@link #adjustable}.
     */
    public void setAdjustable(boolean b) {
        adjustable = b;
    }

    /**
     * Creates empty condition.
     */
    protected NominalAttributesCondition() {
    }

    /**
     * Creates empty condition.
     */
    protected NominalAttributesCondition(boolean negated) {
        this.negated = negated;
    }

    /**
     * Initializes members.
     *
     * @param attributes Attribute.
     */
    public NominalAttributesCondition(Set<String> attributes) {
        this.attributes = attributes;
    }
    /**
     * Initializes members.
     *
     * @param attributes Attribute.
     * @Param negated Logical value indicating whether condition is negated
     */
    public NominalAttributesCondition(Set<String> attributes, boolean negated) {
        this.attributes = attributes;
        this.negated = negated;
        if (attributes.size()>2 && negated) {
            throw new IllegalArgumentException("Negations are only supported for NominalAttributesCondition for 2 attributes");
        }
    }
    /**
     * Evaluates the condition on a given example.
     *
     * @param ex Example to be examined.
     * @return Logical value indicating whether the example fulfills the condition.
     */
    @Override
    protected boolean internalEvaluate(Example ex) {
        boolean satisfied = true;
        int i = 0;
        String v_prev = "";
        String v = "";
        for (String attribute: attributes){
            Attribute attr = ex.getAttributes().get(attribute);
            if(i==0){
                v_prev = attr.getMapping().mapIndex((int)ex.getValue(attr));
            }else{
                v = attr.getMapping().mapIndex((int)ex.getValue(attr));
            }
            if(i > 0 && !v.equals(v_prev)) {
                satisfied = false;
                break;
            }
            i++;
        }
        return isNegated() ? !satisfied : satisfied;
    }

    /**
     * Evaluates the condition on a specified dataset.
     *
     * @param set        Input dataset.
     * @param outIndices Output set of indices covered by the condition.
     */
    @Override
    protected void internalEvaluate(ExampleSet set, Set<Integer> outIndices) {

        List<String> tmpList = new ArrayList(this.attributes);
        int id = 0;
        for (Example e : set) {
            DataRow dr = e.getDataRow();

            boolean wasNan = false;
            for (String attr : tmpList) {
                Attribute attribute = e.getAttributes().get(attr);
                double value = dr.get(attribute);
                if (Double.isNaN(value)) {
                    wasNan = true;
                    break;
                }
            }
            if (!wasNan && this.internalEvaluate(e)) {
                outIndices.add(id);
            }
            ++id;
        }
    }

    /**
     * Generates a text representation of the condition.
     *
     * @return Text representation.
     */
    public String toString() {
        String s = "";
        for (String attribute: attributes){
            s += attribute + " = ";
        }
        s = s.substring(0, s.length() - 3); //delete last =
        if (this.negated) {
            s = s.replace("=", "!=");
        }
        if (type == Type.FORCED) {
            s = "[[" + s + "]]";
        } else if (type == Type.PREFERRED) {
            s = "[" + s + "]";
        }
        return s;
    }

    /**
     * Verifies whether the condition is equal to another one.
     *
     * @param obj Reference object.
     * @return Logical value indicating whether conditions are equal.
     */
    @Override
    public boolean equals(Object obj) {
        if (obj == this) {
            return true;
        } else {
            NominalAttributesCondition ref = (obj instanceof NominalAttributesCondition) ? (NominalAttributesCondition) obj : null;
            if (ref != null) {
                return this.attributes.equals(ref.attributes) && negated == ref.negated;
            } else {
                return false;
            }
        }
    }

    /**
     * Calculates object hash code.
     *
     * @return Hash code.
     */
    @Override
    public int hashCode() {
        int result = attributes.hashCode();
        result = 31 * result;
        return result;
    }



}
