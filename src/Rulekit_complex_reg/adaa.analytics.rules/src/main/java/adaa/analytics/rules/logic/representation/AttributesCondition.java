package adaa.analytics.rules.logic.representation;

import com.rapidminer.example.Attribute;
import com.rapidminer.example.Example;
import com.rapidminer.example.ExampleSet;
import org.apache.commons.lang.builder.HashCodeBuilder;

import java.util.HashSet;
import java.util.Set;


public class AttributesCondition extends ConditionBase {

    public enum Operators {

        EQUAL(0),
        GREATER(1),
        LOWER(2);

        int code;

        public boolean evaluate(double left, double right) {
            if (this == Operators.EQUAL) {
                return left == right;
            }
            if (this == Operators.LOWER) {
                return left < right;
            } else {
                return left > right;
            }
        }

        public String toString() {
            if (this == Operators.EQUAL) {
                return "=";
            }
            if (this == Operators.LOWER) {
                return "<";
            } else {
                return ">";
            }
        }

        Operators(int code) {
            this.code = code;
        }
    }

    protected Attribute left;
    protected Attribute right;
    protected Operators operator;

    private Set<String> attributes;

    public AttributesCondition(Attribute left, Attribute right, Operators operator) {
        this.left = left;
        this.right = right;
        this.operator = operator;
        this.attributes = new HashSet<>();
        this.attributes.add(left.getName());
        this.attributes.add(right.getName());
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == this) {
            return true;
        } else {
            AttributesCondition ref = (obj instanceof AttributesCondition) ? (AttributesCondition) obj : null;
            if (ref != null) {
                return (left.equals(ref.left) && right.equals(ref.right) && operator.code == ref.operator.code) ||
                        (left.equals(ref.right) && right.equals(ref.left) && (
                                (operator.code == Operators.LOWER.code && ref.operator.code == Operators.GREATER.code) ||
                                        (operator.code == Operators.GREATER.code && ref.operator.code == Operators.LOWER.code)
                        ));
            } else {
                return false;
            }
        }
    }

    @Override
    public Set<String> getAttributes() {
        return this.attributes;
    }

    @Override
    protected boolean internalEvaluate(Example ex) {
        double leftValue = ex.getValue(left);
        double rightValue = ex.getValue(right);
        return this.operator.evaluate(leftValue, rightValue);
    }

    @Override
    protected void internalEvaluate(ExampleSet set, Set<Integer> outIndices) {
		/* The following code does not work for SplittedExampleSet
		ExampleTable tab = set.getExampleTable();
		DataRowReader drr = tab.getDataRowReader();

		int id = 0;
		while (drr.hasNext()) {
			DataRow dr = drr.next();

			double v = dr.get(a);
			if (valueSet.contains(v)) {
				outIndices.add(id);
			}
			++id;
		}*/
        for (int id = 0; id < set.size(); ++id) {
            Example ex = set.getExample(id);
            double leftValue = ex.getValue(left);
            double rightValue = ex.getValue(right);
            if (operator.evaluate(leftValue, rightValue)) {
                outIndices.add(id);
            }
        }
    }

    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder
                .append(left.getName())
                .append(" ")
                .append(operator.toString())
                .append(" ")
                .append(right.getName());
        return builder.toString();
    }

    @Override
    public int hashCode() {
        return new HashCodeBuilder().append(left.getName()).append(operator.code).append(right.getName()).toHashCode();
    }

    public Attribute getLeft() {
        return left;
    }

    public Attribute getRight() {
        return right;
    }

    public int getOperator() {
        return operator.code;
    }
}
