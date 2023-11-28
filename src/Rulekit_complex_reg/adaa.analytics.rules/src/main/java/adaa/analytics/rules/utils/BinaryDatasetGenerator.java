package adaa.analytics.rules.utils;

import adaa.analytics.rules.logic.representation.*;
import com.rapidminer.example.Attribute;
import com.rapidminer.example.Example;
import com.rapidminer.example.ExampleSet;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class BinaryDatasetGenerator {

    protected static void extractNestedConditions(Map<String, ConditionBase> conditionsMap, CompoundCondition condition) {
        for (ConditionBase subcondition : condition.getSubconditions()) {
            conditionsMap.put(subcondition.toString(), subcondition);
            if (subcondition instanceof CompoundCondition) {
                extractNestedConditions(conditionsMap, (CompoundCondition) subcondition);
            }
        }
    }

    protected static Map<String, ConditionBase> extractRuleSetConditions(RuleSetBase ruleSet) {
        Map<String, ConditionBase> conditionsMap = new HashMap<>();
        for (Rule rule : ruleSet.getRules()) {
//            conditionsMap.put(rule.getPremise().toString(), rule.getPremise());
            extractNestedConditions(conditionsMap, rule.getPremise());
        }
        return conditionsMap;
    }

    public static void writeAttributesMappingToFile(Map<String, ConditionBase> conditionsMap, String outputFilePath) {
        List<String> conditionsStrings = new ArrayList<>(conditionsMap.keySet());
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath, false))){
            for (String conditionKey : conditionsStrings) {
                ConditionBase condition = conditionsMap.get(conditionKey);
                StringBuilder line = new StringBuilder();
                if (!(condition instanceof AttributesCondition)) {
                    for (String attr : condition.getAttributes()) {
                        line.append(attr).append(",");
                    }
                }
                writer.write(line.toString());
                writer.newLine();
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static void fixNominalAttributesMappings(RuleSetBase ruleSet, ExampleSet exampleSet) {
        for (Rule r : ruleSet.getRules()) {
            List<ConditionBase> toCheck = new ArrayList<ConditionBase>(); // list of elementary conditions to check
            toCheck.addAll(r.getPremise().getSubconditions());
            toCheck.add(r.getConsequence());

            for (ConditionBase c: toCheck) {
                ElementaryCondition ec = (c instanceof ElementaryCondition) ? (ElementaryCondition)c : null;
                if (ec != null) {
                    Attribute a = exampleSet.getAttributes().get(ec.getAttribute());
                    if (a.isNominal()) {
                        if (ec.getValueSet() instanceof SingletonSet) {
                            SingletonSet ss = (SingletonSet) ec.getValueSet();
                            String valName = ss.getMapping().get((int)ss.getValue());
                            int newValue = a.getMapping().getIndex(valName);
                            ss.setValue(newValue);
                            ss.setMapping(a.getMapping().getValues());
                        }
                    }
                }
            }
        }
    }

    public static void writeBinaryDatasetToCsvFile(RuleSetBase ruleSet, ExampleSet exampleSet, String outputFilePath) {
//        BinaryDatasetGenerator.fixNominalAttributesMappings(ruleSet, exampleSet);
        Map<String, ConditionBase> conditionsMap = extractRuleSetConditions(ruleSet);
        List<String> lines = new ArrayList<>(exampleSet.size() + 1);
        List<String> conditionsStrings = new ArrayList<>(conditionsMap.keySet());
        lines.add(String.join(";", conditionsStrings));

        List<String> lineBuffer = new ArrayList<>();

        for (int i = 0; i < exampleSet.size(); ++i) {
            Example example = exampleSet.getExample(i);
            for (String conditionString : conditionsStrings) {
                ConditionBase condition = conditionsMap.get(conditionString);
                if (condition instanceof ElementaryCondition && ((ElementaryCondition) condition).getValueSet() instanceof SingletonSet) {
                    SingletonSet valueSet = (SingletonSet) ((ElementaryCondition) condition).getValueSet();
                    String conditionValue = valueSet.getMapping().get((int) valueSet.getValue());
                    Attribute attr = example.getAttributes().get(((ElementaryCondition) condition).getAttribute());
                    String attrValue = attr.getMapping().mapIndex((int) example.getValue(attr));
                    lineBuffer.add(attrValue.equals(conditionValue) ? "1" : "0");
                } else
                    lineBuffer.add(condition.evaluate(example) ? "1" : "0");
            }
            lines.add(String.join(";", lineBuffer));
            lineBuffer.clear();
        }
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath, false))){
            for (String line : lines) {
                writer.write(line);
                writer.newLine();
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        writeAttributesMappingToFile(conditionsMap, outputFilePath.replace(".csv", ".mappings.txt"));
    }
}
