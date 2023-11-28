package adaa.analytics.rules.consoles;

import adaa.analytics.rules.logic.representation.*;
import adaa.analytics.rules.operator.RuleGenerator;
import adaa.analytics.rules.utils.RapidMiner5;
import com.rapidminer.RapidMiner;
import com.rapidminer.example.Attributes;
import com.rapidminer.example.ExampleSet;
import com.rapidminer.operator.OperatorCreationException;
import com.rapidminer.operator.OperatorException;
import com.rapidminer.operator.performance.PerformanceVector;
import com.rapidminer.operator.preprocessing.filter.ChangeAttributeRole;
import com.rapidminer.tools.PlatformUtilities;
import com.rapidminer5.operator.io.ArffExampleSource;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.*;

import static adaa.analytics.rules.operator.RuleGenerator.*;

public class ComplexConditionsExperimentalConsole {

    long startTime = 0;
    long endTime = 0;
    public static void main(String[] args) throws OperatorCreationException, ParserConfigurationException, OperatorException, SAXException, IOException {
        ComplexConditionsExperimentalConsole console = new ComplexConditionsExperimentalConsole();
        console.test_rulekit(null);
        System.out.println("Finished");

    }


    private void test_rulekit(String configFile)throws ParserConfigurationException, SAXException, IOException, OperatorCreationException, OperatorException {
        // Read config file
        configFile = "C:\\Users\\cezar\\OneDrive\\Pulpit\\EMAG\\GIT\\Rulekit_complex_reg\\iris_reg.xml";
        DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
        DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
        Document doc = dBuilder.parse(configFile);

        //init RapidMiner to use needed operators
        System.setProperty(PlatformUtilities.PROPERTY_RAPIDMINER_HOME, Paths.get("").toAbsolutePath().toString());
        RapidMiner.setExecutionMode(RapidMiner.ExecutionMode.COMMAND_LINE);
        RapidMiner.init();

        NodeList experimentsNodes = doc.getElementsByTagName("experiment");
        for (int experimentId = 0; experimentId < experimentsNodes.getLength(); experimentId++) {
            Element node = (Element) experimentsNodes.item(experimentId);

            String expName = node.getAttribute("name");
            String datasetName = expName.split("_")[0];

            // get Dataset from config file
            ExampleSet exampleSet = retrieveExampleSet(node);

            RuleGenerator ruleGenerator = RapidMiner5.createOperator(RuleGenerator.class);
            ruleGenerator.setParameter(PARAMETER_INDUCTION_MEASURE, "Precision");
            ruleGenerator.setParameter(PARAMETER_PRUNING_MEASURE, "Precision");
            ruleGenerator.setParameter(PARAMETER_VOTING_MEASURE, "Precision");

            ruleGenerator.setParameter(PARAMETER_DISCRETE_SET_CONDITIONS_ENABLED, "true");
            ruleGenerator.setParameter(PARAMETER_NEGATED_CONDITIONS_ENABLED, "true");
            ruleGenerator.setParameter(PARAMETER_INTERVALS_CONDITIONS_ENABLED, "true");
            ruleGenerator.setParameter(PARAMETER_NUMERICAL_ATTRIBUTES_CONDITIONS_ENABLED, "true");
            ruleGenerator.setParameter(PARAMETER_DISCRETE_ATTRIBUTES_CONDITIONS_ENABLED, "true");

            ruleGenerator.setParameter(PARAMETER_INNER_ALTERNATIVES_MAX_SEARCH_ITERATIONS, "3");
            ruleGenerator.setParameter(PARAMETER_INNER_ALTERNATIVES_SEARCH_BEAM_SIZE, "5");

            RuleSetBase ruleSetBase = (RuleSetBase) ruleGenerator.learn(exampleSet);
            List<Rule> rulesList = ruleSetBase.getRules();

            ExampleSet prediction = ruleSetBase.apply(exampleSet);

            PerformanceVector performance = RuleGenerator.recalculatePerformance(ruleSetBase);
            StringBuilder sb = new StringBuilder();
            for (String name : performance.getCriteriaNames()) {
                double avg = performance.getCriterion(name).getAverage();
                sb.append(name).append(": ").append(avg).append("\n");
            }
            System.out.println(sb.toString());
            System.out.println("Examined rules:");
            double precision;
            for (Rule rule : rulesList) {
                System.out.println(rule.toString() +" " +rule.printStats());
            }
        }
    }



    private ExampleSet retrieveExampleSet(Element doc) throws OperatorCreationException, OperatorException {
        String label = doc.getElementsByTagName("label").item(0).getTextContent();
        String inFile = doc.getElementsByTagName("in_file").item(0).getTextContent();
        File f = new File(inFile);
        String inFilePath = f.isAbsolute() ? inFile : (System.getProperty("user.dir") + "/" + inFile);

        ArffExampleSource arff = RapidMiner5.createOperator(ArffExampleSource.class);
        arff.setParameter(ArffExampleSource.PARAMETER_DATA_FILE, inFilePath);
        ExampleSet exampleSet = arff.createExampleSet();

        ChangeAttributeRole roleSetter = RapidMiner5.createOperator(ChangeAttributeRole.class);
        roleSetter.setParameter(ChangeAttributeRole.PARAMETER_NAME, label);
        roleSetter.setParameter(ChangeAttributeRole.PARAMETER_TARGET_ROLE, Attributes.LABEL_NAME);

        if (doc.getElementsByTagName(SurvivalRule.SURVIVAL_TIME_ROLE).getLength() > 0) {
            String val = doc.getElementsByTagName(SurvivalRule.SURVIVAL_TIME_ROLE).item(0).getTextContent();
            roleSetter.setListParameter(ChangeAttributeRole.PARAMETER_CHANGE_ATTRIBUTES, Collections.singletonList(new String[]{val, SurvivalRule.SURVIVAL_TIME_ROLE}));
        }

        roleSetter.apply(exampleSet);

        return exampleSet;
    }


    private Set<String> retrieveRules(Element doc) {
        Set<String> rulesSet = new TreeSet<>();
        NodeList ruleNodes = doc.getElementsByTagName("rule");

        for (int ruleId = 0; ruleId < ruleNodes.getLength(); ++ruleId) {
            Element ruleNode = (Element) ruleNodes.item(ruleId);
            String ruleContent = ruleNode.getTextContent();
            rulesSet.add(ruleContent);
        }
        return rulesSet;
    }

    private HashMap<String, String> retrieveParams(Element doc) {
        HashMap<String, String> paramsMap = new HashMap<>();

        NodeList paramNodes = doc.getElementsByTagName("param");

        for (int paramId = 0; paramId < paramNodes.getLength(); ++paramId) {
            Element paramNode = (Element) paramNodes.item(paramId);
            String name = paramNode.getAttribute("name");
            String value = paramNode.getTextContent();
            paramsMap.put(name,value);
        }

        return paramsMap;
    }


}
