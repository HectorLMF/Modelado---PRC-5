package experiment;

import knn.KNNClassifier;
import model.Attribute;
import model.Dataset;
import model.Instance;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class ExperimentManager {
    private Dataset dataset;
    private List<Instance> trainSet;
    private List<Instance> testSet;

    public ExperimentManager(Dataset dataset) {
        this.dataset = dataset;
        trainSet = new ArrayList<>();
        testSet = new ArrayList<>();
    }

    /**
     * Separa el dataset en entrenamiento y prueba basado en el porcentaje testRatio.
     * Si 'random' es true, mezcla el dataset utilizando la semilla dada.
     */
    public void splitDatasetRatio(float testRatio, boolean random, int seed) {
        List<Instance> allInstances = new ArrayList<>(dataset.getInstances());
        int total = allInstances.size();
        int testSize = Math.round(total * testRatio);
        if (random) {
            Random rnd = new Random(seed);
            Collections.shuffle(allInstances, rnd);
        }
        trainSet = new ArrayList<>(allInstances.subList(0, total - testSize));
        testSet = new ArrayList<>(allInstances.subList(total - testSize, total));
    }

    public void saveSplit(String trainPath, String testPath) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(trainPath))) {
            List<String> headers = new ArrayList<>();
            for (Attribute attr : dataset.getAttributes()) {
                headers.add(attr.getName());
            }
            writer.write(String.join(",", headers));
            writer.newLine();
            for (Instance instance : trainSet) {
                List<String> values = new ArrayList<>();
                for (Object val : instance.getValues()) {
                    values.add(val.toString());
                }
                writer.write(String.join(",", values));
                writer.newLine();
            }
        } catch (IOException e) {
            System.out.println("Error guardando el conjunto de entrenamiento: " + e.getMessage());
        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(testPath))) {
            List<String> headers = new ArrayList<>();
            for (Attribute attr : dataset.getAttributes()) {
                headers.add(attr.getName());
            }
            writer.write(String.join(",", headers));
            writer.newLine();
            for (Instance instance : testSet) {
                List<String> values = new ArrayList<>();
                for (Object val : instance.getValues()) {
                    values.add(val.toString());
                }
                writer.write(String.join(",", values));
                writer.newLine();
            }
        } catch (IOException e) {
            System.out.println("Error guardando el conjunto de prueba: " + e.getMessage());
        }
    }

    public void runExperiment(KNNClassifier classifier) {
        int correct = 0;
        int total = testSet.size();
        List<String> classLabels = dataset.getClassValues();
        int[][] confusion = new int[classLabels.size()][classLabels.size()];
        for (Instance instance : testSet) {
            String actual = instance.getValue(dataset.getClassAttributeIndex()).toString();
            String predicted = classifier.classify(instance);
            if (actual.equals(predicted)) {
                correct++;
            }
            int actIndex = classLabels.indexOf(actual);
            int predIndex = classLabels.indexOf(predicted);
            confusion[actIndex][predIndex]++;
        }
        float accuracy = (total > 0) ? (correct / (float) total) : 0;
        System.out.println("Precisión: " + accuracy);
        System.out.println("Matriz de Confusión:");
        System.out.print("\t");
        for (String label : classLabels) {
            System.out.print(label + "\t");
        }
        System.out.println();
        for (int i = 0; i < classLabels.size(); i++) {
            System.out.print(classLabels.get(i) + "\t");
            for (int j = 0; j < classLabels.size(); j++) {
                System.out.print(confusion[i][j] + "\t");
            }
            System.out.println();
        }
    }
}
