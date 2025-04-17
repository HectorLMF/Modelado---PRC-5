package experiment;

import model.Attribute;
import model.Dataset;
import model.Instance;
import knn.KNNClassifier;
import java.io.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

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
     * Divide el dataset en conjuntos de entrenamiento y prueba según el porcentaje indicado.
     * Si 'random' es true, mezcla los datos usando la semilla indicada.
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

    /**
     * Guarda los conjuntos de entrenamiento y prueba en archivos CSV con nombres basados en fecha/hora.
     */
    public String[] saveSplit() {
        LocalDateTime now = LocalDateTime.now();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm");
        String timestamp = now.format(formatter);

        String pathTrain = "datasets/train_" + timestamp + ".csv";
        String pathTest = "datasets/test_" + timestamp + ".csv";

        saveDataset(trainSet, pathTrain);
        saveDataset(testSet, pathTest);

        System.out.println("Conjuntos guardados en: " + pathTrain + " y " + pathTest);
        return new String[]{pathTrain, pathTest};  // Retornar rutas de archivos guardados
    }

    private void saveDataset(List<Instance> instances, String path) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(path))) {
            List<String> headers = new ArrayList<>();
            for (Attribute attr : dataset.getAttributes()) {
                headers.add(attr.getName());
            }
            writer.write(String.join(",", headers));
            writer.newLine();

            for (Instance instance : instances) {
                List<String> strValues = new ArrayList<>();
                for (Object val : instance.getValues()) {
                    strValues.add(val.toString());
                }
                writer.write(String.join(",", strValues));
                writer.newLine();
            }
        } catch (IOException e) {
            System.out.println("Error al guardar el dataset en " + path + ": " + e.getMessage());
        }
    }

    /**
     * Ejecuta un experimento y genera un informe detallado.
     */
    public void runExperiment(KNNClassifier classifier, String originalDatasetPath) {
        int correct = 0;
        int total = testSet.size();
        List<String> classLabels = dataset.getClassValues();
        int[][] confusion = new int[classLabels.size()][classLabels.size()];

        for (Instance instance : testSet) {
            int classIndex = dataset.getClassAttributeIndex();
            String actual = instance.getValue(classIndex).toString();
            String predicted = classifier.classify(instance);
            if (actual.equals(predicted)) {
                correct++;
            }
            int actualIndex = classLabels.indexOf(actual);
            int predictedIndex = classLabels.indexOf(predicted);
            confusion[actualIndex][predictedIndex]++;
        }

        float accuracy = total > 0 ? (correct / (float) total) : 0;
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

        // Preguntar si se desea guardar los conjuntos en 'datasets/'
        Scanner scanner = new Scanner(System.in);
        System.out.println("¿Desea guardar el informe de experimento, conjunto de entrenamiento y prueba en 'datasets/'? (s/n)");
        String saveOption = scanner.nextLine().trim();
        String[] datasetPaths = null;
        if (saveOption.equalsIgnoreCase("s")) {
            datasetPaths = saveSplit();
        }

        // Generar informe del experimento
        generateExperimentReport(originalDatasetPath, datasetPaths, classifier, accuracy, confusion, classLabels);

        System.out.println("Resultados guardados en /experiments_output/");
        System.out.println("Conjuntos guardados en: /datasets/");
    }

    /**
     * Genera un informe de cada experimento y lo guarda en 'experiments_output/'
     */
    private void generateExperimentReport(String originalDatasetPath, String[] datasetPaths,
                                          KNNClassifier classifier, float accuracy, int[][] confusion,
                                          List<String> classLabels) {
        LocalDateTime now = LocalDateTime.now();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd | HH-mm");
        String timestamp = now.format(formatter);

        File outputFolder = new File("experiments_output");
        if (!outputFolder.exists()) {
            outputFolder.mkdir();  // Crear carpeta si no existe
        }

        String reportPath = "experiments_output/experiment_" + timestamp + ".txt";
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(reportPath))) {
            writer.write("=== Informe del Experimento ===\n");
            writer.write("Fecha y hora: " + timestamp + "\n");
            writer.write("Archivo de entrada: " + originalDatasetPath + "\n");

            if (datasetPaths != null) {
                writer.write("Conjunto de entrenamiento guardado en: " + datasetPaths[0] + "\n");
                writer.write("Conjunto de prueba guardado en: " + datasetPaths[1] + "\n");
            } else {
                writer.write("No se guardaron conjuntos de experimentación.\n");
            }

            writer.write("\n=== Parámetros de configuración ===\n");
            writer.write("Valor de k: " + classifier.getK() + "\n");
            writer.write("Métrica de distancia utilizada: " + classifier.getDistanceMetric()+ "\n");
            writer.write("Estrategia de pesado de casos: " + classifier.getCaseWeightingStrategy() + "\n");
            writer.write("Regla de clasificación utilizada: " + classifier.getClassificationRule()  + "\n");

            writer.write("\n=== Matriz de confusión ===\n");
            writer.write("Precisión del modelo: " + accuracy + "\n");
            writer.write("\t");
            for (String label : classLabels) {
                writer.write(label + "\t");
            }
            writer.write("\n");
            for (int i = 0; i < classLabels.size(); i++) {
                writer.write(classLabels.get(i) + "\t");
                for (int j = 0; j < classLabels.size(); j++) {
                    writer.write(confusion[i][j] + "\t");
                }
                writer.write("\n");
            }

            System.out.println("Informe del experimento guardado en: " + reportPath);
        } catch (IOException e) {
            System.out.println("Error al generar el informe del experimento: " + e.getMessage());
        }
    }
}
