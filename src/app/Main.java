package app;

import classification.ClassificationRule;
import classification.MajorityVote;
import classification.ThresholdVote;
import distance.ChebyshevDistance;
import distance.DistanceMetric;
import distance.EuclideanDistance;
import distance.ManhattanDistance;
import experiment.ExperimentManager;
import knn.KNNClassifier;
import model.*;
import weighting.CaseWeightingStrategy;
import weighting.DistanceInverseWeighting;
import weighting.EqualWeighting;
import weighting.RankBasedWeighting;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        // Selección del archivo CSV
        System.out.println("Introduce el path del archivo CSV (ej.: iris.csv o glass.csv):");
        String filepath = scanner.nextLine().trim();

        // Selección del modo de normalización
        System.out.println("Seleccione el modo de normalización: 0 = RAW, 1 = MIN_MAX, 2 = Z_SCORE");
        int normOption = scanner.nextInt();
        NormalizationMode normMode = (normOption == 0) ? NormalizationMode.RAW :
                (normOption == 2) ? NormalizationMode.Z_SCORE :
                        NormalizationMode.MIN_MAX;

        Dataset dataset = CSVLoader.loadDataset(filepath, normMode);
        if (dataset == null) {
            System.out.println("Error al cargar el dataset.");
            scanner.close();
            return;
        }

        System.out.println("Número de instancias: " + dataset.getNumInstances());
        System.out.println("Número de atributos: " + dataset.getNumAttributes());
        System.out.println("Atributos:");
        for (Attribute attr : dataset.getAttributes()) {
            System.out.println("  " + attr);
        }

        // Mostrar estadísticas de atributos
        System.out.println("Estadísticas de Atributos:");
        List<AttributeStats> stats = dataset.getAttributeStats();
        for (Object stat : stats) {
            System.out.println(stat);
        }

        // Configuración del valor de k
        System.out.println("Introduce el valor de k para k-NN:");
        int k = scanner.nextInt();

        // Selección de la métrica de distancia
        System.out.println("Seleccione la métrica de distancia:");
        System.out.println("1 = Euclidean");
        System.out.println("2 = Manhattan");
        System.out.println("3 = Chebyshev");
        int distOption = scanner.nextInt();
        DistanceMetric metric = (distOption == 2) ? new ManhattanDistance() :
                (distOption == 3) ? new ChebyshevDistance() :
                        new EuclideanDistance();

        // Selección de estrategia de pesado de vecinos
        System.out.println("Seleccione la estrategia de pesado de casos:");
        System.out.println("1 = Igual (EqualWeighting)");
        System.out.println("2 = Inverso a la distancia (DistanceInverseWeighting)");
        System.out.println("3 = Basado en ranking (RankBasedWeighting)");
        int weightOption = scanner.nextInt();
        CaseWeightingStrategy caseWeighting = (weightOption == 2) ? new DistanceInverseWeighting() :
                (weightOption == 3) ? new RankBasedWeighting(List.of(1.0f, 0.8f, 0.6f, 0.4f, 0.2f)) :
                        new EqualWeighting();

        // Selección del algoritmo de votación
        System.out.println("Seleccione el algoritmo de votación:");
        System.out.println("1 = Mayoría simple (MajorityVote)");
        System.out.println("2 = Umbral mínimo (ThresholdVote)");
        int voteOption = scanner.nextInt();
        ClassificationRule rule = (voteOption == 2) ? new ThresholdVote(0.5f) : new MajorityVote();

        // Crear el clasificador k-NN con las opciones seleccionadas
        KNNClassifier classifier = new KNNClassifier(k, metric, rule, dataset, caseWeighting);

        scanner.nextLine();  // Consumir salto de línea pendiente.

        // Ingresar nueva instancia a clasificar
        System.out.println("Introduce una nueva instancia para clasificar (valores separados por comas, sin el atributo clase):");
        System.out.println("Ejemplo para iris.csv: 5.1,3.5,1.4,0.2");
        String instLine = scanner.nextLine();
        String[] tokens = instLine.split(",");
        Instance newInst = new Instance();
        List<Attribute> attributes = dataset.getAttributes();
        for (int i = 0; i < attributes.size(); i++) {
            if (attributes.get(i).isClass()) {
                newInst.addValue(""); // Se predice la clase
            } else {
                String token = (i < tokens.length) ? tokens[i].trim() : "";
                if (attributes.get(i).isQuantitative()) {
                    try {
                        float value = Float.parseFloat(token);
                        newInst.addValue(value);
                    } catch (NumberFormatException e) {
                        newInst.addValue(0.0f);
                    }
                } else {
                    newInst.addValue(token);
                }
            }
        }

        String predicted = classifier.classify(newInst);
        System.out.println("Clase predicha: " + predicted);

        // Preguntar si se desea ejecutar el módulo de experimentación
        System.out.println("¿Desea ejecutar el módulo de experimentación? (s/n)");
        String expOption = scanner.nextLine().trim();
        if (expOption.equalsIgnoreCase("s")) {
            System.out.println("Introduce el porcentaje para el conjunto de pruebas (ej.: 0.3 para 30%):");
            float testRatio = scanner.nextFloat();
            scanner.nextLine(); // Consumir salto de línea
            System.out.println("¿Generar splits aleatorios? (s/n):");
            String randomOption = scanner.nextLine().trim();
            boolean randomSplit = randomOption.equalsIgnoreCase("s");
            System.out.println("Introduce una semilla entera para la aleatorización:");
            int seed = scanner.nextInt();

            ExperimentManager expManager = new ExperimentManager(dataset);
            expManager.splitDatasetRatio(testRatio, randomSplit, seed);
            expManager.runExperiment(classifier);
        }

        scanner.close();
    }
}
