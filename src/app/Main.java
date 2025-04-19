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

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Main {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        Dataset dataset = null;
        KNNClassifier classifier = null;
        ExperimentManager expManager = null;
        String datasetPath = "";

        // Variables para la configuración del clasificador
        int k = 5;
        DistanceMetric distanceMetric = null;
        CaseWeightingStrategy weightingStrategy = null;
        ClassificationRule classificationRule = null;
        NormalizationMode normalizationMode = NormalizationMode.MIN_MAX; // valor por defecto

        boolean exit = false;

        while (!exit) {
            System.out.println("\n=== MENÚ PRINCIPAL ===");
            System.out.println("1. Cargar dataset");
            System.out.println("2. Mostrar información del dataset");
            System.out.println("3. Configurar clasificador k-NN");
            System.out.println("4. Clasificar nueva instancia");
            System.out.println("5. Ejecutar experimento");
            System.out.println("6. Salir");
            System.out.print("Elige una opción: ");

            int option = scanner.nextInt();
            scanner.nextLine(); // Consumir salto de línea

            switch (option) {
                case 1:
                    // Cargar dataset con validación de archivo
                    System.out.print("Introduce el path del archivo CSV: ");
                    datasetPath = scanner.nextLine().trim();
                    File file = new File(datasetPath);
                    while (!file.exists() || !file.isFile()) {
                        System.out.println("Error: El archivo no existe. Ingresa otra ruta:");
                        datasetPath = scanner.nextLine().trim();
                        file = new File(datasetPath);
                    }

                    // Selección del modo de normalización
                    System.out.println("Seleccione el modo de normalización: 0 = RAW, 1 = MIN_MAX, 2 = Z_SCORE");
                    int normOpt = scanner.nextInt();
                    scanner.nextLine(); // Consumir salto de línea
                    if (normOpt == 0) {
                        normalizationMode = NormalizationMode.RAW;
                    } else if (normOpt == 2) {
                        normalizationMode = NormalizationMode.Z_SCORE;
                    } else {
                        normalizationMode = NormalizationMode.MIN_MAX;
                    }

                    dataset = CSVLoader.loadDataset(datasetPath, normalizationMode);
                    if (dataset != null) {
                        System.out.println("Dataset cargado exitosamente.");
                        expManager = new ExperimentManager(dataset);
                    } else {
                        System.out.println("Error al cargar el dataset.");
                    }
                    break;

                case 2:
                    // Mostrar información del dataset
                    if (dataset == null) {
                        System.out.println("Primero carga un dataset.");
                    } else {
                        System.out.println("Información del dataset:");
                        System.out.println("Instancias: " + dataset.getNumInstances());
                        System.out.println("Atributos: " + dataset.getNumAttributes());
                        System.out.println("Listado de atributos:");
                        for (Attribute attr : dataset.getAttributes()) {
                            System.out.println("  " + attr);
                        }
                        System.out.println("Estadísticas de atributos:");
                        List<AttributeStats> stats = dataset.getAttributeStats();
                        for (AttributeStats stat : stats) {
                            System.out.println(stat);
                        }
                    }
                    break;

                case 3:
                    // Configurar el clasificador k-NN
                    if (dataset == null) {
                        System.out.println("Primero carga un dataset.");
                        break;
                    }

                    System.out.print("Introduce el valor de k: ");
                    k = scanner.nextInt();
                    scanner.nextLine();

                    // Selección de la métrica de distancia
                    System.out.println("Seleccione la métrica de distancia:");
                    System.out.println("1 = Euclidean");
                    System.out.println("2 = Manhattan");
                    System.out.println("3 = Chebyshev");
                    int distOpt = scanner.nextInt();
                    scanner.nextLine();
                    if (distOpt == 2) {
                        distanceMetric = new ManhattanDistance();
                    } else if (distOpt == 3) {
                        distanceMetric = new ChebyshevDistance();
                    } else {
                        distanceMetric = new EuclideanDistance();
                    }

                    // Selección de estrategia de pesado de vecinos
                    System.out.println("Seleccione la estrategia de pesado de vecinos:");
                    System.out.println("1 = Igual (EqualWeighting)");
                    System.out.println("2 = Inverso a la distancia (DistanceInverseWeighting)");
                    System.out.println("3 = Basado en ranking (RankBasedWeighting)");
                    int weightOpt = scanner.nextInt();
                    scanner.nextLine();
                    if (weightOpt == 2) {
                        weightingStrategy = new DistanceInverseWeighting();
                    } else if (weightOpt == 3) {
                        List<Float> rankWeights = new ArrayList<>();
                        rankWeights.add(1.0f);
                        rankWeights.add(0.8f);
                        rankWeights.add(0.6f);
                        weightingStrategy = new RankBasedWeighting(rankWeights);
                    } else {
                        weightingStrategy = new EqualWeighting();
                    }

                    // Selección del algoritmo de votación
                    System.out.println("Seleccione el algoritmo de votación:");
                    System.out.println("1 = MajorityVote");
                    System.out.println("2 = ThresholdVote");
                    int voteOpt = scanner.nextInt();
                    scanner.nextLine();
                    if (voteOpt == 2) {
                        classificationRule = new ThresholdVote(0.5f); // Umbral por defecto 0.5
                    } else {
                        classificationRule = new MajorityVote();
                    }

                    classifier = new KNNClassifier(k, distanceMetric, classificationRule, dataset, weightingStrategy);
                    System.out.println("Clasificador configurado exitosamente.");
                    break;

                case 4:
                    // Clasificar nueva instancia
                    if (classifier == null) {
                        System.out.println("Primero configura el clasificador.");
                        break;
                    }
                    System.out.println("Introduce los valores de la instancia (separados por comas, sin la etiqueta de clase):");
                    String instanceLine = scanner.nextLine();
                    String[] tokens = instanceLine.split(",");
                    Instance newInst = new Instance();
                    List<Attribute> attrs = dataset.getAttributes();
                    for (int i = 0; i < attrs.size(); i++) {
                        if (attrs.get(i).isClass()) {
                            newInst.addValue(""); // Deja en blanco la clase (será predicha)
                        } else {
                            String token = (i < tokens.length) ? tokens[i].trim() : "";
                            if (attrs.get(i).isQuantitative()) {
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
                    break;

                case 5:
                    // Ejecutar experimento
                    if (classifier == null || expManager == null) {
                        System.out.println("Primero configura el clasificador y carga el dataset.");
                        break;
                    }
                    System.out.println("Introduce el porcentaje para el conjunto de pruebas (ej.: 0.3 para 30%):");
                    float testRatio;
                    while (!scanner.hasNextFloat()) {
                        System.out.println("Error: Ingresa un número entre 0 y 1 (ejemplo: 0.3).");
                        scanner.next();
                    }
                    testRatio = scanner.nextFloat();
                    scanner.nextLine();

                    System.out.println("¿Generar splits aleatorios? (s/n):");
                    boolean randomSplit = scanner.nextLine().trim().equalsIgnoreCase("s");

                    System.out.println("Introduce una semilla entera para la aleatorización:");
                    int seedExp;
                    while (!scanner.hasNextInt()) {
                        System.out.println("Error: Ingresa una semilla entera válida.");
                        scanner.next();
                    }
                    seedExp = scanner.nextInt();
                    scanner.nextLine();

                    expManager.splitDatasetRatio(testRatio, randomSplit, seedExp);
                    expManager.runExperiment(classifier, datasetPath);
                    break;

                case 6:
                    System.out.println("Saliendo...");
                    exit = true;
                    break;

                default:
                    System.out.println("Opción no válida. Intenta nuevamente.");
            } // Fin del switch
        } // Fin del while

        scanner.close();
    }
}
