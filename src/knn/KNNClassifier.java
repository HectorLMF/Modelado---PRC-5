package knn;

import classification.ClassificationRule;
import distance.DistanceMetric;
import model.Dataset;
import model.Instance;
import weighting.CaseWeightingStrategy;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class KNNClassifier {
    private int k;
    private DistanceMetric distanceMetric;
    private ClassificationRule rule;
    private Dataset dataset;
    private List<Float> attributeWeights;  // Se obtiene directamente del dataset
    private CaseWeightingStrategy caseWeightingStrategy;

    public KNNClassifier(int k, DistanceMetric distanceMetric, ClassificationRule rule, Dataset dataset,
                         CaseWeightingStrategy caseWeightingStrategy) {
        this.k = k;
        this.distanceMetric = distanceMetric;
        this.rule = rule;
        this.dataset = dataset;
        this.attributeWeights = dataset.getAttributeWeights();
        this.caseWeightingStrategy = caseWeightingStrategy;
    }

    public String classify(Instance instance) {
        List<Neighbor> neighbors = new ArrayList<>();
        for (Instance trainInst : dataset.getInstances()) {
            float d = distanceMetric.compute(trainInst, instance, dataset.getAttributes(), attributeWeights);
            neighbors.add(new Neighbor(trainInst, d));
        }
        Collections.sort(neighbors, Comparator.comparingDouble(n -> n.distance));

        Map<String, Float> votes = new HashMap<>();
        int limit = Math.min(k, neighbors.size());
        for (int i = 0; i < limit; i++) {
            float weightVote = caseWeightingStrategy.getWeight(i + 1, neighbors.get(i).distance);
            String classLabel = neighbors.get(i).instance.getValue(dataset.getClassAttributeIndex()).toString();
            votes.put(classLabel, votes.getOrDefault(classLabel, 0f) + weightVote);
        }
        return rule.predictClass(votes);
    }

    public void setK(int k) {
        this.k = k;
    }

    // Clase interna para almacenar un vecino y su distancia.
    private class Neighbor {
        Instance instance;
        float distance;

        Neighbor(Instance instance, float distance) {
            this.instance = instance;
            this.distance = distance;
        }
    }
}
