package model;

import stats.QualitativeStats;
import stats.QuantitativeStats;

import java.util.ArrayList;
import java.util.List;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Dataset {
    private List<Instance> data;
    private List<Attribute> attributes;
    private int classAttributeIndex;
    private NormalizationMode normalizationMode;
    private List<Float> attributeWeights; // Pesos para cada atributo

    public Dataset(List<Attribute> attributes) {
        data = new ArrayList<>();
        this.attributes = attributes;
        normalizationMode = NormalizationMode.MIN_MAX;
        classAttributeIndex = attributes.size() - 1;
        attributeWeights = new ArrayList<>();
        for (int i = 0; i < attributes.size(); i++) {
            attributeWeights.add(1.0f);  // Peso por defecto 1.0 para cada atributo.
        }
    }

    public void addInstance(Instance instance) {
        data.add(instance);
    }

    public int getNumAttributes() {
        return attributes.size();
    }

    public int getNumInstances() {
        return data.size();
    }

    public List<Attribute> getAttributes() {
        return attributes;
    }

    public Instance getInstance(int i) {
        return data.get(i);
    }

    public List<Instance> getInstances() {
        return data;
    }

    public void setClassAttributeIndex(int index) {
        this.classAttributeIndex = index;
    }

    public int getClassAttributeIndex() {
        return classAttributeIndex;
    }

    public List<Float> getAttributeWeights() {
        return attributeWeights;
    }

    public void setAttributeWeights(List<Float> weights) {
        if (weights.size() == attributes.size()) {
            this.attributeWeights = weights;
        } else {
            System.out.println("El número de pesos no coincide con el número de atributos.");
        }
    }

    // Retorna la lista de clases encontradas en el dataset.
    public List<String> getClassValues() {
        List<String> classes = new ArrayList<>();
        for (Instance instance : data) {
            String val = instance.getValue(classAttributeIndex).toString();
            if (!classes.contains(val)) {
                classes.add(val);
            }
        }
        return classes;
    }

    /**
     * Normaliza los atributos cuantitativos predictivos según el modo seleccionado.
     * El atributo de clase NO se normaliza.
     */
    public void normalize(NormalizationMode mode) {
        normalizationMode = mode;
        for (int i = 0; i < attributes.size(); i++) {
            if (attributes.get(i).isClass()) continue;
            if (attributes.get(i).isQuantitative()) {
                float min = Float.MAX_VALUE, max = -Float.MAX_VALUE;
                float sum = 0, sumSquared = 0;
                int count = 0;

                // Calcular valores mínimos, máximos y media.
                for (Instance instance : data) {
                    float value = ((Number) instance.getValue(i)).floatValue();
                    if (value < min) min = value;
                    if (value > max) max = value;
                    sum += value;
                    count++;
                }
                float mean = sum / count;
                for (Instance instance : data) {
                    float value = ((Number) instance.getValue(i)).floatValue();
                    sumSquared += Math.pow(value - mean, 2);
                }
                float stdDev = (float) Math.sqrt(sumSquared / count);

                // Normalizar según el modo.
                for (Instance instance : data) {
                    float value = ((Number) instance.getValue(i)).floatValue();
                    float normalized = value;
                    if (mode == NormalizationMode.MIN_MAX) {
                        normalized = (max - min != 0) ? (value - min) / (max - min) : 0;
                    } else if (mode == NormalizationMode.Z_SCORE) {
                        normalized = (stdDev != 0) ? (value - mean) / stdDev : 0;
                    }
                    instance.setValue(i, normalized);
                }
            }
        }
    }

    /**
     * Genera y retorna una lista de estadísticas para cada atributo.
     * Para atributos cuantitativos: mínimo, máximo, media y desviación estándar.
     * Para atributos cualitativos: calcula las frecuencias relativas de cada categoría.
     */
    public List<AttributeStats> getAttributeStats() {
        List<AttributeStats> stats = new ArrayList<>();
        for (int i = 0; i < attributes.size(); i++) {
            Attribute attr = attributes.get(i);
            if (attr.isQuantitative()) {
                float min = Float.MAX_VALUE, max = -Float.MAX_VALUE, sum = 0, sumSquared = 0;
                int count = 0;
                for (Instance instance : data) {
                    float value = ((Number) instance.getValue(i)).floatValue();
                    min = Math.min(min, value);
                    max = Math.max(max, value);
                    sum += value;
                    count++;
                }
                float mean = sum / count;
                for (Instance instance : data) {
                    float value = ((Number) instance.getValue(i)).floatValue();
                    sumSquared += Math.pow(value - mean, 2);
                }
                float stdDev = (float) Math.sqrt(sumSquared / count);
                stats.add(new QuantitativeStats(attr.getName(), min, max, mean, stdDev));
            } else {
                Map<String, Integer> frequency = new HashMap<>();
                int total = 0;
                for (Instance instance : data) {
                    String val = instance.getValue(i).toString();
                    frequency.put(val, frequency.getOrDefault(val, 0) + 1);
                    total++;
                }
                Map<String, Float> relative = new HashMap<>();
                for (String key : frequency.keySet()) {
                    relative.put(key, frequency.get(key) / (float) total);
                }
                stats.add(new QualitativeStats(attr.getName(), relative));
            }
        }
        return stats;
    }
}
