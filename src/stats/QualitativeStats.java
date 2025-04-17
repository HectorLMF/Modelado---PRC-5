package stats;

import model.AttributeStats;

import java.util.Map;

public class QualitativeStats implements AttributeStats {
    private String name;
    private Map<String, Float> categoryFrequencies;

    public QualitativeStats(String name, Map<String, Float> categoryFrequencies) {
        this.name = name;
        this.categoryFrequencies = categoryFrequencies;
    }

    @Override
    public String getName() {
        return name;
    }

    public Map<String, Float> getCategoryFrequencies() {
        return categoryFrequencies;
    }

    @Override
    public String toString() {
        return "QualitativeStats{name='" + name + "', categoryFrequencies=" + categoryFrequencies + "}";
    }
}

