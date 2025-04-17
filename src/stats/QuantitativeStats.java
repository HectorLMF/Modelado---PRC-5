package stats;

import model.AttributeStats;

public class QuantitativeStats implements AttributeStats {
    private String name;
    private float min;
    private float max;
    private float mean;
    private float stdDev;

    public QuantitativeStats(String name, float min, float max, float mean, float stdDev) {
        this.name = name;
        this.min = min;
        this.max = max;
        this.mean = mean;
        this.stdDev = stdDev;
    }

    @Override
    public String getName() {
        return name;
    }

    public float getMin() {
        return min;
    }

    public float getMax() {
        return max;
    }

    public float getMean() {
        return mean;
    }

    public float getStdDev() {
        return stdDev;
    }

    @Override
    public String toString() {
        return "QuantitativeStats{name='" + name + "', min=" + min + ", max=" + max +
                ", mean=" + mean + ", stdDev=" + stdDev + "}";
    }
}
