package distance;

import model.Attribute;
import model.Instance;

import java.util.List;

public class ManhattanDistance implements DistanceMetric {
    @Override
    public float compute(Instance a, Instance b, List<Attribute> attributes, List<Float> weights) {
        float sum = 0;
        List<Object> valuesA = a.getValues();
        List<Object> valuesB = b.getValues();
        for (int i = 0; i < attributes.size(); i++) {
            if (attributes.get(i).isClass())
                continue;
            float weight = (weights != null && i < weights.size()) ? weights.get(i) : 1.0f;
            if (attributes.get(i).isQuantitative()) {
                float vA = ((Number) valuesA.get(i)).floatValue();
                float vB = ((Number) valuesB.get(i)).floatValue();
                sum += weight * Math.abs(vA - vB);
            } else {
                String sA = valuesA.get(i).toString();
                String sB = valuesB.get(i).toString();
                float diff = sA.equals(sB) ? 0f : 1f;
                sum += weight * Math.abs(diff);
            }
        }
        return sum;
    }
    public String getMetricName() {
        return "Manhattan Distance";
    }
}
