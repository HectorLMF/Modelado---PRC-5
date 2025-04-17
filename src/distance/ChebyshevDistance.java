package distance;

import model.Attribute;
import model.Instance;

import java.util.List;

public class ChebyshevDistance implements DistanceMetric {
    @Override
    public float compute(Instance a, Instance b, List<Attribute> attributes, List<Float> weights) {
        float maxDiff = 0;
        List<Object> valuesA = a.getValues();
        List<Object> valuesB = b.getValues();
        for (int i = 0; i < attributes.size(); i++) {
            if (attributes.get(i).isClass())
                continue;
            float weight = (weights != null && i < weights.size()) ? weights.get(i) : 1.0f;
            float diff = 0;
            if (attributes.get(i).isQuantitative()) {
                float vA = ((Number) valuesA.get(i)).floatValue();
                float vB = ((Number) valuesB.get(i)).floatValue();
                diff = weight * Math.abs(vA - vB);
            } else {
                String sA = valuesA.get(i).toString();
                String sB = valuesB.get(i).toString();
                diff = sA.equals(sB) ? 0f : weight;
            }
            if (diff > maxDiff) {
                maxDiff = diff;
            }
        }
        return maxDiff;
    }
}
