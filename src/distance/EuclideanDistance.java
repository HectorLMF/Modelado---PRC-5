package distance;

import model.Attribute;
import model.Instance;

import java.util.List;

public class EuclideanDistance implements DistanceMetric {
    @Override
    public float compute(Instance a, Instance b, List<Attribute> attributes, List<Float> attributeWeights) {
        float sum = 0;
        List<Object> valuesA = a.getValues();
        List<Object> valuesB = b.getValues();
        for (int i = 0; i < attributes.size(); i++) {
            // Omitir el atributo de clase
            if (attributes.get(i).isClass())
                continue;
            float weight = (attributeWeights != null && i < attributeWeights.size()) ? attributeWeights.get(i) : 1.0f;
            if (attributes.get(i).isQuantitative()) {
                float vA = ((Number) valuesA.get(i)).floatValue();
                float vB = ((Number) valuesB.get(i)).floatValue();
                sum += weight * (vA - vB) * (vA - vB);
            } else {
                // Para cualitativos: 0 si iguales, 1 si distintos.
                String sA = valuesA.get(i).toString();
                String sB = valuesB.get(i).toString();
                float diff = sA.equals(sB) ? 0f : 1f;
                sum += weight * diff * diff;
            }
        }
        return (float) Math.sqrt(sum);
    }
}

