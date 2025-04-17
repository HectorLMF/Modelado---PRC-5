package distance;

import model.Attribute;
import model.Instance;

import java.util.List;

public interface DistanceMetric {
    float compute(Instance a, Instance b, List<Attribute> attributes, List<Float> weights);
}
