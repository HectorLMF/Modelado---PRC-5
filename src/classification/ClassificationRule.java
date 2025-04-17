package classification;

import java.util.Map;

public interface ClassificationRule {
    String predictClass(Map<String, Float> votes);
}
