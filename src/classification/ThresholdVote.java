package classification;

import java.util.Map;
import java.util.Map.Entry;

public class ThresholdVote implements ClassificationRule {
    private float threshold;

    public ThresholdVote(float threshold) {
        this.threshold = threshold;
    }

    @Override
    public String predictClass(Map<String, Float> votes) {
        float totalVotes = 0;
        for (float v : votes.values()) {
            totalVotes += v;
        }

        String predicted = null;
        float maxVotes = -Float.MAX_VALUE;

        for (Entry<String, Float> entry : votes.entrySet()) {
            if (totalVotes > 0 && (entry.getValue() / totalVotes) >= threshold && entry.getValue() > maxVotes) {
                maxVotes = entry.getValue();
                predicted = entry.getKey();
            }
        }

        return predicted; // Puede retornar null si ningún voto supera el umbral.
    }

    public float getThreshold() {
        return threshold;
    }
}
