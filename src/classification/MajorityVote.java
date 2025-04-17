package classification;

import java.util.Map;
import java.util.Map.Entry;

public class MajorityVote implements ClassificationRule {
    @Override
    public String predictClass(Map<String, Float> votes) {
        String predicted = null;
        Float maxVotes = -1F;
        for (Entry<String, Float> entry : votes.entrySet()) {
            if (entry.getValue() > maxVotes) {
                maxVotes = entry.getValue();
                predicted = entry.getKey();
            }
        }
        return predicted;
    }
}
