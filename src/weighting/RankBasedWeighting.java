package weighting;

import java.util.List;

public class RankBasedWeighting implements CaseWeightingStrategy {
    private List<Float> rankWeights;

    public RankBasedWeighting(List<Float> rankWeights) {
        this.rankWeights = rankWeights;
    }

    @Override
    public float getWeight(int rank, float distance) {
        if (rank - 1 < rankWeights.size()) {
            return rankWeights.get(rank - 1);
        }
        return 1.0f;
    }
}
