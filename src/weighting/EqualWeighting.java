package weighting;

public class EqualWeighting implements CaseWeightingStrategy {
    @Override
    public float getWeight(int rank, float distance) {
        return 1.0f;
    }
}