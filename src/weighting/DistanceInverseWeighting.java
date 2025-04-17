package weighting;

public class DistanceInverseWeighting implements CaseWeightingStrategy {
    @Override
    public float getWeight(int rank, float distance) {
        final float epsilon = 1e-5f;
        return 1.0f / (distance + epsilon);
    }
}