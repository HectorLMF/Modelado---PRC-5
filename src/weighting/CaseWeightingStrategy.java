package weighting;

public interface CaseWeightingStrategy {
    float getWeight(int rank, float distance);
}
