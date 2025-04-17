package model;

public class Attribute {
    private String name;
    private boolean quantitative;
    private boolean isClass;

    public Attribute(String name, boolean quantitative, boolean isClass) {
        this.name = name;
        this.quantitative = quantitative;
        this.isClass = isClass;
    }

    public String getName() {
        return name;
    }

    public boolean isQuantitative() {
        return quantitative;
    }

    public boolean isClass() {
        return isClass;
    }

    @Override
    public String toString() {
        return "Attribute{name='" + name + "', quantitative=" + quantitative + ", isClass=" + isClass + "}";
    }
}
