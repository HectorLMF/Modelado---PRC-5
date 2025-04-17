package model;

import java.util.ArrayList;
import java.util.List;

public class Instance {
    private List<Object> values;

    public Instance() {
        values = new ArrayList<>();
    }

    public Instance(List<Object> values) {
        this.values = new ArrayList<>(values);
    }

    public Object getValue(int index) {
        return values.get(index);
    }

    public void setValue(int index, Object value) {
        values.set(index, value);
    }

    public List<Object> getValues() {
        return values;
    }

    public void addValue(Object value) {
        values.add(value);
    }
}
