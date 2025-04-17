package app;

import model.Attribute;
import model.Dataset;
import model.Instance;
import model.NormalizationMode;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


public class CSVLoader {

    public static Dataset loadDataset(String filepath, NormalizationMode normMode) {
        List<String[]> rows = new ArrayList<>();
        String[] header = null;

        try (BufferedReader br = new BufferedReader(new FileReader(filepath))) {
            String line = br.readLine();
            if (line != null) {
                header = line.split(",");
            }
            while ((line = br.readLine()) != null) {
                if (!line.trim().isEmpty()) {
                    rows.add(line.split(","));
                }
            }
        } catch (IOException e) {
            System.out.println("Error leyendo el archivo CSV: " + e.getMessage());
            return null;
        }

        if (header == null || rows.isEmpty()) {
            System.out.println("El archivo CSV no contiene datos.");
            return null;
        }

        int numColumns = header.length;
        int classIndex = numColumns - 1;

        List<Attribute> attributes = new ArrayList<>();
        for (int i = 0; i < numColumns; i++) {
            boolean isClass = (i == classIndex);
            boolean quantitative = isClass ? false : isNumeric(rows, i);
            attributes.add(new Attribute(header[i].trim(), quantitative, isClass));
        }

        Dataset dataset = new Dataset(attributes);
        dataset.setClassAttributeIndex(classIndex);

        for (String[] row : rows) {
            Instance inst = new Instance();
            for (int i = 0; i < numColumns; i++) {
                String token = row[i].trim();
                if (attributes.get(i).isQuantitative()) {
                    try {
                        inst.addValue(Float.parseFloat(token));
                    } catch (NumberFormatException e) {
                        inst.addValue(0.0f);
                    }
                } else {
                    inst.addValue(token);
                }
            }
            dataset.addInstance(inst);
        }

        if (normMode != NormalizationMode.RAW) {
            dataset.normalize(normMode);
        }

        return dataset;
    }

    private static boolean isNumeric(List<String[]> rows, int colIndex) {
        for (String[] row : rows) {
            try {
                Float.parseFloat(row[colIndex].trim());
            } catch (NumberFormatException e) {
                return false;
            }
        }
        return true;
    }
}
