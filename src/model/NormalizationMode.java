package model;

public enum NormalizationMode {
    RAW,      // No se aplica normalización
    MIN_MAX,  // Escala los valores al rango [0,1]
    Z_SCORE   // Estandarización (media 0, desviación 1)
}
