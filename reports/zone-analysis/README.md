# AP por zona de impacto: ¿iluminación o posición?

Análisis exploratorio, 2026-11-05. Pregunta: el AP de daño es bajo en `derecha`, `superior` e `inferior` y alto en `centro` e `izquierda`. ¿Se debe a la iluminación?

```bash
python scripts/zone_analysis.py --reports-root ~/edge_mango/runs/final/reports \
    --dataset ~/edge_mango/data/v4_captura_v3_final --test-folds 0 1 --test-seeds 42 \
    --photos-export reports/final-runs-export/photos.csv --output reports/zone-analysis
python scripts/plot_zone_analysis.py reports/zone-analysis
```

El panel 3 de la figura y `zone_within.csv` (opción `--photos-export`) cruzan `zone_photos.csv` con `reports/final-runs-export/photos.csv`.

## Datos

- **AP por zona:** 12 evaluaciones de validación (folds 0-3, brazos terminados) y 8 de test (folds 0 y 1, semilla 42, leídas según la sección 8.6 del plan). El test de los demás folds no se leyó.
- **Condiciones por zona:** las 3466 fotos con caja de daño del dataset, medidas sobre las imágenes y las etiquetas, sin salidas de modelos.

## Resultados

**1. Los nombres de zona están espejados respecto de la imagen.** Se refieren a la fruta, no a la foto: `izquierda` cae en el borde derecho de la imagen (x = +0.60), `superior` abajo (y = +0.50), `inferior` arriba (y = −0.66), y `derecha` a la izquierda pero cerca del centro (x = −0.26).

**2. El orden de las zonas es estable.**

| Zona | AP50 latente, validación | AP50 latente, test | Distancia al centro | Brillo caja / fruta | Contraste RGB | Contraste NIR |
| :-- | --: | --: | --: | --: | --: | --: |
| `izquierda` | 0.501 | 0.673 | 0.62 | 1.70 | 1.86 | 1.75 |
| `centro` | 0.402 | 0.501 | 0.22 | 0.95 | 0.30 | 0.26 |
| `derecha` | 0.203 | 0.154 | 0.32 | 0.71 | 0.68 | 0.59 |
| `superior` | 0.096 | 0.046 | 0.59 | 1.15 | 0.48 | 0.36 |
| `inferior` | 0.046 | 0.084 | 0.73 | 0.75 | 0.63 | 0.39 |

`derecha` es la mejor de las tres zonas malas en las 20 evaluaciones (AP de todas las horas).

**3. La luz no explica las tres zonas malas.** No comparten una condición de luz: `superior` está más iluminada que el promedio de la fruta (1.15) y `derecha` e `inferior` menos (0.71 y 0.75). La zona más oscura, `derecha`, es la mejor de las tres. `centro`, algo más oscura que el promedio y con el menor contraste, es la segunda mejor.

**4. La luz sí explica la mejor zona.** La iluminación entra por el lado derecho de la imagen, en los cinco lotes: el brillo relativo sube de 0.64 en el borde izquierdo a 1.72 en el derecho. `izquierda` cae en ese lado: su caja está un 70 % más iluminada que la fruta, con un contraste 3 a 6 veces mayor que el de las demás zonas, y es la mejor zona aunque esté lejos del centro.

**5. La posición explica las demás.** Fuera de `izquierda`, el AP baja con la distancia del daño al centro de la fruta en la imagen: `centro` (0.22) > `derecha` (0.32) > `superior` (0.59) ≈ `inferior` (0.73). Dentro de cada una de las cinco zonas, la mitad de las fotos con el daño más cerca del centro tiene más aciertos que la mitad más lejana (`zone_within.csv`): 0.80 frente a 0.74 en `centro`, 0.84 frente a 0.68 en `izquierda`, 0.73 frente a 0.24 en `derecha`, 0.42 frente a 0.15 en `superior` y 0.51 frente a 0.19 en `inferior`. Dividir por brillo dentro de cada zona da signos opuestos según la zona, porque dentro de una zona el brillo depende de dónde cae la caja respecto del lado iluminado.

## Lectura

- La afirmación "las tres zonas malas son malas por la iluminación" no se sostiene: esas zonas no comparten la condición de luz, y la más oscura es la mejor de las tres.
- La iluminación sí importa, pero explica por qué `izquierda` es buena: es el lado iluminado.
- La explicación consistente con todas las zonas es la posición: cerca del borde de la fruta la superficie queda oblicua a la cámara, y el daño se ve comprimido y con menos contraste, salvo en el lado iluminado.

## Límites

- Son cinco zonas y la posición de la luz y la cámara es fija, así que zona, posición y luz están parcialmente confundidas. No hay inferencia.
- Las evaluaciones agrupadas no son independientes: las mismas fotos aparecen en varios brazos.
- El brillo y el contraste se miden en la caja dibujada: el contraste incluye diferencias de iluminación, no solo del daño.
- "Acierto" es al menos una caja correcta (IoU 0.5) a confianza 0.25.

## Archivos

| Archivo | Contenido |
| :-- | :-- |
| `zone_ap.csv` | AP50 por zona de cada evaluación, de todas las horas y de la ventana latente, con el número de cajas |
| `zone_photos.csv` | Una fila por foto con daño: posición, tamaño, brillo y contraste de la caja |
| `zone_conditions.csv` | Medianas por zona, de todas las horas y de la ventana latente |
| `zone_within.csv` | Aciertos por zona, partidos por la mediana de brillo y de distancia dentro de la zona |
| `zone_analysis.png` | La figura |
