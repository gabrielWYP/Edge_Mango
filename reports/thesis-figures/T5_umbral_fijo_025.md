Tabla T5. Unidad de decisión = foto (daño si alguna detección alcanza el umbral): umbral fijo 0.25 (umbral de evaluación) en prueba. Exhaustividad, especificidad, precisión y tasa de descarte de fotos sanas, más precisión y exhaustividad de cajas. Reglas definidas a posteriori; descriptivas. Brazos principales (pre-registrados y referencia NIR).

| Brazo | Métrica | n folds | Media ± sd | Lote 1 | Lote 2 | Lote 3 | Lote 4 | Lote 5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Maestro RGB | Exhaustividad por foto (sensibilidad en fotos dañadas) | 5 | 1.000 ± 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Maestro NIR | Exhaustividad por foto (sensibilidad en fotos dañadas) | 5 | 0.999 ± 0.002 | 1.000 | 1.000 | 1.000 | 1.000 | 0.995 |
| Maestro RGB+NIR | Exhaustividad por foto (sensibilidad en fotos dañadas) | 5 | 1.000 ± 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Alumno RGB, CrossKD desde RGB | Exhaustividad por foto (sensibilidad en fotos dañadas) | 5 | 1.000 ± 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Alumno RGB, CrossKD desde RGB+NIR | Exhaustividad por foto (sensibilidad en fotos dañadas) | 5 | 1.000 ± 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Maestro RGB | Especificidad por foto (fotos sanas conservadas) | 5 | 0.000 ± 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Maestro NIR | Especificidad por foto (fotos sanas conservadas) | 5 | 0.002 ± 0.004 | 0.000 | 0.000 | 0.010 | 0.000 | 0.000 |
| Maestro RGB+NIR | Especificidad por foto (fotos sanas conservadas) | 5 | 0.000 ± 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Alumno RGB, CrossKD desde RGB | Especificidad por foto (fotos sanas conservadas) | 5 | 0.000 ± 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Alumno RGB, CrossKD desde RGB+NIR | Especificidad por foto (fotos sanas conservadas) | 5 | 0.000 ± 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Maestro RGB | Precisión por foto | 5 | 0.799 ± 0.001 | 0.800 | 0.798 | 0.799 | 0.800 | 0.799 |
| Maestro NIR | Precisión por foto | 5 | 0.799 ± 0.001 | 0.800 | 0.798 | 0.800 | 0.800 | 0.798 |
| Maestro RGB+NIR | Precisión por foto | 5 | 0.799 ± 0.001 | 0.800 | 0.798 | 0.799 | 0.800 | 0.799 |
| Alumno RGB, CrossKD desde RGB | Precisión por foto | 5 | 0.799 ± 0.001 | 0.800 | 0.798 | 0.799 | 0.800 | 0.799 |
| Alumno RGB, CrossKD desde RGB+NIR | Precisión por foto | 5 | 0.799 ± 0.001 | 0.800 | 0.798 | 0.799 | 0.800 | 0.799 |
| Maestro RGB | Tasa de descarte de sanas (1 - especificidad) | 5 | 1.000 ± 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Maestro NIR | Tasa de descarte de sanas (1 - especificidad) | 5 | 0.998 ± 0.004 | 1.000 | 1.000 | 0.990 | 1.000 | 1.000 |
| Maestro RGB+NIR | Tasa de descarte de sanas (1 - especificidad) | 5 | 1.000 ± 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Alumno RGB, CrossKD desde RGB | Tasa de descarte de sanas (1 - especificidad) | 5 | 1.000 ± 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Alumno RGB, CrossKD desde RGB+NIR | Tasa de descarte de sanas (1 - especificidad) | 5 | 1.000 ± 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Maestro RGB | Línea base «todo positivo»: F1 (foto) | 5 | 0.888 ± 0.001 | 0.889 | 0.887 | 0.888 | 0.889 | 0.888 |
| Maestro NIR | Línea base «todo positivo»: F1 (foto) | 5 | 0.888 ± 0.001 | 0.889 | 0.887 | 0.888 | 0.889 | 0.888 |
| Maestro RGB+NIR | Línea base «todo positivo»: F1 (foto) | 5 | 0.888 ± 0.001 | 0.889 | 0.887 | 0.888 | 0.889 | 0.888 |
| Alumno RGB, CrossKD desde RGB | Línea base «todo positivo»: F1 (foto) | 5 | 0.888 ± 0.001 | 0.889 | 0.887 | 0.888 | 0.889 | 0.888 |
| Alumno RGB, CrossKD desde RGB+NIR | Línea base «todo positivo»: F1 (foto) | 5 | 0.888 ± 0.001 | 0.889 | 0.887 | 0.888 | 0.889 | 0.888 |
| Maestro RGB | Precisión de cajas (IoU 0.5) | 5 | 0.104 ± 0.021 | 0.119 | 0.130 | 0.098 | 0.075 | 0.098 |
| Maestro NIR | Precisión de cajas (IoU 0.5) | 5 | 0.143 ± 0.041 | 0.080 | 0.160 | 0.145 | 0.135 | 0.193 |
| Maestro RGB+NIR | Precisión de cajas (IoU 0.5) | 5 | 0.100 ± 0.009 | 0.089 | 0.098 | 0.098 | 0.111 | 0.106 |
| Alumno RGB, CrossKD desde RGB | Precisión de cajas (IoU 0.5) | 5 | 0.127 ± 0.021 | 0.132 | 0.146 | 0.112 | 0.100 | 0.145 |
| Alumno RGB, CrossKD desde RGB+NIR | Precisión de cajas (IoU 0.5) | 5 | 0.132 ± 0.029 | 0.128 | 0.181 | 0.118 | 0.103 | 0.130 |
| Maestro RGB | Exhaustividad de cajas (IoU 0.5) | 5 | 0.711 ± 0.075 | 0.642 | 0.769 | 0.801 | 0.632 | 0.713 |
| Maestro NIR | Exhaustividad de cajas (IoU 0.5) | 5 | 0.723 ± 0.082 | 0.657 | 0.850 | 0.756 | 0.655 | 0.695 |
| Maestro RGB+NIR | Exhaustividad de cajas (IoU 0.5) | 5 | 0.751 ± 0.105 | 0.649 | 0.904 | 0.788 | 0.657 | 0.756 |
| Alumno RGB, CrossKD desde RGB | Exhaustividad de cajas (IoU 0.5) | 5 | 0.675 ± 0.101 | 0.637 | 0.746 | 0.804 | 0.545 | 0.645 |
| Alumno RGB, CrossKD desde RGB+NIR | Exhaustividad de cajas (IoU 0.5) | 5 | 0.682 ± 0.105 | 0.612 | 0.799 | 0.776 | 0.555 | 0.670 |
