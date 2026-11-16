Tabla T1. Detección y segmentación en prueba, IoU 0.5: AP50 latente de cajas y de máscaras. Media ± sd entre lotes y valor de cada lote de prueba.

| Brazo | Métrica | n folds | Media ± sd | Lote 1 | Lote 2 | Lote 3 | Lote 4 | Lote 5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Maestro RGB | AP50 latente (cajas) | 5 | 0.306 ± 0.072 | 0.239 | 0.398 | 0.327 | 0.228 | 0.340 |
| Maestro NIR | AP50 latente (cajas) | 5 | 0.434 ± 0.080 | 0.353 | 0.558 | 0.448 | 0.374 | 0.439 |
| Maestro RGB+NIR | AP50 latente (cajas) | 5 | 0.388 ± 0.083 | 0.315 | 0.525 | 0.356 | 0.339 | 0.405 |
| Alumno RGB, detección | AP50 latente (cajas) | 5 | 0.307 ± 0.098 | 0.217 | 0.461 | 0.335 | 0.236 | 0.285 |
| Alumno RGB, CrossKD desde RGB | AP50 latente (cajas) | 5 | 0.318 ± 0.063 | 0.303 | 0.388 | 0.353 | 0.219 | 0.325 |
| Alumno RGB, CrossKD desde RGB+NIR | AP50 latente (cajas) | 5 | 0.354 ± 0.097 | 0.336 | 0.490 | 0.400 | 0.229 | 0.316 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | AP50 latente (cajas) | 5 | 0.316 ± 0.104 | 0.215 | 0.474 | 0.351 | 0.232 | 0.305 |
| Maestro RGB | AP50 latente (máscaras) | 5 | 0.295 ± 0.078 | 0.223 | 0.409 | 0.326 | 0.223 | 0.294 |
| Maestro NIR | AP50 latente (máscaras) | 5 | 0.429 ± 0.097 | 0.329 | 0.580 | 0.443 | 0.362 | 0.429 |
| Maestro RGB+NIR | AP50 latente (máscaras) | 5 | 0.391 ± 0.090 | 0.319 | 0.543 | 0.361 | 0.338 | 0.396 |
| Alumno RGB, detección | AP50 latente (máscaras) | 0 | n/d | n/d | n/d | n/d | n/d | n/d |
| Alumno RGB, CrossKD desde RGB | AP50 latente (máscaras) | 5 | 0.308 ± 0.067 | 0.281 | 0.395 | 0.344 | 0.217 | 0.304 |
| Alumno RGB, CrossKD desde RGB+NIR | AP50 latente (máscaras) | 5 | 0.334 ± 0.102 | 0.308 | 0.470 | 0.405 | 0.223 | 0.266 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | AP50 latente (máscaras) | 5 | 0.312 ± 0.100 | 0.220 | 0.465 | 0.353 | 0.236 | 0.284 |
