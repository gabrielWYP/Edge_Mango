Tabla T4. Datos de selección del checkpoint (validación): época elegida y AP50 en validación. No constituyen evidencia de generalización.

| Brazo | Métrica | n folds | Media ± sd | Lote 1 | Lote 2 | Lote 3 | Lote 4 | Lote 5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Maestro RGB | Época seleccionada | 5 | 44.4 ± 11.4 | 36.0 | 64.0 | 42.0 | 37.0 | 43.0 |
| Maestro NIR | Época seleccionada | 5 | 46.2 ± 8.8 | 38.0 | 48.0 | 36.0 | 55.0 | 54.0 |
| Maestro RGB+NIR | Época seleccionada | 5 | 37.0 ± 11.0 | 32.0 | 41.0 | 21.0 | 50.0 | 41.0 |
| Maestro RGB, solo cajas | Época seleccionada | 5 | 73.0 ± 1.9 | 72.0 | 73.0 | 73.0 | 76.0 | 71.0 |
| Alumno RGB, CrossKD desde RGB | Época seleccionada | 5 | 51.2 ± 10.2 | 55.0 | 49.0 | 36.0 | 52.0 | 64.0 |
| Alumno RGB, CrossKD desde RGB+NIR | Época seleccionada | 5 | 57.4 ± 8.6 | 69.0 | 49.0 | 50.0 | 63.0 | 56.0 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | Época seleccionada | 5 | 45.8 ± 19.9 | 21.0 | 46.0 | 36.0 | 51.0 | 75.0 |
| Maestro RGB | AP50 val, clase daño, en esa época | 5 | 0.407 ± 0.089 | 0.539 | 0.398 | 0.289 | 0.420 | 0.388 |
| Maestro NIR | AP50 val, clase daño, en esa época | 5 | 0.456 ± 0.073 | 0.546 | 0.460 | 0.346 | 0.488 | 0.438 |
| Maestro RGB+NIR | AP50 val, clase daño, en esa época | 5 | 0.465 ± 0.068 | 0.541 | 0.471 | 0.367 | 0.511 | 0.434 |
| Maestro RGB, solo cajas | AP50 val, clase daño, en esa época | 5 | 0.346 ± 0.041 | 0.391 | 0.374 | 0.293 | 0.315 | 0.358 |
| Alumno RGB, CrossKD desde RGB | AP50 val, clase daño, en esa época | 5 | 0.416 ± 0.083 | 0.511 | 0.446 | 0.288 | 0.441 | 0.397 |
| Alumno RGB, CrossKD desde RGB+NIR | AP50 val, clase daño, en esa época | 5 | 0.400 ± 0.069 | 0.483 | 0.407 | 0.292 | 0.413 | 0.403 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | AP50 val, clase daño, en esa época | 5 | 0.404 ± 0.064 | 0.493 | 0.405 | 0.315 | 0.419 | 0.390 |
| Maestro RGB | AP50 latente en val (checkpoint elegido) | 5 | 0.375 ± 0.076 | 0.467 | 0.371 | 0.260 | 0.412 | 0.366 |
| Maestro NIR | AP50 latente en val (checkpoint elegido) | 5 | 0.418 ± 0.070 | 0.516 | 0.398 | 0.323 | 0.438 | 0.414 |
| Maestro RGB+NIR | AP50 latente en val (checkpoint elegido) | 5 | 0.427 ± 0.066 | 0.510 | 0.424 | 0.337 | 0.468 | 0.398 |
| Maestro RGB, solo cajas | AP50 latente en val (checkpoint elegido) | 5 | 0.319 ± 0.041 | 0.336 | 0.352 | 0.267 | 0.283 | 0.356 |
| Alumno RGB, CrossKD desde RGB | AP50 latente en val (checkpoint elegido) | 5 | 0.381 ± 0.075 | 0.454 | 0.410 | 0.256 | 0.403 | 0.381 |
| Alumno RGB, CrossKD desde RGB+NIR | AP50 latente en val (checkpoint elegido) | 5 | 0.368 ± 0.060 | 0.428 | 0.376 | 0.266 | 0.378 | 0.389 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | AP50 latente en val (checkpoint elegido) | 5 | 0.365 ± 0.056 | 0.440 | 0.368 | 0.282 | 0.370 | 0.364 |
