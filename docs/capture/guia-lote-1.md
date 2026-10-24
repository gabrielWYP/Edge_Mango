# Guía de captura del lote 1 (mangos 1-25)

Procedimiento completo del operador para el primer lote, paso a paso y con los comandos. El manual del
programa está en [`capture/README.md`](../../capture/README.md).

| | |
| --- | --- |
| Programa | `captura_v3` (`capture/` en el repositorio, `~/captura_v3/` en la Pi) |
| Mangos | 1-25 (los IDs son globales: el lote 2 es 26-50, y así hasta 125) |
| Momentos | `sano` + 7 después del impacto: +0, +2, +4, +6, +8, +10 y +12 h |
| Pares RGB+NIR | 25 × 4 caras × 8 momentos = **800** (100 sanos y 700 con daño) |
| Disco | ~1,4 MB por cara (RGB ~0,3 MB, NIR ~1,1 MB): ~1,2 GB en total (la Pi tiene ~37 GB libres) |
| Duración | ~14,5 h con un operador (ejemplo: 06:30-21:00) |

## Resumen

| Paso | Qué haces | Comando | Duración |
| --- | --- | --- | --- |
| 1 | Comprar los 25 mangos | — | la víspera |
| 2 | Marcarlos | — | ~10 min |
| 3 | Ordenarlos en la placa | — | ~5 min |
| 4 | Arranque y calibración | `captura.py jornada` | ~15 min |
| 5 | Fotos sanas de las 4 caras | `captura.py sanas` | ~40 min |
| 6 | Golpear y fotografiar (+0 h), mango por mango | `captura.py impactos` | ~75 min |
| 7 | Rondas +2 h … +12 h | `captura.py ronda` | 6 × ~75 min, con ~45 min libres entre rondas |
| 8 | Respaldo y cierre | `rsync`, `captura.py estado` | ~10 min |

Teclas en todo el programa: **Enter** = seguir · **s** = omitir la cara · **q** = salir. Después de salir
(o de un corte), vuelve a correr el mismo comando: el avance se guarda después de cada cara y solo se pierde
la cara en curso.

## Antes del día de captura

### Alimentación de las luces

El script enciende y apaga las luces con los relés, y eso funciona. Lo que falla es la **alimentación**: tras un
rato sin uso, la luz deja de encender aunque el relé conmute, y reintentar no la recupera. Medido en la Pi el
2026-10-24 (brillo medio de una toma RGB con la caja cerrada):

| Momento | Brillo medio |
| --- | --- |
| Al empezar | 129,9 (encendida) |
| Tras 90 s sin uso | 10,0 (apagada: solo la luz ambiente de la caja) |
| Tras 180 s sin uso | 128,8 |
| Tras 60, 90 y 120 s sin uso, con reintento inmediato y otro a los 2 s | 10,0 en los 9 intentos |

En el ensayo del 2026-10-23 pasó lo mismo con la RGB y la NIR a la vez, después de ~1 min de espera. Las luces se
alimentan de una batería USB que se duerme sin carga. Entre mangos de una ronda hay más de un minuto de espera,
así que esto pasaría casi en cada mango.

- **Solución:** alimenta las luces con un cargador de pared de 5 V (≥ 2 A) en lugar de la batería. Las luces
  siguen dentro de la caja; solo cambia de dónde viene el cable.
- **Si sigues con la batería:** presiona su botón antes de cada mango. Si el QC muestra `luz apagada?`,
  despiértala y presiona Enter para repetir la cara.

### Opcional

- **tmux** (`sudo apt install tmux`): conserva la sesión si se corta el SSH. No es imprescindible, porque el
  estado se reanuda con el mismo comando. Si operas desde el escritorio de la Pi, no hace falta.
- **Ensayo con reloj acelerado** (~5 min, 1 "hora" = 1 min) en una carpeta aparte, para no tocar la
  configuración real:

  ```bash
  cd ~/captura_v3
  python3 captura.py --auto --base ~/capturas_ensayo jornada --id ENSAYO --lote ensayo --mangos 1-2 --caras 1,2 \
      --intervalos 0,1,2 --unidad-min 1 --slot-min 0.5 --tolerancia-min 0.5 --no-subir --semilla 1
  python3 captura.py --auto --base ~/capturas_ensayo sanas
  python3 captura.py --auto --base ~/capturas_ensayo impactos
  python3 captura.py --auto --base ~/capturas_ensayo ronda
  rm -rf ~/capturas_ensayo
  ```

- **Redesplegar el código**, solo si cambió en el repositorio (la versión de PR #47 ya está en la Pi).
  `core.autocrlf=false` es obligatorio: con `true` (la configuración global de esta máquina), `git archive`
  convierte los archivos a CRLF.

  ```bash
  git -c core.autocrlf=false archive HEAD capture | ssh mangopi 'tar -x -C ~/captura_v3 --strip-components=1'
  ```

## El día de captura

### 1. Comprar los 25 mangos

De madurez parecida y sin golpes ni manchas. Déjalos varias horas (idealmente la noche anterior) en el cuarto de
captura para que tomen su temperatura.

### 2. Marcarlos

- El **número** (1-25) va en un post-it **sobre la placa**, junto a cada mango, no pegado al mango: pegado
  saldría en las fotos y se puede caer.
- En cada mango, un **punto pequeño** con marcador fino cerca del pedúnculo, fuera de la zona que se
  fotografía. Marca la **cara 1**.
- Las caras 2, 3 y 4 se obtienen rotando 90° **siempre en el mismo sentido**. Así cada cara es la misma en
  todas las fotos del día.

### 3. Ordenarlos en la placa

Separados y en orden del 1 al 25. Ten a mano la espuma, el tubo de PVC de 50 cm, la bola de 66 g, la tarjeta
gris y el tablero ArUco.

### 4. Arranque y calibración (~15 min)

```bash
ssh mangopi
```

```bash
rpicam-hello --list-cameras
```

Debe listar `imx219`. Si no aparece, reasienta la cinta CSI de la NoIR.

Si el montaje se movió desde la última vez, enfoca a mano los dos lentes (Ctrl-C para terminar):

```bash
python3 ~/captura_v3/captura.py enfoque --camara rgb
python3 ~/captura_v3/captura.py enfoque --camara nir
```

Crea la jornada. Usa la fecha real en el ID y la temperatura del cuarto:

```bash
python3 ~/captura_v3/captura.py jornada --no-subir --id lote1-AAAAMMDD --lote "lote 1" --mangos 1-25 --temperatura-c 24
```

El programa pide, en orden:

1. la **tarjeta gris** (calibra el balance de blancos);
2. la **bandeja vacía** (la referencia del control de calidad);
3. el **tablero ArUco** (para la homografía RGB-NIR).

Al final verifica que la cámara respete la exposición manual. `--no-subir` es porque la Pi no tiene
configurada la URL de OCI: los archivos se respaldan con `rsync` (paso 8).

### 5. Fotos sanas de las 4 caras (~40 min)

```bash
python3 ~/captura_v3/captura.py sanas
```

En cada `Mango N · cara C → coloca y presiona Enter`, pon esa cara hacia la cámara y presiona Enter
(~20 s por cara con la manipulación).

### 6. Golpear y fotografiar, mango por mango (~75 min)

```bash
python3 ~/captura_v3/captura.py impactos
```

Para cada mango, en orden:

1. El programa muestra la zona sorteada de cada cara (`centro`, `superior`, `inferior`, `izquierda` o
   `derecha`).
2. Golpea las 4 caras. En cada una: apoya el mango en la espuma con esa cara arriba, alinea el tubo sobre la
   zona, suelta la bola y presiona Enter **al soltarla**. Atrapa la bola tras el rebote para que no golpee dos
   veces.
3. Apenas terminas el cuarto golpe, el programa pide las 4 fotos +0 h del mismo mango. Tómalas rotándolo.

Mantén el ritmo de un mango cada ~3 min. El programa muestra el desfase, y las rondas lo heredan.

### 7. Rondas, hasta +12 h

```bash
python3 ~/captura_v3/captura.py ronda
```

No hay que esperar "desde el primero" ni calcular nada. Cada mango vence exactamente a las 2, 4, … 12 h de
**su propio** golpe (tolerancia ±10 min). Como los golpeaste en orden y a ritmo parejo, las rondas siguen ese
mismo orden, y el programa dice qué mango toca y cuándo, con cuenta regresiva. Un solo comando cubre las 6
rondas.

Ejemplo, con los golpes de 07:45 a 09:00:

| Ronda | Primer mango vence | Última foto (aprox.) |
| --- | --- | --- |
| +2 h | 09:45 | 11:00 |
| +4 h | 11:45 | 13:00 |
| +6 h | 13:45 | 15:00 |
| +8 h | 15:45 | 17:00 |
| +10 h | 17:45 | 19:00 |
| +12 h | 19:45 | 21:00 |

### 8. Respaldo y cierre

En cada descanso entre rondas, desde WSL y en la raíz del repositorio (la carpeta `data/cache/` no se
versiona):

```bash
rsync -av mangopi:capturas/lote1-AAAAMMDD/ data/cache/incoming/AAAAMMDD/lote1-AAAAMMDD/
```

Hazlo solo durante la cuenta regresiva, no mientras se captura: en la Zero 2 W el cifrado de SSH compite por
CPU con la cámara.

Al terminar la ronda +12 h:

```bash
python3 ~/captura_v3/captura.py estado
```

Todo debe estar en ✓ (800 pares si no se omitió ninguna cara). Haz el último `rsync` y anota la temperatura
final del cuarto.

## Si algo falla

| Síntoma | Qué hacer |
| --- | --- |
| `luz apagada?` | La alimentación de las luces se durmió: despierta la batería (o revisa el cargador) y presiona Enter para repetir la cara. |
| `la NoIR quedó trabada` | `sudo reboot` y vuelve a correr el mismo comando; el estado se conserva. |
| Se cortó el SSH | `ssh mangopi` y vuelve a correr el mismo comando (con tmux: `tmux attach`). |
| Una ronda salió tarde | Captúrala igual (queda marcada fuera de tolerancia) o omítela con `s`. |
| `la cámara RGB no está respetando la exposición manual` | Desconecta y reconecta la cámara (o reinicia la Pi) y repite `jornada`. |
| La NIR trae una fila oscura parcial | Es el contacto CSI marginal y el QC no lo detecta: revisa las NIR al respaldarlas y anota las caras afectadas. |
