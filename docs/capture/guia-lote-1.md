# Guía de captura del lote 1 (mangos 1-25)

Lista de pasos para el operador del primer lote. El manual completo del programa está en
[`capture/README.md`](../../capture/README.md); aquí solo va lo necesario para el día de captura.

| | |
| --- | --- |
| Programa | `captura_v3` (`capture/` en el repositorio, `~/captura_v3/` en la Pi) |
| Mangos | 1-25 (los IDs son globales: el lote 2 es 26-50, y así hasta 125) |
| Momentos | `sano` + 7 después del impacto: +0, +2, +4, +6, +8, +10 y +12 h |
| Pares RGB+NIR | 25 × 4 caras × 8 momentos = **800** (100 sanos y 700 con daño) |
| Disco | ~1,4 MB por cara (RGB ~0,3 MB, NIR ~1,1 MB): ~1,2 GB en total (la Pi tiene ~37 GB libres) |
| Duración | ~14,5 h con un operador (ejemplo: 06:30-21:00) |

## Por qué este flujo es el de menor overhead

La duración del día la fija el protocolo, no la cámara: la última ronda vence 12 h después del último
impacto. Por eso no conviene bajar la resolución para ganar segundos (la toma de un par con su QC tarda 5-8 s,
medido en la Pi el 2026-10-23, y la RGB a resolución completa conserva el detalle fino; ver
[PR #43](https://github.com/gabrielWYP/Edge_Mango/pull/43)). El overhead que sí importa es el del
operador, y el programa lo elimina:

| Antes (`captura_dual.py`, 2026-10-18) | Ahora (`captura_v3`) |
| --- | --- |
| Un comando escrito a mano por cara, con 10 opciones | Un Enter por cara |
| Hora de impacto tecleada a mano (con errores de tipeo) | Hora de cada cara registrada al soltar la bola |
| El operador calcula cuándo toca cada mango | El programa calcula los vencimientos y hace la cuenta regresiva |
| Sin control de calidad | QC contra la bandeja vacía en cada cara (detecta luz apagada) |
| Una caída pierde el contexto | Estado atómico: se reanuda con el mismo comando |

## La víspera

1. **Luces a una fuente de pared de 5 V** en lugar de la batería USB. La batería se duerme a los ~30 s sin
   carga y entre mangos siempre pasa más que eso; con la batería, hay que presionar su botón antes de cada
   mango o el QC rechaza la cara con `luz apagada?`. En el ensayo del 2026-10-23 bastó una espera de ~1 min
   entre rondas para que pasara.
2. **tmux en la Pi** (`sudo apt install tmux`). Una sesión de 14 h por SSH no puede depender de la Wi-Fi.
   Si no se instala, corre el programa desde una terminal en el escritorio de la Pi.
3. **Mangos:** numéralos del 1 al 25 y marca las caras 1-4 con un marcador fino cerca del pedúnculo, fuera
   de la zona que se fotografía, para que cada cara sea la misma en todas las rondas. Déjalos toda la noche
   en el cuarto de captura, en una bandeja numerada y en orden.
4. **Montaje:** reflector difuso del lado izquierdo (el lado en sombra; sin él, el tercio izquierdo del mango
   queda en V ≈ 40 contra ~90 del derecho). Espuma, tubo de PVC de 50 cm, bola de 66 g, tarjeta gris y
   tablero ArUco a mano.
5. **Desplegar el código** desde el repositorio. `core.autocrlf=false` es obligatorio: con `true` (la
   configuración global de esta máquina), `git archive` convierte los archivos a CRLF.

   ```bash
   git -c core.autocrlf=false archive HEAD capture | ssh mangopi 'tar -x -C ~/captura_v3 --strip-components=1'
   ssh mangopi 'python3 ~/captura_v3/captura.py jornada --help | tail -15'
   ```

   La ayuda debe decir que los intervalos por defecto son `0,2,4,6,8,10,12` y el slot 3 min.
6. **Ensayo con reloj acelerado** (~5 min, 1 "hora" = 1 min). Despierta las luces antes de cada paso:

   ```bash
   cd ~/captura_v3
   python3 captura.py --auto jornada --id ENSAYO --lote ensayo --mangos 1-2 --caras 1,2 \
       --intervalos 0,1,2 --unidad-min 1 --slot-min 0.5 --tolerancia-min 0.5 --no-subir --semilla 1
   python3 captura.py --auto sanas && python3 captura.py --auto impactos && python3 captura.py --auto ronda
   python3 captura.py estado
   rm -rf ~/capturas/ENSAYO
   ```

## El día de captura

### 1. Arranque (06:30, ~20 min)

```bash
ssh mangopi
tmux new -A -s lote1
rpicam-hello --list-cameras        # debe listar imx219; si no, reasienta la cinta CSI
timedatectl | head -5              # "System clock synchronized: yes"
vcgencmd get_throttled             # throttled=0x0
```

Si el montaje se movió, enfoca a mano los dos lentes (`captura.py enfoque --camara rgb` con luz blanca y
`--camara nir` con luz NIR; Ctrl-C para terminar).

Crea la jornada y sigue sus indicaciones (tarjeta gris → barrido de balance de blancos, bandeja vacía,
tablero ArUco, autoverificación de exposición):

```bash
python3 ~/captura_v3/captura.py jornada --no-subir --id lote1-AAAAMMDD --lote "lote 1" --mangos 1-25 --temperatura-c 24.5
```

`--no-subir` porque la Pi no tiene configurada la URL de OCI: los archivos se respaldan con `rsync` (paso 5).

### 2. Sanas (07:00-07:40)

```bash
python3 ~/captura_v3/captura.py sanas
```

Por cada cara: colócala hacia la cámara y presiona Enter (~20 s por cara con la manipulación).

### 3. Impactos y fotos +0 h (07:45-09:00)

```bash
python3 ~/captura_v3/captura.py impactos
```

Por cada mango, en orden:

1. El programa muestra las 4 zonas sorteadas (`centro`, `superior`, `inferior`, `izquierda` o `derecha`).
2. Para cada cara: apoya el mango en la espuma con esa cara arriba, alinea el tubo sobre la zona, suelta la
   bola y presiona Enter **al soltarla**. Atrapa la bola tras el rebote para que no golpee dos veces.
3. Apenas termina el cuarto impacto, el programa pide las 4 fotos +0 h del mismo mango.

Mantén el ritmo de un mango cada 3 min: el programa muestra el desfase, y las rondas lo heredan.

### 4. Rondas (+2 h a +12 h)

```bash
python3 ~/captura_v3/captura.py ronda
```

Un solo comando cubre las 6 rondas: muestra qué mango vence, hace la cuenta regresiva si es temprano y pide
las 4 caras. Cada mango se fotografía a *k* horas de **su** impacto (tolerancia ±10 min).

| Ronda | Primer mango vence | Última foto (aprox.) |
| --- | --- | --- |
| +2 h | 09:45 | 11:00 |
| +4 h | 11:45 | 13:00 |
| +6 h | 13:45 | 15:00 |
| +8 h | 15:45 | 17:00 |
| +10 h | 17:45 | 19:00 |
| +12 h | 19:45 | 21:00 |

Entre rondas quedan ~45 min libres.

### 5. Respaldo en cada descanso

Desde WSL, en la raíz del repositorio (la carpeta `data/cache/` no se versiona):

```bash
rsync -av mangopi:capturas/lote1-AAAAMMDD/ data/cache/incoming/AAAAMMDD/lote1-AAAAMMDD/
```

Hazlo solo durante la cuenta regresiva, no mientras se captura (en la Zero 2 W el cifrado de SSH compite por
CPU con la cámara).

### 6. Cierre

```bash
python3 ~/captura_v3/captura.py estado          # todo en ✓
ls ~/capturas/lote1-AAAAMMDD/news/meta | wc -l   # 800 si no se omitió ninguna cara
```

Haz el último `rsync` y anota la temperatura final del cuarto.

## Si algo falla

| Síntoma | Qué hacer |
| --- | --- |
| `luz apagada?` | Despierta la batería (o revisa la fuente) y presiona Enter para repetir la cara. |
| `la NoIR quedó trabada` | `sudo reboot` y vuelve a correr el mismo comando; el estado se conserva. |
| Se cortó el SSH | `ssh mangopi` y `tmux attach -t lote1`. Sin tmux: vuelve a correr el mismo comando (solo se pierde la cara en curso). |
| Una ronda salió tarde | Captúrala igual (queda marcada fuera de tolerancia) o omítela con `s`. |
| `la cámara RGB no está respetando la exposición manual` | Desconecta y reconecta la cámara (o reinicia la Pi) y repite `jornada`. |
| La NIR trae una fila oscura parcial | Es el contacto CSI marginal y el QC no lo detecta: revisa las NIR al respaldarlas y anota las caras afectadas. |
