# Captura dual RGB/NIR con protocolo de golpes (Raspberry Pi)

`captura.py` guía una jornada completa de captura: fotos sanas, impactos (con la foto +0 h de cada mango justo
después de golpearlo) y rondas cronometradas a +2, +4, +6, +8, +10 y +12 h, con el menor trabajo manual posible.
El operador coloca y retira cada mango a mano (no hay motor); el programa se encarga de las luces, las dos
cámaras, el control de calidad (QC), los metadatos, la subida a OCI y de decir cuándo toca cada mango.

Reemplaza a `docs/findings/2026-10-18/capture/captura_dual.py` (una foto por ejecución). Los nombres de
archivo no cambian, porque el pipeline empareja por la expresión `(\w+)_(rgb|nir)_(\d+)\.jpg`.

## Protocolo

- **125 mangos**, en **5 lotes de 25**, son la unidad de agrupación (GroupKFold). Cada uno se fotografía en
  **4 caras** y en **8 momentos**: `sano` (antes del impacto) y **7 momentos con daño**: justo después del
  impacto (t = 0) y a +2, +4, +6, +8, +10 y +12 h.
  Total: 500 pares sanos y 3500 pares con daño (RGB + NIR), es decir, 4000 pares.
- Se trabaja en **jornadas**: un lote de 25 mangos por jornada. En una jornada el operador hace tres tandas,
  siempre en el mismo orden y al mismo ritmo (un mango cada 3 min):
  1. fotos sanas de todos los mangos;
  2. impacto en las 4 caras de cada mango y, en cuanto está golpeado, su foto **+0 h** (cada mango son
     4 impactos + 4 pares RGB/NIR de ~5-8 s con su QC, así que la tanda dura lo mismo que una ronda);
  3. una ronda de fotos por cada intervalo restante (+2 … +12 h).

  Así cada mango se fotografía a *k* horas de **su** impacto (tolerancia ±10 min). La foto +0 h no puede
  esperar a una ronda aparte: al terminar la tanda de impactos habría pasado cerca de una hora y saldría
  fuera de tolerancia.
- **Impacto:** bola de acero de 66 g soltada desde 50 cm por un tubo de PVC sobre la fruta apoyada en
  espuma (energía ≈ 0,324 J).
- **Zona de impacto:** para evitar el sesgo de que el daño siempre quede al centro, la zona de cada cara
  se sortea entre `centro`, `superior`, `inferior`, `izquierda` y `derecha`. La semilla queda guardada en
  `jornada.json` y las zonas son reproducibles.
- Los intervalos son configurables (`--intervalos`; por defecto `0,2,4,6,8,10,12`). Una jornada creada sin el
  0 (el protocolo anterior, rondas desde +2 h) sigue funcionando igual: `impactos` solo registra horas y no usa
  las cámaras.

## Archivos

| Archivo | Contenido |
| --- | --- |
| `captura.py` | Interfaz de línea de comandos y arranque. |
| `flujo.py` | Lógica de cada subcomando (jornada, sanas, impactos, ronda, estado, subir, enfoque). |
| `protocolo.py` | Calendario, tolerancias, ritmo y sorteo de zonas (lógica pura). |
| `estado.py` | `jornada.json`: escritura atómica y reanudación. |
| `calidad.py` | QC contra la referencia de la bandeja vacía. |
| `balance.py` | Calibración del balance de blancos sobre la tarjeta gris. |
| `enfoque.py` | Nitidez en vivo (varianza del Laplaciano) para enfocar a mano. |
| `camaras.py` | Constructores de comandos y parseo de `v4l2-ctl`, `rpicam`, `vcgencmd`. |
| `hardware.py` | Luces (GPIO), cámara RGB, cámara NIR, salud; y un simulador. |
| `metadatos.py` | Esquema "3.0" de los metadatos. |
| `subida.py` | URL pre-autenticada, cola persistente y reintentos. |
| `configuracion.py`, `consola.py`, `marcas_tiempo.py`, `vista.py` | Utilidades. |

Las pruebas están en `tests/capture/` (no se despliegan a la Pi).

## Instalación en la Pi

Requisitos (ya presentes en la Pi Zero 2 W con Bookworm 32 bits y Python 3.11): `numpy`, `Pillow`,
`requests`, `gpiozero`, `ffmpeg`, `v4l2-ctl` y `rpicam-apps`. No hace falta OpenCV ni instalar nada con `apt`.
Para verificarlo:

```bash
ssh mangopi 'python3 -c "import numpy, PIL, requests, gpiozero; print(\"ok\")"; which ffmpeg v4l2-ctl rpicam-still'
```

Copiar los módulos a una carpeta propia (sin tocar los scripts existentes del home) desde lo commiteado.
`core.autocrlf=false` es obligatorio: con `true`, tanto la copia de trabajo como `git archive` traen CRLF.

```bash
ssh mangopi 'mkdir -p ~/captura_v3'
git -c core.autocrlf=false archive HEAD capture | ssh mangopi 'tar -x -C ~/captura_v3 --strip-components=1'
ssh mangopi 'python3 ~/captura_v3/captura.py --help'
```

### pipewire y wireplumber (deben estar enmascarados)

`pipewire` y `wireplumber` retenían abiertos todos los nodos CSI y de medios (`/dev/media*`, `/dev/video*`) y bloqueaban la
captura. En la Pi quedaron detenidos y **enmascarados** (unidades de usuario). Para reproducirlo en otra Pi:

```bash
export XDG_RUNTIME_DIR=/run/user/$(id -u)      # necesario si se hace por ssh
systemctl --user stop pipewire.socket pipewire-pulse.socket pipewire.service pipewire-pulse.service wireplumber.service
systemctl --user mask pipewire.socket pipewire-pulse.socket pipewire.service pipewire-pulse.service wireplumber.service
```

`mask` crea enlaces a `/dev/null` en `~/.config/systemd/user/`, así que ni siquiera la activación por socket los vuelve a
levantar. Para comprobarlo: `systemctl --user is-enabled pipewire.service wireplumber.service` debe decir `masked`.

**Para revertirlo** (si hace falta audio en esa Pi):

```bash
systemctl --user unmask pipewire.socket pipewire-pulse.socket pipewire.service pipewire-pulse.service wireplumber.service
systemctl --user start pipewire.socket pipewire-pulse.socket wireplumber.service
```

Al arrancar cualquier comando con cámara, el programa usa `fuser` para avisar (sin abortar) si algún otro proceso tiene abiertos
`/dev/media*` o los nodos de video. Si `fuser` no está instalado, no se verifica nada y sigue normalmente. A mano:
`fuser -v /dev/media* /dev/video*`.

### URL pre-autenticada de OCI

La URL es una credencial: **nunca va en el código, ni se imprime, ni se guarda en logs o metadatos**. Se lee solo
de la variable de entorno `OCI_PAR_URL` o del archivo `~/.config/captura/oci_par_url` (con permisos 600; si no,
el programa avisa). Debe terminar en `/o/`.

```bash
mkdir -p ~/.config/captura
printf '%s\n' 'https://.../o/' > ~/.config/captura/oci_par_url     # pegar la URL real
chmod 600 ~/.config/captura/oci_par_url
```

### Configuración de las cámaras

`~/capturas/config_camaras.json` se crea con valores por defecto en la primera ejecución. Si falta alguna clave
(por una versión nueva) se completa sola sin tocar lo que ya ajustaste. **Cada captura copia los ajustes que usó en
sus metadatos.**

| Clave | Por defecto | Notas |
| --- | --- | --- |
| `rgb.device_by_id` | `/dev/v4l/by-id/usb-Arducam_..._SN0001-video-index0` | Siempre por `by-id`: el número `/dev/videoN` cambia entre arranques. |
| `rgb.resolucion` | `3264x2448` | Solo existe en MJPG. Se guardan los bytes JPEG de la cámara sin recodificar. |
| `rgb.controles.exposure_time_absolute` | `104` | 10,4 ms (unidades de 100 µs). Techo medido el 2026-10-21 con el mango a la altura nueva, sin reflector: 0,1 % del mango recortado (con 21,2 ms, 19 % del lado iluminado). El lado en sombra sigue oscuro: hace falta un reflector difuso. |
| `rgb.controles.gain` | `0` | 0-100. |
| `rgb.controles.white_balance_temperature` | `6500` | 5000 daba un tinte cian fuerte. `jornada` lo afina sobre la tarjeta gris, medida a la mitad de la exposición de captura para que no se recorte. |
| `rgb.controles.sharpness` | `3` | Valor de fábrica (con 0 la imagen se veía blanda y granulosa). |
| `rgb.controles.focus_absolute` | `300` | **No hace nada** (el foco es manual); se registra igualmente. |
| `rgb.controles.backlight_compensation`, `rgb.controles.power_line_frequency` | `0`, `0` | **Obligatorios en 0** (ver más abajo). Se validan al cargar la configuración. |
| `rgb.controles` (resto) | modo manual, sin balance automático, sin autofoco | Todos se fijan y se leen de vuelta. |
| `rgb.mjpeg2jpeg` | `auto` | Ver "Cadena de captura RGB". `siempre` o `nunca` fuerzan el paso. |
| `rgb.cuadros_descartados` | `5` | Cuadros que se descartan tras abrir la cámara (ver `descarte`). |
| `nir.shutter_us`, `nir.gain` | `6000`, `1.0` | Exposición manual del NIR. Barrido del 2026-10-21 (3/6/10 ms con ganancia 1): 6 ms deja 0,1 % del mango recortado; 20000 µs con ganancia 2 quemaba el 66 %. |
| `nir.denoise`, `nir.awbgains` | `cdn_off`, `1,1` | Conserva la textura fina del daño. |
| `nir.timeout_s` | `25` | Tiempo máximo de `rpicam-still`. Al agotarse: SIGTERM y la NoIR se considera trabada (ver más abajo). |
| `nir.raw` | `false` | `true` guarda además el DNG (~10 MB por cara) en `news/nir_raw/`. |
| `nir.width/height/mode` | `3280x2464`, `3280:2464` | Sensor completo. El modo 1920x1080 es un **recorte central**: no usarlo. |
| `luces.rgb_gpio`, `luces.nir_gpio`, `luces.espera_s` | `27`, `22`, `0.4` | Luz blanca: BCM 27 (pin físico 13). NIR 850 nm: BCM 22 (pin 15). Relés verificados. |
| `qc.*` | ver más abajo | Umbrales del control de calidad. |

### Cadena de captura RGB

1. Se leen los controles V4L2 y se fijan **de uno en uno** (los modos automáticos primero) solo los que difieren,
   salvo la exposición, que se fija siempre con el ciclo descrito en "Trampa del firmware de exposición".
   `v4l2-ctl` rechaza (exit 255) una llamada con varios controles si un modo automático sigue activo. Un control
   que falla se registra en `rgb.controles_fallidos` y avisa, pero no aborta; la excepción son los controles
   obligatorios (`backlight_compensation` y `power_line_frequency`): si no se pueden fijar, o si al leerlos de vuelta
   no valen 0, es un **error** y la foto se descarta.
2. `v4l2-ctl --stream-mmap=3 --stream-skip=N --stream-count=1` guarda el cuadro MJPEG crudo en un temporal
   `.mjpg.tmp`. **No se usa ffmpeg para abrir la cámara:** a 3264x2448 su demultiplexor v4l2 pide ~32 buffers de
   16 MB y el kernel lo mata por falta de memoria en la Zero 2 W (425 MB).
3. Se recorta el relleno posterior al marcador EOI. Si el cuadro no trae tablas de Huffman, se empaqueta con
   `ffmpeg -i <archivo>.mjpg -c:v copy -bsf:v mjpeg2jpeg <salida>.jpg` (entrada de **archivo**, sin recodificar).
   Con `rgb.mjpeg2jpeg = auto` ese paso solo se aplica cuando faltan las tablas; el metadato
   `rgb.tablas_huffman_en_origen` dice qué ocurrió.
4. Después de la captura se vuelcan **todos** los controles (`v4l2-ctl --list-ctrls`) a los metadatos.

### Trampa del firmware de exposición (Arducam)

Después de un ciclo de alimentación, escribir `auto_exposure=1` cuando el control ya lee 1 **no** cambia el modo interno:
la cámara conserva su propia exposición mientras informa "manual" (0,5, 2 y 6 ms daban el mismo brillo). Por eso, antes de
**cada** fijación de exposición, aunque lea 1, el programa escribe en este orden y de a un control por llamada:

1. `auto_exposure=3` (automático)
2. pausa de ~0,2 s
3. `auto_exposure=1` (manual)
4. `exposure_dynamic_framerate=0`
5. `exposure_time_absolute=<E>`

Verificado en la Pi: tras el ciclo, 1,2 ms dio media 22, 4,5 ms dio 62, y 0,5 contra 6 ms dio 13 contra 75.

**Autoverificación en `jornada`:** al terminar la calibración, con la luz blanca encendida, se toman dos cuadros rápidos
(1600x1200) a E y a E/3 y se exige que el brillo medio a E sea al menos 1,5 veces el de E/3. Si no, se aborta con
`la cámara RGB no está respetando la exposición manual: desconectá y reconectá la alimentación o reiniciá la Pi`, se
invalida la calibración (se hizo con una exposición que no era la pedida) y `sanas`, `ronda` e `impactos` (si la
jornada incluye la ronda +0 h) se niegan a correr hasta repetir `jornada`. La razón queda en `jornada.json` (`autoverificacion_exposicion`).

`backlight_compensation=0` y `power_line_frequency=0` son obligatorios: de fábrica `backlight_compensation=1` es un realce
adaptativo de brillo que anula la exposición fija. Se fijan en cada sesión y un valor distinto al leerlo de vuelta es un error.

### NoIR trabada (no dejar `rpicam` colgado)

Un `rpicam-still` colgado o matado a la fuerza deja la canalización CSI en un estado donde **toda** captura posterior falla
con `Camera frontend has timed out`; solo se arregla reiniciando la Pi (no era el cable). Por eso:

- `rpicam-still` se ejecuta con un timeout de 25 s (`nir.timeout_s`). Al agotarse se envía SIGTERM, se esperan hasta 5 s y
  SIGKILL queda como último recurso.
- Tras un timeout (o un `Camera frontend has timed out`) la sesión se **aborta** con
  `la NoIR quedó trabada: reiniciá la Pi (sudo reboot) y retomá con el mismo comando; el estado se conserva`
  (código de salida 4). No se reintenta ninguna captura NIR en el mismo proceso. Tras reiniciar, vuelve a ejecutar el mismo comando.
- `jornada` hace antes de empezar una toma NIR de prueba (640x480, `--immediate`, 10 s) para detectar una cámara trabada antes
  de calibrar.

## Procedimiento diario

Material: bandeja con espuma, tubo de PVC y bola, tarjeta gris (o hoja blanca sin saturar), tablero ArUco,
batería USB despierta (ver "Batería USB").

### 0. Antes de empezar

- Conecta la Pi, espera a que sincronice la hora (el programa lo verifica) y despierta la batería USB.
- Si es la primera vez con este montaje: enfoca a mano ambos lentes con `captura.py enfoque` (ver más abajo).

### 1. Jornada

```bash
python3 ~/captura_v3/captura.py jornada --id 2026-10-22 --lote "Kent A" --mangos 1-25 --temperatura-c 24.5
```

Antes de nada avisa si otro proceso usa las cámaras (`fuser`), verifica ambas cámaras (`rpicam-hello --list-cameras`,
una toma NIR de prueba de 640x480 con `--immediate` y el dispositivo `by-id` de la RGB) y la sincronización del reloj;
aborta con un mensaje claro si algo falta o si la NoIR está trabada. Crea `~/capturas/<id>/jornada.json`, sortea las zonas y guía
la calibración (todo va a `calib/` y nunca entra al dataset):

1. **Tarjeta gris** ("Coloca la tarjeta gris ... y presiona Enter"). Se muestra el recuadro de medición (vista
   previa en `~/capturas/<id>/tmp/wb_recuadro.jpg`; el recuadro rojo debe quedar dentro de la tarjeta). Se barre
   `white_balance_temperature` de 2800 a 6500 K y se elige el valor que minimiza |R−G|+|B−G|; se guarda en
   `config_camaras.json`. Si la tarjeta sale recortada (saturada) se pide bajar la exposición. Después se toma la
   foto de la tarjeta gris con el balance ya calibrado (sirve para corregir viñeteo y gradiente de luz).
2. **Bandeja vacía** ("Coloca la bandeja VACIA ..."): guarda la referencia de brillo que usa el QC.
3. **Tablero ArUco** ("Coloca el tablero ArUco ..."): para recalcular la homografía RGB-NIR.

El balance se calibra *antes* de la bandeja vacía porque cambia los valores que usa el QC. Se puede omitir la
tarjeta o el tablero con `s`.

Al final se hace la **autoverificación de exposición** (ver "Trampa del firmware de exposición"): dos cuadros rápidos a E y E/3.

### 2. Sanas

```bash
python3 ~/captura_v3/captura.py sanas
```

Para cada mango y cara: `Mango 7 · cara 2 → coloca y presiona Enter`. Captura RGB y NIR, aplica el QC y muestra una
línea con el resultado. Marca el avance después de cada cara, así que reanuda donde quedó. `q` sale.

### 3. Impactos

```bash
python3 ~/captura_v3/captura.py impactos
```

Para cada mango muestra las 4 zonas sorteadas y para cada cara pide
`Cara N · zona <zona> → presiona Enter AL SOLTAR la bola`; registra la hora exacta de **cada cara**. Muestra el
ritmo objetivo (un mango cada 3 min) y avisa si te desfasas más de la mitad de la tolerancia.

Con la ronda +0 h en la jornada (el valor por defecto), en cuanto las 4 caras del mango están golpeadas el
programa lo fotografía de inmediato, antes de pasar al mango siguiente:
`Mango 7 · +0 h · cara 1 → coloca y presiona Enter` (RGB y NIR, con QC y metadatos, igual que en `sanas`).
Por eso `impactos` ahora **necesita las cámaras**: hace las mismas comprobaciones que `ronda`
(autoverificación de exposición hecha en `jornada`, NoIR presente, aviso de procesos ajenos), apaga las luces al
salir y sube lo capturado. Si la jornada no tiene el 0 en `--intervalos`, nada de esto ocurre.

- **Reanudar:** vuelve a ejecutar `impactos`. Primero se piden los impactos que faltan; un mango con los
  impactos completos y fotos +0 h pendientes se fotografía antes de seguir con el siguiente. Si su ventana +0 h ya
  pasó (más de la tolerancia desde su primer impacto) avisa y ofrece
  `Enter = capturar igual · s = omitir esta ronda del mango · q = salir`; lo capturado tarde queda marcado fuera de
  tolerancia, y lo omitido no se vuelve a pedir.
- Una cara omitida con `s` tras un error de QC queda pendiente: `impactos` la vuelve a ofrecer, y `ronda`
  también (ya atrasada). `ronda` nunca repite las caras +0 h que `impactos` ya tomó.
- `q` sale en cualquier momento; el avance queda guardado.
- Al terminar indica el siguiente paso y cuándo vence la primera ronda +2 h.

### 4. Rondas

```bash
python3 ~/captura_v3/captura.py ronda
```

Calcula qué (mango, ronda) vence primero a partir de las horas de impacto y lo muestra:
`Mango 7 · ronda +4 h · vence 14:32 (faltan 3 min)`.

- Si falta **más que la tolerancia**: ofrece esperar con cuenta regresiva (o seguir ahora con `--forzar`).
- Si ya pasó la ventana: avisa del atraso, permite capturar igual (queda marcada fuera de tolerancia) u omitir (`s`).
- Captura las 4 caras y guarda, por cara, las horas reales desde **ese** impacto y la desviación en minutos.
- Continúa con el siguiente vencimiento hasta que salgas con `q` o no quede nada pendiente.
- La ronda +0 h ya la tomó `impactos`; solo aparece aquí si esa tanda se interrumpió y quedaron caras sin foto.

### Horario de ejemplo (25 mangos, 3 min por mango)

| Tanda | Horario |
| --- | --- |
| Sanas | 07:00 - 07:40 |
| Impactos + fotos +0 h | 07:45 - 09:00 |
| Ronda +2 h | 09:45 (hasta 11:00) |
| Ronda +4 h | 11:45 |
| Ronda +6 h | 13:45 |
| Ronda +8 h | 15:45 |
| Ronda +10 h | 17:45 |
| Ronda +12 h | 19:45 (hasta 21:00) |

Las sanas no dependen del ritmo (~20 s por cara con la manipulación). La tanda de impactos con sus fotos +0 h y
cada ronda duran ~75 min (25 mangos × 3 min) y los vencimientos están separados 2 h: quedan ~45 min de descanso
entre rondas. La lista de pasos del lote 1 está en [`docs/capture/guia-lote-1.md`](../docs/capture/guia-lote-1.md).

### Estado y subida

```bash
python3 ~/captura_v3/captura.py estado    # tabla mangos x momentos, siguiente vencimiento y descanso
python3 ~/captura_v3/captura.py subir     # sube lo pendiente con reintentos
```

`estado` marca ✓ hecho, ⏳ hora de vencimiento, ✗ perdido o fuera de tolerancia, `·` parcial (caras hechas/total).
La ronda +0 h es una columna más (`+0 h`), a continuación de `imp`.

La subida corre sola en un hilo mientras capturas, desde la cola persistente `pendientes.jsonl`, así que la captura
nunca espera a la red. Con `--no-subir` no se encola nada; `captura.py subir` encuentra en disco lo que falte.
Disposición del bucket: `news/rgb/`, `news/nir/`, `news/meta/`, `news/nir_raw/` (si hay RAW) y `calib/`.

### Enfoque (manual)

El foco de **ambas** cámaras es manual (en la RGB, `focus_absolute` existe pero no hace nada: la nitidez medida
quedó plana entre 20 y 900). `enfoque` muestra la nitidez en vivo mientras giras el lente a mano:

```bash
python3 ~/captura_v3/captura.py enfoque --camara rgb     # luz blanca encendida
python3 ~/captura_v3/captura.py enfoque --camara nir     # luz NIR de 850 nm encendida (el foco cambia a 850 nm)
```

Usa tomas rápidas (RGB 1600x1200, NIR en el modo binned 1640x1232 que sigue viendo el sensor completo), imprime la
varianza del Laplaciano del centro con el máximo visto y termina con Ctrl-C. Las luces se apagan siempre al salir.
Pon un objetivo con detalle fino (texto impreso) a la altura de la fruta.

### Descarte de cuadros (avanzado)

`captura.py descarte --valores 0,2,5,10,30` mide cuánto tarda la captura RGB según los cuadros descartados y el
brillo del borde resultante. Elige el menor valor cuyo brillo ya coincide con el de valores mayores y fíjalo en
`rgb.cuadros_descartados`.

## Batería USB

Las luces se alimentan de una batería USB que **se duerme** si la carga está inactiva (~30 s). Cuando eso pasa la luz
no enciende y la foto sale oscura: el QC lo detecta como `luz apagada?` (ver más abajo). Antes de capturar tras una
pausa, presiona el botón de la batería. Si el mensaje aparece, despierta la batería y pulsa Enter para repetir la cara.

## Después de una caída o un reinicio

Nada se pierde salvo la cara en curso: `jornada.json` se escribe de forma atómica (archivo temporal + renombrado)
después de cada cara y la cola de subida es un registro de eventos que tolera una línea cortada. Para continuar:

1. Verifica la hora (`timedatectl`) y despierta la batería.
2. Vuelve a ejecutar el mismo comando (`sanas`, `impactos` o `ronda`): sigue donde quedó. La jornada activa se
   recuerda en `~/capturas/jornada_activa`; usa `--id` para elegir otra.
3. Si cambió el nombre del dispositivo `/dev/videoN` no importa: la RGB se abre por `by-id`.
4. `python3 ~/captura_v3/captura.py subir` completa la subida de lo que quedó en cola.

Si la jornada se interrumpe durante la calibración, vuelve a ejecutar `jornada` con el mismo `--id`: reanuda los
pasos pendientes (no repite el barrido de balance ya hecho).

## Control de calidad (QC)

Cada captura se compara con la **bandeja vacía** de la jornada usando el promedio de una región de **borde** (el
marco exterior de 10 %, lejos de donde va la fruta), por modalidad. Se descarta la métrica antigua de saturación
del centro: incluía una hoja blanca que siempre se recorta (35 % "saturado" incluso en fotos buenas) y no detectó el
cambio de ganancia del 2026-10-18.

| Mensaje | Significado | Qué hacer |
| --- | --- | --- |
| `la iluminación o la cámara cambió` | El borde se desvía más de 15 % de la referencia. | Revisa la luz, la ganancia y el balance; la foto se guarda con la alerta. |
| `luz apagada?` (**ERROR**) | El borde es menor que 30 % de la referencia. Casi siempre es la batería USB dormida. | Despierta la batería y pulsa Enter para repetir; `s` omite la cara. En `--auto` se aborta. La toma mala va a `descartes/` y no se sube. |
| `sin referencia de bandeja vacia` | No se hizo la bandeja vacía en `jornada`. | Repite `jornada` con la bandeja vacía. |
| `controles V4L2 distintos de lo pedido: gain, ...` | La cámara no quedó con los valores configurados. | Revisa la cámara/USB. `focus_absolute` no genera alertas. |
| `no se pudieron fijar los controles V4L2: ...` | `v4l2-ctl` rechazó un control. | Revisa `rgb.controles_fallidos` en los metadatos. |
| `la cámara RGB no está respetando la exposición manual: ...` | La autoverificación de `jornada` no vio bajar el brillo al dividir la exposición por 3. | Desconecta y reconecta la alimentación de la cámara o reinicia la Pi y repite `jornada`. |
| `la NoIR quedó trabada: ...` | `rpicam-still` agotó el tiempo o falló con `Camera frontend has timed out`. | Reinicia la Pi (`sudo reboot`) y retoma con el mismo comando; el estado se conserva. |
| `Otro proceso está usando las cámaras ...` | `fuser` ve otro proceso con abiertos los nodos de video o medios. | Detenlo; ver "pipewire y wireplumber". Solo avisa. |
| `rgb: el control obligatorio backlight_compensation ...` | `backlight_compensation` o `power_line_frequency` no quedaron en 0. | Error: revisa la cámara. La foto se descarta. |
| `fuera de tolerancia horaria (+12.3 min)` | La foto salió fuera de ±tolerancia respecto de la ronda nominal. | Informativo: queda en los metadatos para filtrar. |
| `% recortado` (en la línea del resultado) | Porcentaje de píxeles con algún canal ≥ 250 en la zona central. | Solo informativo. |

Umbrales (`qc.desviacion_max` = 0,15; `qc.luz_apagada_ratio` = 0,30; `qc.borde_frac` = 0,10) y la zona central
(`qc.centro_frac`) están en `config_camaras.json`. El QC decodifica con el modo `draft` de Pillow (JPEG a 1/4 de
tamaño), porque decodificar 8 MP completos en la Zero tarda unos 2-3 s.

Un borde casi negro en la referencia hace inestable la comparación relativa; por eso se ignoran diferencias menores
de 3 niveles (de 255). Si el fondo de la bandeja es muy oscuro, conviene que la hoja blanca llegue al marco de la
imagen o aumentar `qc.borde_frac`.

## Memoria

La Zero 2 W tiene 425 MB y decodificar varios cuadros de 8 MP a arreglos `float` hizo que el sistema matara procesos. El QC
usa el modo `draft` de Pillow (JPEG a 1/4 de tamaño), analiza **un cuadro a la vez** y no convierte a `float` el cuadro completo
(la nitidez recorta el recuadro central antes de convertir). Hay pruebas que vigilan el pico de memoria con un cuadro de 8 MP.

## Códigos de salida

| Código | Significado |
| --- | --- |
| 0 | Terminó bien (o el operador salió con `q`). |
| 2 | No se puede empezar: cámara o reloj, jornada inexistente, configuración inválida, error de cámara, exposición no respetada. |
| 3 | Captura abortada por el QC en modo `--auto` (por ejemplo `luz apagada?`). |
| 4 | NoIR trabada: reinicia la Pi. |
| 130 | Ctrl-C; el avance queda guardado. |

## Metadatos (esquema "3.0")

Un archivo `mango_meta_<ts>.json` por cara, con el mismo timestamp que `mango_rgb_<ts>.jpg` y `mango_nir_<ts>.jpg`.
El timestamp es único por cara (si el segundo ya se usó, avanza al siguiente libre).

| Campo | Contenido |
| --- | --- |
| `script_version`, `schema` | `"3.0.0"`, `"3.0"`. |
| `timestamp`, `fecha_local` | Segundos Unix y hora local con zona. |
| `jornada` | `{id, lote, temperatura_c}`. |
| `mango_id`, `cara` | Mango (1..125) y cara (1..4). |
| `fase`, `ronda_h` | `"sano"` o `"post_impacto"`; ronda nominal en horas (`null` en sano; `0` = foto justo después del impacto). |
| `impacto` | `null` en sano. Si no: `{zona, hora_iso, horas_desde_impacto, masa_bola_g, altura_cm, energia_j, desviacion_min, en_tolerancia}`, con la hora **de esa cara** y las horas medidas al empezar su captura. |
| `rgb` | `{device_by_id, resolucion, metodo, comando, tablas_huffman_en_origen, comando_mjpeg2jpeg, controles (volcado completo), controles_pedidos, controles_aplicados, controles_fallidos, diferencias, config}`. |
| `nir` | `{comando, rpicam_metadata (ExposureTime, AnalogueGain, SensorTemperature, Lux, ...), config, raw_dng}`. |
| `calidad` | `{rgb, nir, alertas}`; cada modalidad: `{borde_media, pct_recortado_centro, referencia_borde_media, ratio, desviacion_rel, estado, alertas, error}`. |
| `salud` | `{throttled: {raw, valor, flags}, camaras: {rgb_device, nir_presente}}` (`vcgencmd get_throttled` por captura). |
| `archivos` | `{rgb, nir, raw}`, cada uno `{sha256, bytes}`. |
| `subida` | `{estado: "pendiente"}`: el JSON se sube junto con las fotos y no puede saber el resultado; el avance real está en `pendientes.jsonl`. |
| `tiempos_s` | Segundos de RGB, NIR, QC y total de la cara. |

Los metadatos de calibración (`calib-<tipo>-meta-<ts>.json`) tienen `tipo: "calibracion"` y no llevan mango ni fase.
`calib-wb-meta-<ts>.json` guarda la tabla del barrido de balance de blancos.

## Carpetas

```text
~/capturas/config_camaras.json
~/capturas/jornada_activa
~/capturas/<id>/jornada.json         estado de la jornada
~/capturas/<id>/pendientes.jsonl     cola de subida
~/capturas/<id>/news/{rgb,nir,meta,nir_raw}/   dataset (se sube)
~/capturas/<id>/calib/               calibraciones (se suben, fuera del dataset)
~/capturas/<id>/descartes/           tomas rechazadas por el QC (no se suben)
```

## Pruebas

```bash
.venv/bin/python -m pytest tests/capture -q
```

Cubren el calendario y las tolerancias, el sorteo de zonas, el estado atómico, el QC con arreglos sintéticos, los
constructores de comandos, el parseo de una salida real de `v4l2-ctl`, la resolución por `by-id`, la URL pre-autenticada
(nunca expuesta), la cola de subida con HTTP simulado y los flujos completos con hardware simulado
(`captura.py --simular`, opción oculta).

Prueba rápida en la Pi con el reloj acelerado (1 "hora" = 0,3 min). Despierta la batería USB justo antes de cada paso
(se duerme a los ~30 s de inactividad) y ejecuta de a un subcomando:

```bash
cd ~/captura_v3
python3 captura.py --auto descarte --valores 0,2,5,10,30                       # ~1 min
python3 captura.py --auto enfoque --camara rgb --max-lecturas 5                # ~10 s (opcional; también --camara nir)
python3 captura.py --auto jornada --id PRUEBA_20261021 --lote prueba --mangos 1-2 --caras 1,2 \
    --intervalos 1,2 --unidad-min 0.3 --slot-min 0.5 --tolerancia-min 0.1 --no-subir --semilla 1   # ~2-3 min
python3 captura.py --auto sanas                                                # ~1 min
python3 captura.py --auto impactos                                             # segundos
python3 captura.py --auto ronda                                                # ~2 min
python3 captura.py estado
rm -rf ~/capturas/PRUEBA_20261021                                              # al terminar
```

Para ensayar también la foto +0 h agrega el 0 a los intervalos (`--intervalos 0,1,2`): `impactos` pasa a usar las
cámaras y esas fotos saldrán marcadas fuera de la tolerancia reducida del ensayo (es informativo).

Si el QC informa `luz apagada?` en esta prueba es la batería dormida: es una detección válida, no un error del programa.
