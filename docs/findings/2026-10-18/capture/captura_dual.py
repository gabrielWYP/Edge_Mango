"""Captura dual RGB/NIR para la tesis (Raspberry Pi).

Cambios respecto de la version anterior:
  * Exposicion manual opcional en ambas camaras, para que todas las fotos y
    todos los lotes sean comparables (sin auto-exposicion ni balance de blancos
    automatico que cambien entre tomas).
  * NIR con rpicam-still: --awbgains 1,1 fijos, denoise configurable y RAW DNG
    opcional (10 bits en lugar de 8 bits JPEG).
  * Metadatos en un JSON aparte (ID del mango, cara, sesion, parametros de
    camara). Los nombres de imagen NO cambian (mango_rgb_<ts>.jpg /
    mango_nir_<ts>.jpg) porque el pipeline empareja RGB y NIR con
    `(\\w+)_(rgb|nir)_(\\d+)\\.jpg` y exige el timestamp al final.
  * Modo calibracion (--calib gris|aruco) que sube a calib/, fuera de news/,
    para que nunca entre al dataset.

Ejemplos:
  # Calibracion al inicio de cada sesion (hoja gris uniforme y tablero ArUco)
  python3 captura_dual.py --calib gris  --nir-shutter 20000 --nir-gain 1 --rgb-exposure 150 --rgb-wb 6500
  python3 captura_dual.py --calib aruco --nir-shutter 20000 --nir-gain 1 --rgb-exposure 150 --rgb-wb 6500

  # Mango 12, cara 3, impactado en la cara 3 (zona superior) a las 10:15
  python3 captura_dual.py --mango-id 12 --cara 3 --impacto si --cara-impacto 3 --zona-impacto superior \
      --hora-impacto 10:15 --nir-shutter 20000 --nir-gain 1 --rgb-exposure 150 --rgb-wb 6500

  # Mismo mango, cara 1 (no impactada); el JSON marca que en esta foto no se espera dano
  python3 captura_dual.py --mango-id 12 --cara 1 --impacto si --cara-impacto 3 --zona-impacto superior \
      --hora-impacto 10:15 --nir-shutter 20000 --nir-gain 1 --rgb-exposure 150 --rgb-wb 6500

  # Mango control sin impacto, todo automatico como antes (NIR con EV)
  python3 captura_dual.py --mango-id 13 --cara 2 --impacto no --ev -3

Parametros obligatorios:
  * Siempre (salvo --calib): --mango-id, --cara, --impacto si|no.
  * Si --impacto si: --cara-impacto (una o varias caras), --zona-impacto y exactamente
    uno de --hora-impacto o --minutos-desde-impacto.
  * Opcionales con valor por defecto: --n-impactos (uno por cara impactada),
    --masa-bola-g 66, --altura-cm 50, --ev -3, --sesion (fecha de hoy).

Como elegir los valores fijos (una vez por montaje de luces):
  1. NIR: subir --nir-shutter (microsegundos) con --nir-gain bajo (1-2) hasta que la
     zona mas clara del mango quede cerca de 230/255 sin saturar. Con difusor o
     filtro hace falta mas exposicion que hoy.
  2. RGB: `v4l2-ctl -d /dev/video1 --list-ctrls` muestra los rangos validos de
     exposicion y temperatura de color. Elegir valores con el mango bien expuesto
     y la hoja gris neutra.
  3. Usar SIEMPRE los mismos valores en toda la sesion (idealmente en todas).
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import time

import requests
from gpiozero import OutputDevice

SCRIPT_VERSION = "2.2"

# --- CONFIGURACION ---
OUTPUT_DIR = "fotos_tesis"
# La URL pre-autenticada de OCI nunca va en el codigo: solo se lee del entorno.
try:
    OCI_PAR_URL = os.environ["OCI_PAR_URL"]
except KeyError:
    raise SystemExit(
        "Falta la variable de entorno OCI_PAR_URL (URL pre-autenticada del bucket de OCI, terminada en '/o/'). "
        "Definela antes de ejecutar, por ejemplo: export OCI_PAR_URL='https://.../o/'"
    )

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- HARDWARE (RELES) ---
rele_rgb = OutputDevice(27, active_high=True, initial_value=False)
rele_nir = OutputDevice(22, active_high=True, initial_value=False)

# Nombres de controles V4L2 segun la version del kernel/driver (nuevo, antiguo).
V4L2_ALIASES = {
    "auto_exposure": ("auto_exposure", "exposure_auto"),
    "exposure": ("exposure_time_absolute", "exposure_absolute"),
    "auto_wb": ("white_balance_automatic", "white_balance_temperature_auto"),
    "wb_temperature": ("white_balance_temperature",),
    "gain": ("gain",),
}
V4L2_EXPOSURE_MANUAL = 1  # valor "Manual Mode" en ambos nombres de auto exposicion

# Zona de la cara donde cae la bola. Variarla entre mangos evita que el modelo
# aprenda "el dano siempre esta en el centro" (sesgo de posicion del protocolo).
ZONAS_IMPACTO = ["centro", "superior", "inferior", "izquierda", "derecha", "otro"]


# --- RGB (USB, Arducam IMX219) ---
def listar_controles_v4l2(device):
    """Devuelve el conjunto de nombres de controles que expone la camara."""
    try:
        out = subprocess.run(["v4l2-ctl", "-d", device, "--list-ctrls"],
                             capture_output=True, text=True, timeout=5, check=True).stdout
    except Exception as e:
        print(f"[RGB] No se pudo listar controles V4L2: {e}")
        return set()
    return set(re.findall(r"^\s*(\w+)\s+0x", out, flags=re.MULTILINE))


def resolver(nombre, disponibles):
    for alias in V4L2_ALIASES[nombre]:
        if alias in disponibles:
            return alias
    return None


def configurar_rgb(device, exposure, wb, gain):
    """Fija exposicion/balance/ganancia si se pidieron. Devuelve lo aplicado y lo que falto."""
    disponibles = listar_controles_v4l2(device)
    pedidos = []
    if exposure is not None:
        pedidos += [("auto_exposure", V4L2_EXPOSURE_MANUAL), ("exposure", exposure)]
    if wb is not None:
        pedidos += [("auto_wb", 0), ("wb_temperature", wb)]
    if gain is not None:
        pedidos += [("gain", gain)]
    aplicados, faltantes = {}, []
    for nombre, valor in pedidos:  # el orden importa: primero apagar el modo automatico
        ctrl = resolver(nombre, disponibles)
        if ctrl is None:
            faltantes.append(nombre)
            continue
        r = subprocess.run(["v4l2-ctl", "-d", device, "-c", f"{ctrl}={valor}"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            aplicados[ctrl] = valor
        else:
            faltantes.append(f"{nombre} ({r.stderr.strip()})")
    if faltantes:
        print(f"[RGB] Controles NO aplicados: {faltantes}")
    return aplicados, faltantes


def leer_controles_rgb(device, nombres):
    """Lee el valor real de los controles despues de la captura (para el JSON)."""
    leidos = {}
    for ctrl in nombres:
        r = subprocess.run(["v4l2-ctl", "-d", device, f"--get-ctrl={ctrl}"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and ":" in r.stdout:
            leidos[ctrl] = r.stdout.split(":", 1)[1].strip()
    return leidos


def tomar_foto_usb(ruta_archivo, device, resolucion, skip):
    print(f"[USB-RGB] Capturando {os.path.basename(ruta_archivo)}...")
    cmd = ["fswebcam", "-d", device, "-r", resolucion, "--palette", "mjpeg",
           "--jpeg", "95", "--no-banner", "--skip", str(skip), ruta_archivo]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=15)
        if os.path.exists(ruta_archivo):
            print("[USB] Foto capturada correctamente.")
            return True
        print(f"[USB] fswebcam no genero el archivo.\n{result.stderr.strip()}")
        return False
    except subprocess.TimeoutExpired:
        print("[USB] Error: fswebcam se colgo (timeout).")
        subprocess.run(["pkill", "-9", "fswebcam"], stderr=subprocess.DEVNULL)
        return False
    except subprocess.CalledProcessError as e:
        print(f"[USB] Error del sistema operativo: {e.stderr.strip()}")
        return False


# --- NIR (CSI, Pi Camera NoIR v2) ---
def comando_nir(ruta_archivo, args):
    cmd = ["rpicam-still", "-o", ruta_archivo,
           "--width", "1280", "--height", "720",
           "--mode", "3280:2464",          # sensor completo: mismo FOV que la camara USB
           "--awbgains", "1,1",            # el balance de blancos no tiene sentido en NIR
           "--denoise", args.nir_denoise,  # cdn_off conserva la textura fina del dano
           "--nopreview"]
    if args.nir_shutter is not None:       # exposicion manual: fotos comparables
        cmd += ["--shutter", str(args.nir_shutter), "--gain", str(args.nir_gain), "-t", "1000"]
    else:                                  # exposicion automatica (comportamiento anterior)
        cmd += ["--ev", str(args.ev), "-t", "2000"]
    if args.nir_raw:
        cmd += ["--raw"]                   # ademas guarda <nombre>.dng (RAW de 10 bits)
    return cmd


def tomar_foto_csi(ruta_archivo, args):
    print(f"[CSI-NIR] Capturando {os.path.basename(ruta_archivo)}...")
    cmd = comando_nir(ruta_archivo, args)
    for binario in ("rpicam-still", "libcamera-still"):
        cmd[0] = binario
        try:
            subprocess.run(cmd, check=True, stderr=subprocess.PIPE, timeout=30)
            print(f"[CSI] Foto capturada ({binario}).")
            return True
        except Exception as e:
            print(f"[CSI] {binario} fallo: {e}")
    return False


# --- REPORTE DE BRILLO (calibracion rapida sin mirar la foto) ---
def reportar_brillo(ruta, etiqueta):
    """Imprime y devuelve percentiles de brillo del 50% central de la imagen.

    Objetivo de calibracion: p99 del centro entre ~200 y ~240 y menos de 1% de
    pixeles saturados (>=250). Requiere Pillow y numpy; si faltan, no hace nada.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        print(f"[{etiqueta}] Reporte de brillo omitido (falta Pillow o numpy).")
        return None
    try:
        img = np.asarray(Image.open(ruta).convert("RGB"), dtype=np.float32)
    except Exception as e:
        print(f"[{etiqueta}] No se pudo leer la imagen para el reporte: {e}")
        return None
    h, w = img.shape[:2]
    centro = img[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    gris = centro.mean(axis=2)  # promedio de canales: no pondera solo el verde
    p1, p50, p99 = (float(np.percentile(gris, q)) for q in (1, 50, 99))
    saturado = float((centro.max(axis=2) >= 250).mean() * 100)
    medias = [round(float(centro[..., i].mean()), 1) for i in range(3)]
    estado = "OK" if 200 <= p99 <= 240 and saturado < 1 else ("SUBIR exposicion" if p99 < 200 else "BAJAR exposicion")
    print(f"[{etiqueta}] centro: p1={p1:.0f} p50={p50:.0f} p99={p99:.0f} saturado={saturado:.1f}% "
          f"medias R,G,B={medias} -> {estado}")
    return {"p1": round(p1, 1), "p50": round(p50, 1), "p99": round(p99, 1),
            "saturado_pct": round(saturado, 2), "medias_rgb": medias, "estado": estado}


# --- SUBIDA ---
def subir_a_oci(ruta_local, carpeta_nube):
    nombre = os.path.basename(ruta_local)
    url = OCI_PAR_URL + carpeta_nube + nombre
    print(f"Subiendo a OCI: {carpeta_nube}{nombre} ...")
    try:
        with open(ruta_local, "rb") as f:
            r = requests.put(url, data=f, timeout=30)
        if r.status_code == 200:
            print(f"Subida OK: {carpeta_nube}{nombre}")
            return True
        print(f"Error {r.status_code}: {r.text}")
    except Exception as e:
        print(f"Error de conexion con OCI: {e}")
    return False


# --- PRINCIPAL ---
def parse_args():
    p = argparse.ArgumentParser(description="Ingesta dual RGB/NIR para la tesis.")
    p.add_argument("--mango-id", type=int, help="ID del mango (obligatorio salvo en --calib).")
    p.add_argument("--cara", type=int, choices=[1, 2, 3, 4], help="Cara del mango (1-4).")
    p.add_argument("--sesion", default=dt.date.today().isoformat(), help="ID de sesion (por defecto, la fecha).")
    p.add_argument("--nota", default="", help="Texto libre que se guarda en el JSON.")
    p.add_argument("--calib", choices=["gris", "aruco"], help="Captura de calibracion; se sube a calib/.")
    # NIR
    p.add_argument("--ev", type=float, default=-3.0, help="EV del NIR, solo en modo automatico.")
    p.add_argument("--nir-shutter", type=int, help="Exposicion manual del NIR en microsegundos.")
    p.add_argument("--nir-gain", type=float, default=1.0, help="Ganancia analogica del NIR (modo manual).")
    p.add_argument("--nir-raw", action="store_true", help="Guardar tambien el RAW DNG del NIR.")
    p.add_argument("--nir-denoise", default="cdn_off", choices=["auto", "off", "cdn_off", "cdn_fast", "cdn_hq"])
    # RGB
    p.add_argument("--rgb-device", default="/dev/video1")
    p.add_argument("--rgb-res", default="1280x720", help="Resolucion RGB (ver lo que soporta la camara).")
    p.add_argument("--rgb-exposure", type=int, help="Exposicion manual del RGB (unidades del driver V4L2).")
    p.add_argument("--rgb-wb", type=int, help="Temperatura de color fija del RGB (Kelvin).")
    p.add_argument("--rgb-gain", type=int, help="Ganancia fija del RGB.")
    p.add_argument("--no-upload", action="store_true", help="No subir a OCI (pruebas).")
    # IMPACTO (dano inducido)
    p.add_argument("--impacto", choices=["si", "no"],
                   help="El mango recibio impacto inducido (obligatorio salvo con --calib).")
    p.add_argument("--cara-impacto", type=int, nargs="+", choices=[1, 2, 3, 4],
                   help="Cara(s) que recibieron el impacto (obligatorio si --impacto si).")
    p.add_argument("--zona-impacto", choices=ZONAS_IMPACTO,
                   help="Zona de la cara donde cayo la bola (obligatorio si --impacto si).")
    p.add_argument("--hora-impacto",
                   help="Hora del impacto: HH:MM (hoy) o AAAA-MM-DDTHH:MM. Alternativa a --minutos-desde-impacto.")
    p.add_argument("--minutos-desde-impacto", type=float,
                   help="Minutos transcurridos desde el impacto. Alternativa a --hora-impacto.")
    p.add_argument("--n-impactos", type=int, help="Numero de impactos (por defecto, uno por cara impactada).")
    p.add_argument("--masa-bola-g", type=float, default=66.0, help="Masa de la bola en gramos (por defecto 66).")
    p.add_argument("--altura-cm", type=float, default=50.0, help="Altura de caida en cm (por defecto 50).")
    a = p.parse_args()
    if a.calib is None:
        faltan = [n for n, v in (("--mango-id", a.mango_id), ("--cara", a.cara), ("--impacto", a.impacto)) if v is None]
        if faltan:
            p.error(f"Faltan parametros obligatorios: {', '.join(faltan)} (salvo con --calib).")
        if a.impacto == "si":
            if not a.cara_impacto or a.zona_impacto is None:
                p.error("Con --impacto si son obligatorios --cara-impacto y --zona-impacto.")
            if (a.hora_impacto is None) == (a.minutos_desde_impacto is None):
                p.error("Con --impacto si indique exactamente uno: --hora-impacto o --minutos-desde-impacto.")
        elif any(v is not None for v in (a.cara_impacto, a.zona_impacto, a.hora_impacto, a.minutos_desde_impacto)):
            p.error("Con --impacto no, no indique datos de impacto.")
    return a


def minutos_desde(hora_impacto, ahora):
    """Minutos desde el impacto a partir de HH:MM (hoy) o AAAA-MM-DDTHH:MM."""
    try:
        t = dt.datetime.fromisoformat(hora_impacto) if "T" in hora_impacto or "-" in hora_impacto \
            else dt.datetime.combine(ahora.date(), dt.time.fromisoformat(hora_impacto))
    except ValueError:
        raise SystemExit(f"--hora-impacto invalida: {hora_impacto!r} (use HH:MM o AAAA-MM-DDTHH:MM).")
    minutos = (ahora - t).total_seconds() / 60
    if minutos < 0:
        raise SystemExit(f"--hora-impacto {hora_impacto} es posterior a la hora actual; "
                         "si el impacto fue otro dia use AAAA-MM-DDTHH:MM.")
    return round(minutos, 1)


def datos_impacto(args, ahora):
    if args.calib or args.impacto is None:
        return None
    if args.impacto == "no":
        return {"impactado": False}
    minutos = args.minutos_desde_impacto if args.minutos_desde_impacto is not None \
        else minutos_desde(args.hora_impacto, ahora)
    energia_j = (args.masa_bola_g / 1000) * 9.81 * (args.altura_cm / 100)
    return {
        "impactado": True,
        "caras_impactadas": sorted(set(args.cara_impacto)),
        # La cara fotografiada recibio el impacto: se espera ver el dano en esta foto.
        "cara_fotografiada_impactada": args.cara in args.cara_impacto,
        "zona": args.zona_impacto,
        "hora_impacto": args.hora_impacto,
        "minutos_desde_impacto": minutos,
        "n_impactos": args.n_impactos if args.n_impactos is not None else len(set(args.cara_impacto)),
        "masa_bola_g": args.masa_bola_g,
        "altura_cm": args.altura_cm,
        "energia_j": round(energia_j, 3),
    }


if __name__ == "__main__":
    args = parse_args()
    ts = int(time.time())
    impacto = datos_impacto(args, dt.datetime.fromtimestamp(ts))  # valida la hora antes de usar el hardware
    if args.calib:
        # Nombres sin el patron <x>_rgb_<ts> para que nunca se emparejen como dato.
        archivo_rgb = os.path.join(OUTPUT_DIR, f"calib-{args.calib}-rgb-{ts}.jpg")
        archivo_nir = os.path.join(OUTPUT_DIR, f"calib-{args.calib}-nir-{ts}.jpg")
        archivo_meta = os.path.join(OUTPUT_DIR, f"calib-{args.calib}-meta-{ts}.json")
        carpetas = {"rgb": "calib/", "nir": "calib/", "nir_raw": "calib/", "meta": "calib/"}
    else:
        archivo_rgb = os.path.join(OUTPUT_DIR, f"mango_rgb_{ts}.jpg")
        archivo_nir = os.path.join(OUTPUT_DIR, f"mango_nir_{ts}.jpg")
        archivo_meta = os.path.join(OUTPUT_DIR, f"mango_meta_{ts}.json")
        carpetas = {"rgb": "news/rgb/", "nir": "news/nir/", "nir_raw": "news/nir_raw/", "meta": "news/meta/"}

    print(f"\n--- INICIANDO CAPTURA (ID: {ts}) mango={args.mango_id} cara={args.cara} calib={args.calib} ---")
    manual_rgb = any(v is not None for v in (args.rgb_exposure, args.rgb_wb, args.rgb_gain))
    exito_rgb = exito_nir = False
    rgb_aplicados, rgb_faltantes, rgb_leidos = {}, [], {}
    try:
        # 1. RGB con luz blanca
        rele_rgb.on()
        time.sleep(0.5)
        if manual_rgb:
            rgb_aplicados, rgb_faltantes = configurar_rgb(args.rgb_device, args.rgb_exposure, args.rgb_wb, args.rgb_gain)
        exito_rgb = tomar_foto_usb(archivo_rgb, args.rgb_device, args.rgb_res, skip=10 if manual_rgb else 30)
        if manual_rgb:
            rgb_leidos = leer_controles_rgb(args.rgb_device, list(rgb_aplicados))
        rele_rgb.off()
        time.sleep(0.2)
        # 2. NIR con luz IR de 850 nm
        rele_nir.on()
        exito_nir = tomar_foto_csi(archivo_nir, args)
        rele_nir.off()
    except Exception as e:
        print(f"ERROR CRITICO en el ciclo principal: {e}")
    finally:
        rele_rgb.off()
        rele_nir.off()
        print("Hardware liberado y apagado.")

    brillo = {"rgb": reportar_brillo(archivo_rgb, "RGB") if exito_rgb else None,
              "nir": reportar_brillo(archivo_nir, "NIR") if exito_nir else None}
    archivo_dng = os.path.splitext(archivo_nir)[0] + ".dng"
    meta = {
        "script_version": SCRIPT_VERSION, "timestamp": ts, "fecha_local": dt.datetime.now().isoformat(timespec="seconds"),
        "mango_id": args.mango_id, "cara": args.cara, "sesion": args.sesion, "calib": args.calib, "nota": args.nota,
        "impacto": impacto,
        "brillo_centro": brillo,
        "exito": {"rgb": exito_rgb, "nir": exito_nir},
        "nir": {"modo": "manual" if args.nir_shutter is not None else "auto",
                "shutter_us": args.nir_shutter, "gain": args.nir_gain if args.nir_shutter is not None else None,
                "ev": args.ev if args.nir_shutter is None else None, "awbgains": "1,1",
                "denoise": args.nir_denoise, "raw_dng": bool(args.nir_raw and os.path.exists(archivo_dng)),
                "comando": comando_nir(archivo_nir, args)},
        "rgb": {"modo": "manual" if manual_rgb else "auto", "resolucion": args.rgb_res, "device": args.rgb_device,
                "pedido": {"exposure": args.rgb_exposure, "wb_kelvin": args.rgb_wb, "gain": args.rgb_gain},
                "controles_aplicados": rgb_aplicados, "controles_leidos": rgb_leidos, "no_aplicados": rgb_faltantes},
    }
    with open(archivo_meta, "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print("-" * 20)

    if not args.no_upload:
        if exito_nir:
            subir_a_oci(archivo_nir, carpetas["nir"])
            if meta["nir"]["raw_dng"]:
                subir_a_oci(archivo_dng, carpetas["nir_raw"])
        if exito_rgb:
            subir_a_oci(archivo_rgb, carpetas["rgb"])
        subir_a_oci(archivo_meta, carpetas["meta"])
    print("--- FIN DEL CICLO ---\n")
