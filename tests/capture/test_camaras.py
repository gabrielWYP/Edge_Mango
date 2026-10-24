"""Constructores de comandos y parseo de salidas de las camaras (sin hardware).

Los fixtures `v4l2_list_ctrls_*`, `rpicam_hello_list_cameras.txt` y `by_id_listado.txt`
son salidas reales capturadas en la Raspberry Pi el 2026-10-21.
`rpicam_metadata_ejemplo.json` es un ejemplo escrito a mano con las claves habituales de
rpicam-still; se reemplaza por una salida real en la prueba con hardware.
"""
import json
from pathlib import Path

import pytest

import camaras as c

FIX = Path(__file__).parent / "fixtures"
DEV = "/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._Arducam_8mp_SN0001-video-index0"


def leer(nombre):
    return (FIX / nombre).read_text(encoding="utf-8")


# --- parseo de v4l2-ctl --list-ctrls (salida real de la Pi) ---------------------------

def test_parsea_los_controles_reales_de_la_arducam():
    ctrls = c.parsear_list_ctrls(leer("v4l2_list_ctrls_arducam_defecto.txt"))
    for nombre in ("gain", "auto_exposure", "exposure_time_absolute", "focus_absolute",
                   "white_balance_automatic", "white_balance_temperature", "sharpness",
                   "backlight_compensation", "power_line_frequency",
                   "exposure_dynamic_framerate", "focus_automatic_continuous"):
        assert nombre in ctrls, nombre
    assert len(ctrls) >= 25


def test_parseo_de_enteros_con_rango():
    ctrls = c.parsear_list_ctrls(leer("v4l2_list_ctrls_arducam_defecto.txt"))
    g = ctrls["gain"]
    assert (g["tipo"], g["min"], g["max"], g["step"], g["valor"]) == ("int", 0, 100, 1, 0)
    f = ctrls["focus_absolute"]
    assert (f["min"], f["max"], f["valor"]) == (1, 1023, 144)
    assert ctrls["brightness"]["min"] == -64  # rangos negativos


def test_parseo_de_booleanos_y_menus():
    ctrls = c.parsear_list_ctrls(leer("v4l2_list_ctrls_arducam_defecto.txt"))
    assert ctrls["white_balance_automatic"]["tipo"] == "bool"
    assert ctrls["white_balance_automatic"]["valor"] == 1
    ae = ctrls["auto_exposure"]
    assert ae["tipo"] == "menu" and ae["valor"] == 3 and ae["etiqueta"] == "Aperture Priority Mode"
    assert ctrls["power_line_frequency"]["etiqueta"] == "50 Hz"


def test_parseo_de_banderas():
    ctrls = c.parsear_list_ctrls(leer("v4l2_list_ctrls_arducam_defecto.txt"))
    assert "inactive" in ctrls["exposure_time_absolute"]["flags"]
    assert ctrls["iris_absolute"]["flags"] == ["read-only", "write-only"]
    assert ctrls["gain"]["flags"] == []


def test_parseo_despues_de_fijar_el_modo_manual():
    ctrls = c.parsear_list_ctrls(leer("v4l2_list_ctrls_arducam_manual.txt"))
    assert ctrls["auto_exposure"]["valor"] == 1 and ctrls["auto_exposure"]["etiqueta"] == "Manual Mode"
    assert ctrls["exposure_time_absolute"]["valor"] == 150
    assert "inactive" not in ctrls["exposure_time_absolute"]["flags"]
    assert ctrls["gain"]["valor"] == 20 and ctrls["focus_absolute"]["valor"] == 300


def test_valores_planos_para_el_json():
    ctrls = c.parsear_list_ctrls(leer("v4l2_list_ctrls_arducam_manual.txt"))
    v = c.valores_de(ctrls)
    assert v["gain"] == 20 and v["focus_absolute"] == 300 and v["auto_exposure"] == 1


def test_texto_vacio_o_basura_no_revienta():
    assert c.parsear_list_ctrls("") == {}
    assert c.parsear_list_ctrls("User Controls\n\nlo que sea\n") == {}


def test_verificar_controles_detecta_lo_que_no_quedo_aplicado():
    leidos = c.valores_de(c.parsear_list_ctrls(leer("v4l2_list_ctrls_arducam_manual.txt")))
    pedidos = {"auto_exposure": 1, "gain": 20, "focus_absolute": 300}
    assert c.diferencias_controles(pedidos, leidos) == {}
    pedidos["gain"] = 50
    pedidos["inventado"] = 3
    dif = c.diferencias_controles(pedidos, leidos)
    assert dif == {"gain": {"pedido": 50, "leido": 20}, "inventado": {"pedido": 3, "leido": None}}


# --- resolucion del dispositivo por by-id ------------------------------------------------

LISTADO_REAL = leer("by_id_listado.txt").split()


def test_dispositivo_configurado_que_existe_se_usa_tal_cual():
    assert c.resolver_dispositivo_rgb(DEV, existe=lambda r: r == DEV, listar=lambda: []) == DEV


def test_dispositivo_ausente_con_un_unico_candidato_arducam_lo_usa():
    candidato = DEV.replace("SN0001", "SN0002")
    dev = c.resolver_dispositivo_rgb(DEV, existe=lambda r: r == candidato, listar=lambda: [candidato, candidato.replace("index0", "index1")])
    assert dev == candidato


def test_dispositivo_ausente_sin_candidatos_da_un_error_claro():
    with pytest.raises(c.DispositivoNoEncontrado) as exc:
        c.resolver_dispositivo_rgb(DEV, existe=lambda r: False, listar=lambda: [])
    assert "by-id" in str(exc.value) and "Arducam" in str(exc.value)


def test_dispositivo_ausente_con_candidatos_ambiguos_los_lista():
    a = DEV.replace("SN0001", "SN0002")
    b = DEV.replace("SN0001", "SN0003")
    with pytest.raises(c.DispositivoNoEncontrado) as exc:
        c.resolver_dispositivo_rgb(DEV, existe=lambda r: r in (a, b), listar=lambda: [a, b])
    assert "SN0002" in str(exc.value) and "SN0003" in str(exc.value)


def test_listado_real_solo_el_index0_es_de_captura():
    assert [r for r in LISTADO_REAL if r.endswith("video-index0")] == [DEV]


# --- comandos ----------------------------------------------------------------------------

def test_comando_v4l2_stream_rgb_usa_3_buffers_y_guarda_el_cuadro_crudo():
    cmd = c.comando_v4l2_stream_rgb(DEV, "3264x2448", 5, "/tmp/out.mjpg")
    assert cmd == ["v4l2-ctl", "-d", DEV,
                   "--set-fmt-video=width=3264,height=2448,pixelformat=MJPG",
                   "--stream-mmap=3", "--stream-skip=5", "--stream-count=1", "--stream-to=/tmp/out.mjpg"]


def test_ffmpeg_nunca_abre_la_camara_en_streaming():
    """ffmpeg v4l2 pide ~32 buffers de 16 MB a 3264x2448 y el OOM killer lo mata en la Zero 2 W."""
    assert not hasattr(c, "comando_ffmpeg_rgb")


def test_comando_mjpeg2jpeg_lee_de_archivo_y_copia_sin_recodificar():
    cmd = c.comando_mjpeg2jpeg("/tmp/a.mjpg", "/tmp/a.jpg")
    assert cmd == ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", "/tmp/a.mjpg",
                   "-c:v", "copy", "-bsf:v", "mjpeg2jpeg", "/tmp/a.jpg"]
    assert "v4l2" not in cmd and "-f" not in cmd   # entrada de archivo, no de dispositivo


def jpeg_con_segmentos(*marcadores):
    """JPEG sintetico con los segmentos dados (cada uno con 2 bytes de carga)."""
    cuerpo = b"".join(b"\xff" + bytes([m]) + b"\x00\x04\x00\x00" for m in marcadores)
    return b"\xff\xd8" + cuerpo + b"\xff\xda\x00\x02" + b"\x12\x34" + b"\xff\xd9"


def test_detecta_si_el_cuadro_trae_tablas_de_huffman():
    assert c.tiene_tablas_huffman(jpeg_con_segmentos(0xC0, 0xDB, 0xC4))
    assert not c.tiene_tablas_huffman(jpeg_con_segmentos(0xC0, 0xDB))


def test_la_deteccion_de_huffman_solo_mira_la_cabecera():
    # unos bytes FF C4 dentro de los datos comprimidos (despues de SOS) no cuentan
    datos = jpeg_con_segmentos(0xC0, 0xDB)[:-2] + b"\xff\xc4\xff\xd9"
    assert not c.tiene_tablas_huffman(datos)


def test_la_deteccion_de_huffman_con_datos_ilegibles_no_revienta():
    assert not c.tiene_tablas_huffman(b"")
    assert not c.tiene_tablas_huffman(b"no es un jpeg")
    assert not c.tiene_tablas_huffman(b"\xff\xd8\xff")


def test_comando_v4l2_fijar_controles_conserva_el_orden():
    cmd = c.comando_v4l2_fijar(DEV, {"auto_exposure": 1, "exposure_time_absolute": 150, "gain": 20})
    assert cmd == ["v4l2-ctl", "-d", DEV, "-c", "auto_exposure=1,exposure_time_absolute=150,gain=20"]


def test_comando_v4l2_listar():
    assert c.comando_v4l2_listar(DEV) == ["v4l2-ctl", "-d", DEV, "--list-ctrls"]


NIR = {"width": 3280, "height": 2464, "mode": "3280:2464", "shutter_us": 20000, "gain": 2.0,
       "denoise": "cdn_off", "awbgains": "1,1", "timeout_ms": 1000, "raw": False}


def test_comando_rpicam_a_resolucion_completa_sin_recorte():
    cmd = c.comando_rpicam("/tmp/nir.jpg", "/tmp/nir.json", NIR)
    assert cmd == ["rpicam-still", "-o", "/tmp/nir.jpg",
                   "--width", "3280", "--height", "2464", "--mode", "3280:2464",
                   "--shutter", "20000", "--gain", "2.0", "--awbgains", "1,1",
                   "--denoise", "cdn_off", "-t", "1000", "--nopreview",
                   "--metadata", "/tmp/nir.json", "--metadata-format", "json"]
    assert "1920:1080" not in " ".join(cmd)  # ese modo es un recorte central


def test_comando_rpicam_raw_es_opcional():
    assert "--raw" not in c.comando_rpicam("/tmp/n.jpg", "/tmp/n.json", NIR)
    cmd = c.comando_rpicam("/tmp/n.jpg", "/tmp/n.json", dict(NIR, raw=True))
    assert "--raw" in cmd


def test_ruta_del_dng_junto_al_jpg():
    assert c.ruta_raw("/x/mango_nir_5.jpg") == "/x/mango_nir_5.dng"


def test_comando_listar_camaras_csi():
    assert c.comando_rpicam_listar() == ["rpicam-hello", "--list-cameras"]


def test_deteccion_de_la_camara_nir():
    assert c.camara_nir_presente(leer("rpicam_hello_list_cameras.txt"))
    assert not c.camara_nir_presente("No cameras available!\n")
    assert not c.camara_nir_presente("")
    assert not c.camara_nir_presente("Available cameras\n-----------------\n")


def test_metadata_de_rpicam_se_conserva_completa():
    meta = c.parsear_metadata_rpicam(leer("rpicam_metadata_ejemplo.json"))
    assert meta["ExposureTime"] == 20000 and meta["AnalogueGain"] == 2.0
    assert meta["SensorTemperature"] == 41.0 and meta["Lux"] == pytest.approx(154.31)


def test_metadata_de_rpicam_invalida_no_revienta():
    assert c.parsear_metadata_rpicam("no es json") == {}
    assert c.parsear_metadata_rpicam("") == {}


# --- salud: vcgencmd get_throttled ------------------------------------------------------------

def test_throttled_real_sin_problemas():
    t = c.parsear_throttled(leer("vcgencmd_throttled.txt"))
    assert t == {"raw": "0x0", "valor": 0, "flags": []}


def test_throttled_con_subtension_y_limitacion():
    t = c.parsear_throttled("throttled=0x50005\n")
    assert t["valor"] == 0x50005
    assert set(t["flags"]) == {"subtension_ahora", "limitado_ahora",
                               "subtension_ocurrio", "limitado_ocurrio"}


def test_throttled_ilegible():
    assert c.parsear_throttled("") == {"raw": None, "valor": None, "flags": []}


# --- jpeg crudo de la camara ----------------------------------------------------------------------

def test_recortar_jpeg_quita_el_relleno_despues_del_ultimo_eoi():
    jpg = b"\xff\xd8\xff\xc0abc\xff\xd9"
    assert c.recortar_jpeg(jpg + b"\x00\x00\x00\x00") == jpg
    assert c.recortar_jpeg(jpg) == jpg


def test_recortar_jpeg_sin_eoi_devuelve_los_bytes_intactos():
    assert c.recortar_jpeg(b"\xff\xd8basura") == b"\xff\xd8basura"


# --- configuracion serializable ----------------------------------------------------------------------

def test_los_comandos_son_serializables_a_json():
    cmd = c.comando_rpicam("/tmp/n.jpg", "/tmp/n.json", NIR)
    assert json.loads(json.dumps(cmd)) == cmd


# --- rpicam: prueba rapida de salud con --immediate ------------------------------------------------------------

def test_rpicam_immediate_es_opcional_y_va_antes_de_los_metadatos():
    sin = c.comando_rpicam("/tmp/n.jpg", "/tmp/n.json", NIR)
    assert "--immediate" not in sin
    con = c.comando_rpicam("/tmp/n.jpg", "/tmp/n.json", dict(NIR, immediate=True, width=640, height=480,
                                                              mode="640:480"))
    assert "--immediate" in con
    assert con[con.index("--width") + 1] == "640" and con[con.index("--mode") + 1] == "640:480"


# --- fuser: quien tiene abiertos los nodos de video y de medios -------------------------------------------------

def test_comando_fuser_lista_los_dispositivos():
    assert c.comando_fuser(["/dev/media0", "/dev/video1"]) == ["fuser", "/dev/media0", "/dev/video1"]


def test_pids_de_la_salida_real_de_fuser_sin_repetir():
    # salida real de la Pi: los PIDs van a stdout y los nombres de archivo a stderr
    assert c.parsear_pids_fuser(leer("fuser_stdout_ocupado.txt")) == [2780]
    assert c.parsear_pids_fuser("  10  20m 20\n") == [10, 20]


def test_pids_de_fuser_con_salida_vacia_o_irreconocible():
    assert c.parsear_pids_fuser("") == []
    assert c.parsear_pids_fuser("fuser: error raro\n") == []
