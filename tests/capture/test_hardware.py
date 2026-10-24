"""Adaptadores de hardware (luces, camaras, salud) con un ejecutor de comandos simulado."""
import json
import re
from pathlib import Path

import pytest
from PIL import Image

import camaras
import configuracion
import hardware as hw

FIX = Path(__file__).parent / "fixtures"
DEV = camaras.DEVICE_RGB_DEFECTO
CFG = configuracion.DEFECTOS


def fixture(nombre):
    return (FIX / nombre).read_text(encoding="utf-8")


def jpeg_falso(ruta, relleno=b""):
    """JPEG minimo valido (1x1) con un relleno opcional despues del EOI."""
    import io
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buf, "JPEG")
    Path(ruta).write_bytes(buf.getvalue() + relleno)


def quitar_dht(datos):
    """Copia del JPEG sin los segmentos DHT (tablas de Huffman), como un MJPEG crudo de camara."""
    import struct
    salida, i = bytearray(datos[:2]), 2
    while i < len(datos):
        marcador = datos[i + 1]
        largo = struct.unpack(">H", datos[i + 2:i + 4])[0]
        if marcador != 0xC4:
            salida += datos[i:i + 2 + largo]
        i += 2 + largo
        if marcador == 0xDA:
            salida += datos[i:]
            break
    return bytes(salida)


class EjecutorFalso:
    """Registra los comandos y responde segun el binario."""

    def __init__(self):
        self.llamadas = []
        self.timeouts = []
        self.eventos = []         # ("cmd", comando) y ("dormir", segundos) en orden
        self.fallar = {}          # binario -> Resultado a devolver
        self.relleno = b"\x00\x00\x00\x00"
        self.sin_archivo = False
        self.sin_dht = False      # el cuadro crudo de la camara no trae tablas de Huffman
        # estado de la camara: arranca como la salida real por defecto (modos automaticos activos)
        self.ctrls = camaras.valores_de(camaras.parsear_list_ctrls(fixture("v4l2_list_ctrls_arducam_defecto.txt")))
        self.rechazar = {}        # control -> mensaje: la camara lo rechaza (exit 255)
        self.ignorar = set()      # controles que se aceptan pero no cambian de valor

    def _texto_controles(self):
        texto = fixture("v4l2_list_ctrls_arducam_defecto.txt")
        for nombre, valor in self.ctrls.items():
            texto = re.sub(rf"^(\s*{nombre}\s+0x\w+\s+\(\w+\)\s*:.*?value=)-?\d+", rf"\g<1>{valor}",
                           texto, flags=re.MULTILINE)
        return texto

    def _rechazo(self, nombre, estado):
        """Reglas del driver: un control dependiente se rechaza mientras su modo automatico siga activo."""
        if nombre in self.rechazar:
            return self.rechazar[nombre]
        if nombre == "white_balance_temperature" and estado["white_balance_automatic"] == 1:
            return "white_balance_temperature: inactivo con white_balance_automatic=1"
        if nombre == "exposure_time_absolute" and estado["auto_exposure"] != 1:
            return "exposure_time_absolute: inactivo mientras auto_exposure no sea manual"
        return None

    def _fijar(self, spec):
        pares = [p.split("=") for p in spec.split(",")]
        original = dict(self.ctrls)
        actual = dict(self.ctrls)
        for nombre, valor in pares:
            # una llamada con varios controles valida todo contra el estado ANTERIOR (como el driver)
            motivo = self._rechazo(nombre, original if len(pares) > 1 else actual)
            if motivo:
                return hw.Resultado(255, "", motivo)
            if nombre not in self.ignorar:
                actual[nombre] = int(valor)
        self.ctrls = actual
        return hw.Resultado(0, "", "")

    def dormir(self, segundos):
        self.eventos.append(("dormir", segundos))

    def __call__(self, cmd, timeout=60):
        self.llamadas.append(list(cmd))
        self.timeouts.append(timeout)
        self.eventos.append(("cmd", list(cmd)))
        binario = cmd[0]
        if binario in self.fallar:
            return self.fallar[binario]
        if binario == "v4l2-ctl" and "--list-ctrls" in cmd:
            return hw.Resultado(0, self._texto_controles(), "")
        if binario == "v4l2-ctl" and "-c" in cmd:
            return self._fijar(cmd[cmd.index("-c") + 1])
        if binario == "v4l2-ctl" and any(a.startswith("--stream-to=") for a in cmd):
            destino = next(a.split("=", 1)[1] for a in cmd if a.startswith("--stream-to="))
            if not self.sin_archivo:
                jpeg_falso(destino, self.relleno)
                if self.sin_dht:
                    ruta = Path(destino)
                    ruta.write_bytes(quitar_dht(ruta.read_bytes()))
            return hw.Resultado(0, "", "<<<<<")
        if binario == "ffmpeg":
            assert "-i" in cmd and cmd[cmd.index("-i") + 1] != "", "ffmpeg solo se usa con entrada de archivo"
            assert "v4l2" not in cmd, "ffmpeg no debe abrir el dispositivo de video"
            jpeg_falso(cmd[-1])    # mjpeg2jpeg agrega las tablas de Huffman
            return hw.Resultado(0, "", "")
        if binario == "rpicam-still":
            jpeg_falso(cmd[cmd.index("-o") + 1])
            Path(cmd[cmd.index("--metadata") + 1]).write_text(fixture("rpicam_metadata_ejemplo.json"))
            if "--raw" in cmd:
                Path(camaras.ruta_raw(cmd[cmd.index("-o") + 1])).write_bytes(b"DNG")
            return hw.Resultado(0, "", "")
        if binario == "rpicam-hello":
            return hw.Resultado(0, fixture("rpicam_hello_list_cameras.txt"), "")
        if binario == "vcgencmd":
            return hw.Resultado(0, fixture("vcgencmd_throttled.txt"), "")
        if binario == "timedatectl":
            return hw.Resultado(0, "yes\n", "")
        return hw.Resultado(0, "", "")


def es_lista(cmd):
    return cmd[0] == "v4l2-ctl" and "--list-ctrls" in cmd


def es_stream(cmd):
    return cmd[0] == "v4l2-ctl" and any(a.startswith("--stream-to=") for a in cmd)


def fijados(ej):
    """Controles fijados (uno por llamada `-c nombre=valor`), en orden."""
    return [c[c.index("-c") + 1] for c in ej.llamadas if c[0] == "v4l2-ctl" and "-c" in c]


# --- ejecutor real ---------------------------------------------------------------------------------

def test_ejecutor_real_captura_salida_y_codigo():
    r = hw.ejecutar(["python3", "-c", "print('hola')"], timeout=10)
    assert r.rc == 0 and r.salida.strip() == "hola"


def test_ejecutor_real_binario_inexistente_no_lanza_excepcion():
    r = hw.ejecutar(["no-existe-este-binario"], timeout=5)
    assert r.rc == 127 and "no-existe" in r.error


def test_ejecutor_real_timeout_devuelve_124():
    r = hw.ejecutar(["python3", "-c", "import time; time.sleep(5)"], timeout=0.3)
    assert r.rc == 124


def test_ejecutor_real_en_timeout_manda_sigterm_primero_y_espera_a_que_termine():
    """Un rpicam-still matado con SIGKILL puede dejar la CSI trabada hasta reiniciar la Pi."""
    r = hw.ejecutar(["python3", "-c", "import time; time.sleep(30)"], timeout=0.3, gracia_s=3)
    assert r.rc == 124 and "SIGTERM" in r.error and "SIGKILL" not in r.error


def test_ejecutor_real_usa_sigkill_solo_como_ultimo_recurso():
    script = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
    r = hw.ejecutar(["python3", "-c", script], timeout=1.0, gracia_s=0.5)
    assert r.rc == 124 and "SIGKILL" in r.error


# --- luces ------------------------------------------------------------------------------------------

class LuzFalsa:
    def __init__(self, pin, eventos):
        self.pin, self.eventos = pin, eventos
        eventos.append(("crear", pin))

    def on(self):
        self.eventos.append(("on", self.pin))

    def off(self):
        self.eventos.append(("off", self.pin))

    def close(self):
        self.eventos.append(("close", self.pin))


def luces_de_prueba():
    eventos = []
    luces = hw.Luces(27, 22, espera_s=0.4, fabrica=lambda pin: LuzFalsa(pin, eventos),
                     dormir=lambda s: eventos.append(("dormir", s)))
    return luces, eventos


def test_las_luces_se_apagan_al_crearlas_y_usan_los_pines_del_protocolo():
    luces, eventos = luces_de_prueba()
    assert ("crear", 27) in eventos and ("crear", 22) in eventos


def test_encendida_espera_a_que_se_estabilice_y_apaga_al_salir():
    luces, eventos = luces_de_prueba()
    del eventos[:]
    with luces.encendida("rgb"):
        eventos.append(("captura",))
    assert eventos == [("on", 27), ("dormir", 0.4), ("captura",), ("off", 27)]


def test_encendida_apaga_aunque_la_captura_falle():
    luces, eventos = luces_de_prueba()
    del eventos[:]
    with pytest.raises(RuntimeError):
        with luces.encendida("nir"):
            raise RuntimeError("fallo")
    assert eventos[-1] == ("off", 22)


def test_apagar_todo_apaga_ambas():
    luces, eventos = luces_de_prueba()
    del eventos[:]
    luces.apagar_todo()
    assert ("off", 27) in eventos and ("off", 22) in eventos


def test_luz_desconocida():
    luces, _ = luces_de_prueba()
    with pytest.raises(ValueError):
        with luces.encendida("uv"):
            pass


# --- camara RGB ----------------------------------------------------------------------------------------

# valores reales de la camara con el modo manual aplicado (fixture capturado en la Pi)
_REAL_MANUAL = camaras.valores_de(camaras.parsear_list_ctrls(fixture("v4l2_list_ctrls_arducam_manual.txt")))
CONTROLES_PEDIDOS = {n: _REAL_MANUAL[n] for n in CFG["rgb"]["controles"]}


def camara_rgb(ejecutor, controles=None, **cambios):
    cfg = json.loads(json.dumps(CFG["rgb"]))
    cfg["controles"] = {**CONTROLES_PEDIDOS, **(controles or {})}
    cfg.update(cambios)
    return hw.CamaraRGB(cfg, ejecutor, DEV, dormir=ejecutor.dormir)


def test_aplicar_controles_lee_primero_y_fija_uno_por_llamada_con_los_modos_automaticos_primero():
    """Varios controles en una sola llamada de v4l2-ctl fallan (exit 255) si un modo automatico sigue activo."""
    ej = EjecutorFalso()
    cam = camara_rgb(ej, controles={"white_balance_temperature": 5000})
    r = cam.aplicar_controles()
    assert es_lista(ej.llamadas[0])                       # primero se lee el estado actual
    spec = fijados(ej)
    assert spec and all("," not in x for x in spec)       # un solo control por llamada
    nombres = [x.split("=")[0] for x in spec]
    assert nombres.index("auto_exposure") < nombres.index("exposure_time_absolute")
    assert nombres.index("white_balance_automatic") < nombres.index("white_balance_temperature")
    assert r["fallidos"] == {} and sorted(r["aplicados"]) == sorted(set(nombres))


CICLO_EXPOSICION = ["auto_exposure=3", "auto_exposure=1", "exposure_dynamic_framerate=0"]


def test_exposicion_alterna_a_automatico_y_vuelve_a_manual_ANTES_de_fijar_el_tiempo():
    """Trampa del firmware de la Arducam: tras un ciclo de alimentacion, escribir auto_exposure=1 cuando
    ya lee 1 NO cambia el modo interno (0.5, 2 y 6 ms daban el mismo brillo)."""
    ej = EjecutorFalso()
    ej.ctrls["auto_exposure"] = 1                          # la camara ya "lee" manual
    camara_rgb(ej).aplicar_controles()
    e = CONTROLES_PEDIDOS["exposure_time_absolute"]
    assert fijados(ej)[:4] == CICLO_EXPOSICION + [f"exposure_time_absolute={e}"]


def test_hay_una_pausa_de_0_2_s_entre_auto_exposure_3_y_auto_exposure_1():
    ej = EjecutorFalso()
    camara_rgb(ej).aplicar_controles()
    secuencia = [("pausa", x[1]) if x[0] == "dormir" else ("set", x[1][x[1].index("-c") + 1])
                 for x in ej.eventos if x[0] == "dormir" or "-c" in x[1]]
    i = secuencia.index(("set", "auto_exposure=3"))
    assert secuencia[i + 1] == ("pausa", 0.2) and secuencia[i + 2] == ("set", "auto_exposure=1")


def test_el_ciclo_de_exposicion_se_repite_en_cada_aplicacion_aunque_nada_haya_cambiado():
    ej = EjecutorFalso()
    cam = camara_rgb(ej)
    cam.aplicar_controles()
    antes = len(fijados(ej))
    r = cam.aplicar_controles()
    e = CONTROLES_PEDIDOS["exposure_time_absolute"]
    assert fijados(ej)[antes:] == CICLO_EXPOSICION + [f"exposure_time_absolute={e}"]   # solo la exposicion
    assert r["aplicados"] == ["auto_exposure", "exposure_dynamic_framerate", "exposure_time_absolute"]
    assert r["fallidos"] == {}


def test_el_ciclo_usa_la_exposicion_pedida_en_extra():
    ej = EjecutorFalso()
    camara_rgb(ej).aplicar_controles(extra={"exposure_time_absolute": 21})
    assert fijados(ej)[3] == "exposure_time_absolute=21"


def test_los_controles_obligatorios_se_fijan_aunque_la_camara_arranque_con_el_valor_de_fabrica():
    """backlight_compensation=1 de fabrica es un realce adaptativo que anula la exposicion fija."""
    ej = EjecutorFalso()                                    # de fabrica: backlight=1, power_line=1 (50 Hz)
    camara_rgb(ej).aplicar_controles()
    assert "backlight_compensation=0" in fijados(ej) and "power_line_frequency=0" in fijados(ej)


@pytest.mark.parametrize("control", ["backlight_compensation", "power_line_frequency"])
def test_si_un_control_obligatorio_no_se_puede_fijar_es_un_error_no_un_aviso(control):
    ej = EjecutorFalso()
    ej.rechazar[control] = "Invalid argument"
    with pytest.raises(hw.ErrorCamara) as exc:
        camara_rgb(ej).aplicar_controles()
    assert control in str(exc.value) and "obligatorio" in str(exc.value)


def test_si_el_valor_leido_de_un_control_obligatorio_no_coincide_es_un_error(tmp_path):
    ej = EjecutorFalso()
    ej.ignorar = {"power_line_frequency"}                  # acepta la escritura pero queda en 1
    salida = tmp_path / "a.jpg"
    with pytest.raises(hw.ErrorCamara) as exc:
        camara_rgb(ej).capturar(salida)
    assert "power_line_frequency" in str(exc.value) and "obligatorio" in str(exc.value)
    assert not salida.exists()                             # la foto tomada con el ajuste mal no se conserva


def test_aplicar_controles_solo_fija_lo_que_cambio_salvo_el_ciclo_de_exposicion():
    ej = EjecutorFalso()
    cam = camara_rgb(ej)
    cam.aplicar_controles()
    fijados_antes = fijados(ej)
    cam.aplicar_controles()
    nuevos = fijados(ej)[len(fijados_antes):]
    assert {x.split("=")[0] for x in nuevos} <= {"auto_exposure", "exposure_dynamic_framerate",
                                                 "exposure_time_absolute"}


def test_aplicar_controles_registra_el_que_falla_y_sigue_con_los_demas():
    ej = EjecutorFalso()
    ej.rechazar["focus_absolute"] = "VIDIOC_S_EXT_CTRLS: Invalid argument"
    r = camara_rgb(ej).aplicar_controles()
    assert list(r["fallidos"]) == ["focus_absolute"] and "Invalid argument" in r["fallidos"]["focus_absolute"]
    assert "gain" in r["aplicados"] and "focus_absolute" not in r["aplicados"]


def test_aplicar_controles_si_no_se_pueden_leer_es_un_error_claro():
    ej = EjecutorFalso()
    ej.fallar["v4l2-ctl"] = hw.Resultado(1, "", "Cannot open device")
    with pytest.raises(hw.ErrorCamara) as exc:
        camara_rgb(ej).aplicar_controles()
    assert "rgb" in str(exc.value).lower() and "Cannot open" in str(exc.value)


def test_capturar_rgb_guarda_el_jpeg_de_la_camara_sin_relleno_y_vuelca_los_controles(tmp_path):
    ej = EjecutorFalso()
    salida = tmp_path / "mango_rgb_1.jpg"
    info = camara_rgb(ej).capturar(salida)
    datos = salida.read_bytes()
    assert datos.endswith(b"\xff\xd9")                      # relleno recortado
    assert info["metodo"] == "v4l2-ctl"
    assert info["resolucion"] == "3264x2448" and info["device_by_id"] == DEV
    assert "--stream-skip=5" in info["comando"]
    assert info["controles"]["gain"] == 20 and info["controles"]["focus_absolute"] == 300
    assert info["controles"]["brightness"] == 0              # volcado COMPLETO, no solo los pedidos
    assert info["diferencias"] == {} and info["controles_fallidos"] == {}
    # orden: leer, fijar uno por uno, capturar, y volcar controles DESPUES de la captura
    assert es_lista(ej.llamadas[0]) and es_lista(ej.llamadas[-1])
    indices = [i for i, c in enumerate(ej.llamadas) if es_stream(c)]
    assert len(indices) == 1
    assert not any("-c" in c for c in ej.llamadas[indices[0]:])


def test_capturar_rgb_reporta_controles_que_no_quedaron_como_se_pidieron(tmp_path):
    ej = EjecutorFalso()
    ej.ignorar = {"gain"}                    # la camara acepta el valor pero no lo aplica
    info = camara_rgb(ej, controles={"gain": 30}).capturar(tmp_path / "a.jpg")
    assert info["diferencias"] == {"gain": {"pedido": 30, "leido": 0}}


def test_capturar_rgb_informa_los_controles_que_fallaron_sin_abortar(tmp_path):
    ej = EjecutorFalso()
    ej.rechazar["focus_absolute"] = "Invalid argument"
    info = camara_rgb(ej).capturar(tmp_path / "a.jpg")
    assert "focus_absolute" in info["controles_fallidos"]
    assert (tmp_path / "a.jpg").exists()


def test_capturar_rgb_rapida_con_resolucion_y_descarte_propios(tmp_path):
    ej = EjecutorFalso()
    info = camara_rgb(ej).capturar(tmp_path / "a.jpg", resolucion="1600x1200", descartar=2, volcar=False)
    assert "--set-fmt-video=width=1600,height=1200,pixelformat=MJPG" in info["comando"]
    assert "--stream-skip=2" in info["comando"]
    assert info["resolucion"] == "1600x1200"
    assert not es_lista(ej.llamadas[-1]) and info["controles"] == {}      # sin volcado final: mas rapido


def test_capturar_rgb_con_error_del_stream(tmp_path):
    ej = EjecutorFalso()
    ej.sin_archivo = True
    with pytest.raises(hw.ErrorCamara) as exc:
        camara_rgb(ej).capturar(tmp_path / "a.jpg")
    assert "no genero" in str(exc.value)


def test_el_cuadro_crudo_va_a_un_temporal_y_no_queda_nada_junto_a_la_foto(tmp_path):
    ej = EjecutorFalso()
    salida = tmp_path / "mango_rgb_1.jpg"
    info = camara_rgb(ej).capturar(salida)
    assert [p.name for p in tmp_path.iterdir()] == ["mango_rgb_1.jpg"]
    destino = next(a for a in info["comando"] if a.startswith("--stream-to="))
    assert destino.endswith(".mjpg.tmp") and not destino.endswith("mango_rgb_1.jpg")


def test_si_el_cuadro_ya_trae_tablas_de_huffman_no_hace_falta_ffmpeg(tmp_path):
    ej = EjecutorFalso()
    info = camara_rgb(ej).capturar(tmp_path / "a.jpg")
    assert info["tablas_huffman_en_origen"] is True and info["comando_mjpeg2jpeg"] is None
    assert not any(c[0] == "ffmpeg" for c in ej.llamadas)
    assert camaras.tiene_tablas_huffman((tmp_path / "a.jpg").read_bytes())


def test_si_el_cuadro_no_trae_tablas_se_agregan_con_mjpeg2jpeg_desde_archivo(tmp_path):
    ej = EjecutorFalso()
    ej.sin_dht = True
    salida = tmp_path / "a.jpg"
    info = camara_rgb(ej).capturar(salida)
    assert info["tablas_huffman_en_origen"] is False
    ff = [c for c in ej.llamadas if c[0] == "ffmpeg"]
    assert len(ff) == 1 and ff[0] == info["comando_mjpeg2jpeg"]
    assert ff[0][ff[0].index("-bsf:v") + 1] == "mjpeg2jpeg" and ff[0][-1] == str(salida)
    assert camaras.tiene_tablas_huffman(salida.read_bytes())
    assert [p.name for p in tmp_path.iterdir()] == ["a.jpg"]
    # el orden es: capturar con v4l2-ctl, mjpeg2jpeg desde archivo, y despues volcar los controles
    nombres = [c[0] for c in ej.llamadas]
    i = nombres.index("ffmpeg")
    assert es_stream(ej.llamadas[i - 1]) and es_lista(ej.llamadas[i + 1]) and i + 2 == len(nombres)


def test_mjpeg2jpeg_siempre_lo_aplica_aunque_el_cuadro_ya_tenga_tablas(tmp_path):
    ej = EjecutorFalso()
    info = camara_rgb(ej, mjpeg2jpeg="siempre").capturar(tmp_path / "a.jpg")
    assert info["comando_mjpeg2jpeg"] is not None and any(c[0] == "ffmpeg" for c in ej.llamadas)


def test_mjpeg2jpeg_nunca_no_lo_aplica_aunque_falten_las_tablas(tmp_path):
    ej = EjecutorFalso()
    ej.sin_dht = True
    info = camara_rgb(ej, mjpeg2jpeg="nunca").capturar(tmp_path / "a.jpg")
    assert info["comando_mjpeg2jpeg"] is None and not any(c[0] == "ffmpeg" for c in ej.llamadas)


def test_error_de_ffmpeg_al_agregar_las_tablas(tmp_path):
    ej = EjecutorFalso()
    ej.sin_dht = True
    ej.fallar["ffmpeg"] = hw.Resultado(1, "", "Invalid data found")
    with pytest.raises(hw.ErrorCamara) as exc:
        camara_rgb(ej).capturar(tmp_path / "a.jpg")
    assert "mjpeg2jpeg" in str(exc.value) and "Invalid data" in str(exc.value)
    assert [p.name for p in tmp_path.iterdir()] == []        # sin restos


def test_valor_desconocido_de_mjpeg2jpeg(tmp_path):
    with pytest.raises(hw.ErrorCamara):
        camara_rgb(EjecutorFalso(), mjpeg2jpeg="a_veces").capturar(tmp_path / "a.jpg")


def test_los_controles_extra_se_aplican_sobre_los_de_la_configuracion(tmp_path):
    ej = EjecutorFalso()
    camara_rgb(ej).capturar(tmp_path / "a.jpg", extra={"white_balance_temperature": 3300})
    spec = fijados(ej)
    assert "white_balance_temperature=3300" in spec and "white_balance_temperature=4600" not in spec
    assert "gain=20" in spec                                   # el resto sigue igual


def test_desde_config_resuelve_el_dispositivo_by_id():
    ej = EjecutorFalso()
    cam = hw.CamaraRGB.desde_config(CFG["rgb"], ej, existe=lambda r: True, listar=lambda: [])
    assert cam.device == DEV


def test_desde_config_sin_camara_da_error_claro():
    with pytest.raises(camaras.DispositivoNoEncontrado):
        hw.CamaraRGB.desde_config(CFG["rgb"], EjecutorFalso(), existe=lambda r: False, listar=lambda: [])


# --- camara NIR -------------------------------------------------------------------------------------------

def test_nir_presente_con_la_salida_real():
    assert hw.CamaraNIR(CFG["nir"], EjecutorFalso()).presente()


def test_nir_ausente_cuando_rpicam_no_lista_camaras():
    ej = EjecutorFalso()
    ej.fallar["rpicam-hello"] = hw.Resultado(0, "No cameras available!\n", "")
    assert not hw.CamaraNIR(CFG["nir"], ej).presente()
    ej.fallar["rpicam-hello"] = hw.Resultado(1, "", "error")
    assert not hw.CamaraNIR(CFG["nir"], ej).presente()


def test_capturar_nir_devuelve_comando_y_metadata_de_rpicam(tmp_path):
    ej = EjecutorFalso()
    info = hw.CamaraNIR(CFG["nir"], ej).capturar(tmp_path / "mango_nir_1.jpg", tmp_path / "t.json")
    cmd = ej.llamadas[0]
    assert cmd[0] == "rpicam-still" and "--mode" in cmd and "3280:2464" in cmd
    assert info["comando"] == cmd
    assert info["rpicam_metadata"]["ExposureTime"] == 20000
    assert info["raw"] is None and (tmp_path / "mango_nir_1.jpg").exists()


def test_capturar_nir_acepta_ajustes_extra_para_la_lectura_rapida_de_enfoque(tmp_path):
    ej = EjecutorFalso()
    rapida = {"width": 1640, "height": 1232, "mode": "1640:1232", "timeout_ms": 200}
    hw.CamaraNIR(CFG["nir"], ej).capturar(tmp_path / "n.jpg", tmp_path / "t.json", extra=rapida)
    cmd = ej.llamadas[0]
    assert "1640:1232" in cmd and cmd[cmd.index("-t") + 1] == "200"
    assert cmd[cmd.index("--shutter") + 1] == str(CFG["nir"]["shutter_us"])  # el resto se conserva


def test_capturar_nir_con_raw_devuelve_la_ruta_del_dng(tmp_path):
    ej = EjecutorFalso()
    cfg = dict(CFG["nir"], raw=True)
    info = hw.CamaraNIR(cfg, ej).capturar(tmp_path / "mango_nir_1.jpg", tmp_path / "t.json")
    assert info["raw"] == str(tmp_path / "mango_nir_1.dng")


def test_capturar_nir_con_error(tmp_path):
    ej = EjecutorFalso()
    ej.fallar["rpicam-still"] = hw.Resultado(1, "", "ERROR: no cameras available")
    with pytest.raises(hw.ErrorCamara) as exc:
        hw.CamaraNIR(CFG["nir"], ej).capturar(tmp_path / "n.jpg", tmp_path / "t.json")
    assert "nir" in str(exc.value).lower() and "no cameras" in str(exc.value)


MENSAJE_NOIR_TRABADA = ("la NoIR quedó trabada: reiniciá la Pi (sudo reboot) y retomá con el mismo "
                        "comando; el estado se conserva")


def test_timeout_de_rpicam_deja_la_noir_trabada_y_no_se_reintenta_en_el_mismo_proceso(tmp_path):
    """Un rpicam-still colgado o matado deja la CSI trabada ("Camera frontend has timed out") hasta reiniciar."""
    ej = EjecutorFalso()
    ej.fallar["rpicam-still"] = hw.Resultado(124, "", "timeout")
    cam = hw.CamaraNIR(CFG["nir"], ej)
    with pytest.raises(hw.NIRTrabada) as exc:
        cam.capturar(tmp_path / "n.jpg", tmp_path / "t.json")
    assert MENSAJE_NOIR_TRABADA in str(exc.value)
    assert isinstance(exc.value, hw.ErrorCamara) and cam.trabada is True
    llamadas = len(ej.llamadas)
    with pytest.raises(hw.NIRTrabada):
        cam.capturar(tmp_path / "n.jpg", tmp_path / "t.json")      # no vuelve a ejecutar nada
    assert len(ej.llamadas) == llamadas


def test_camera_frontend_has_timed_out_tambien_es_noir_trabada(tmp_path):
    ej = EjecutorFalso()
    ej.fallar["rpicam-still"] = hw.Resultado(1, "", "ERROR: Camera frontend has timed out!")
    with pytest.raises(hw.NIRTrabada) as exc:
        hw.CamaraNIR(CFG["nir"], ej).capturar(tmp_path / "n.jpg", tmp_path / "t.json")
    assert MENSAJE_NOIR_TRABADA in str(exc.value)


def test_un_error_comun_de_rpicam_no_marca_la_camara_como_trabada(tmp_path):
    ej = EjecutorFalso()
    ej.fallar["rpicam-still"] = hw.Resultado(1, "", "ERROR: no cameras available")
    cam = hw.CamaraNIR(CFG["nir"], ej)
    with pytest.raises(hw.ErrorCamara) as exc:
        cam.capturar(tmp_path / "n.jpg", tmp_path / "t.json")
    assert not isinstance(exc.value, hw.NIRTrabada) and cam.trabada is False


def test_rpicam_se_ejecuta_con_un_timeout_de_unos_25_s(tmp_path):
    ej = EjecutorFalso()
    hw.CamaraNIR(CFG["nir"], ej).capturar(tmp_path / "n.jpg", tmp_path / "t.json")
    assert ej.timeouts[-1] == 25 and CFG["nir"]["timeout_s"] == 25


def test_probar_hace_una_captura_rapida_640x480_inmediata_con_timeout_de_10_s(tmp_path):
    ej = EjecutorFalso()
    hw.CamaraNIR(CFG["nir"], ej).probar(tmp_path)
    cmd = ej.llamadas[0]
    assert "--immediate" in cmd and cmd[cmd.index("--mode") + 1] == "640:480"
    assert cmd[cmd.index("--width") + 1] == "640" and cmd[cmd.index("--height") + 1] == "480"
    assert ej.timeouts[0] == 10
    assert list(tmp_path.iterdir()) == []                  # no deja archivos


def test_probar_con_la_camara_trabada_aborta_con_el_mensaje_de_reinicio(tmp_path):
    ej = EjecutorFalso()
    ej.fallar["rpicam-still"] = hw.Resultado(124, "", "timeout")
    with pytest.raises(hw.NIRTrabada) as exc:
        hw.CamaraNIR(CFG["nir"], ej).probar(tmp_path)
    assert MENSAJE_NOIR_TRABADA in str(exc.value)


# --- salud -------------------------------------------------------------------------------------------------

def test_salud_throttled_y_reloj():
    s = hw.Salud(EjecutorFalso())
    assert s.throttled() == {"raw": "0x0", "valor": 0, "flags": []}
    assert s.reloj_sincronizado() is True


def test_salud_sin_vcgencmd_ni_timedatectl_no_revienta():
    ej = EjecutorFalso()
    ej.fallar["vcgencmd"] = hw.Resultado(127, "", "no se encontro")
    ej.fallar["timedatectl"] = hw.Resultado(127, "", "no se encontro")
    s = hw.Salud(ej)
    assert s.throttled()["raw"] is None
    assert s.reloj_sincronizado() is None


def test_reloj_no_sincronizado():
    ej = EjecutorFalso()
    ej.fallar["timedatectl"] = hw.Resultado(0, "no\n", "")
    assert hw.Salud(ej).reloj_sincronizado() is False


# --- procesos que usan las camaras (fuser) ---------------------------------------------------------------------

def salud_con_fuser(ej, pid_propio=1):
    return hw.Salud(ej, listar_dispositivos=lambda: ["/dev/media4", "/dev/video1"],
                    leer_nombre=lambda pid: "v4l2-ctl", pid_propio=pid_propio)


def test_ocupantes_lista_quien_tiene_abiertos_los_nodos_de_video_y_medios():
    ej = EjecutorFalso()
    ej.fallar["fuser"] = hw.Resultado(0, fixture("fuser_stdout_ocupado.txt"), fixture("fuser_stderr_ocupado.txt"))
    assert salud_con_fuser(ej).ocupantes() == [{"pid": 2780, "nombre": "v4l2-ctl"}]
    assert ej.llamadas[0] == ["fuser", "/dev/media4", "/dev/video1"]


def test_ocupantes_no_cuenta_nuestro_propio_proceso():
    ej = EjecutorFalso()
    ej.fallar["fuser"] = hw.Resultado(0, "  2780", "")
    assert salud_con_fuser(ej, pid_propio=2780).ocupantes() == []


def test_ocupantes_sin_nadie_usando_los_dispositivos():
    ej = EjecutorFalso()
    ej.fallar["fuser"] = hw.Resultado(1, "", "")             # fuser sale con 1 si no hay procesos
    assert salud_con_fuser(ej).ocupantes() == []


def test_ocupantes_sin_fuser_instalado_se_degrada_sin_error():
    ej = EjecutorFalso()
    ej.fallar["fuser"] = hw.Resultado(127, "", "no se encontro el programa 'fuser'")
    assert salud_con_fuser(ej).ocupantes() is None           # desconocido: no se pudo verificar


def test_ocupantes_sin_dispositivos_no_ejecuta_nada():
    ej = EjecutorFalso()
    salud = hw.Salud(ej, listar_dispositivos=lambda: [])
    assert salud.ocupantes() == [] and ej.llamadas == []


# --- hardware simulado (para ensayar sin la Pi) ---------------------------------------------------------------

def test_hardware_simulado_cumple_la_misma_interfaz(tmp_path):
    sim = hw.HardwareSimulado(CFG)
    assert sim.nir.presente() and sim.salud.reloj_sincronizado() is True
    with sim.luces.encendida("rgb"):
        info = sim.rgb.capturar(tmp_path / "r.jpg")
    with sim.luces.encendida("nir"):
        info_nir = sim.nir.capturar(tmp_path / "n.jpg", tmp_path / "t.json")
    assert Image.open(tmp_path / "r.jpg").size == sim.tam
    assert info["controles"] and info_nir["rpicam_metadata"]
    sim.luces.apagar_todo()
    assert sim.luces.encendidas == set()


def test_hardware_simulado_con_luz_apagada_produce_imagenes_oscuras(tmp_path):
    import numpy as np
    sim = hw.HardwareSimulado(CFG)
    sim.rgb.capturar(tmp_path / "oscura.jpg")      # sin encender la luz
    with sim.luces.encendida("rgb"):
        sim.rgb.capturar(tmp_path / "clara.jpg")
    oscura = np.asarray(Image.open(tmp_path / "oscura.jpg")).mean()
    clara = np.asarray(Image.open(tmp_path / "clara.jpg")).mean()
    assert clara > 5 * oscura


def test_el_simulador_escala_el_brillo_con_la_exposicion_para_poder_probar_la_autoverificacion(tmp_path):
    import numpy as np
    sim = hw.HardwareSimulado(CFG)
    with sim.luces.encendida("rgb"):
        sim.rgb.capturar(tmp_path / "alta.jpg", extra={"exposure_time_absolute": 65})
        sim.rgb.capturar(tmp_path / "baja.jpg", extra={"exposure_time_absolute": 21})
    alta = np.asarray(Image.open(tmp_path / "alta.jpg")).mean()
    baja = np.asarray(Image.open(tmp_path / "baja.jpg")).mean()
    assert alta / baja > 2.5


def test_el_simulador_puede_imitar_la_trampa_del_firmware_ignorando_la_exposicion(tmp_path):
    import numpy as np
    sim = hw.HardwareSimulado(CFG)
    sim.rgb.ignora_exposicion = True
    with sim.luces.encendida("rgb"):
        sim.rgb.capturar(tmp_path / "alta.jpg", extra={"exposure_time_absolute": 65})
        sim.rgb.capturar(tmp_path / "baja.jpg", extra={"exposure_time_absolute": 21})
    alta = np.asarray(Image.open(tmp_path / "alta.jpg")).mean()
    baja = np.asarray(Image.open(tmp_path / "baja.jpg")).mean()
    assert alta / baja == pytest.approx(1.0, abs=0.05)


def test_el_simulador_puede_imitar_la_noir_trabada(tmp_path):
    sim = hw.HardwareSimulado(CFG)
    sim.nir.trabada_simulada = True
    with pytest.raises(hw.NIRTrabada):
        sim.nir.capturar(tmp_path / "n.jpg", tmp_path / "t.json")
    with pytest.raises(hw.NIRTrabada):
        sim.nir.probar(tmp_path)


def test_el_simulador_informa_ocupantes_de_los_dispositivos():
    sim = hw.HardwareSimulado(CFG)
    assert sim.salud.ocupantes() == []
    sim.salud.ocupantes_simulados = [{"pid": 7, "nombre": "wireplumber"}]
    assert sim.salud.ocupantes() == [{"pid": 7, "nombre": "wireplumber"}]
