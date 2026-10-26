"""Interfaz de linea de comandos de captura.py (argumentos y recorrido completo simulado)."""
import json
from datetime import datetime, timedelta, timezone

import pytest

import captura
import estado as e
import subida

LIMA = timezone(timedelta(hours=-5))
SECRETO = "https://objectstorage.sa-test.oraclecloud.com/p/SECRETO123/n/ns/b/bucket/o/"


# --- argumentos ------------------------------------------------------------------------------------------

def test_jornada_con_sus_opciones_y_defectos_del_protocolo():
    a = captura.parse_args(["jornada", "--id", "J01", "--lote", "lote-A", "--mangos", "1-32"])
    assert a.comando == "jornada" and a.id == "J01" and a.lote == "lote-A" and a.mangos == "1-32"
    assert a.intervalos == "0,2,4,6,8,10,12"
    assert a.slot_min == 3.0 and a.tolerancia_min == 10
    assert a.temperatura_c is None and a.no_subir is False and a.auto is False
    assert a.caras == "1-4" and a.unidad_min == 60


def test_jornada_acepta_todas_las_opciones_de_la_especificacion():
    a = captura.parse_args(["jornada", "--id", "J02", "--lote", "L", "--mangos", "33-64",
                            "--temperatura-c", "24.5", "--intervalos", "1,2", "--slot-min", "3",
                            "--tolerancia-min", "5", "--no-subir"])
    assert a.temperatura_c == 24.5 and a.intervalos == "1,2" and a.slot_min == 3
    assert a.tolerancia_min == 5 and a.no_subir is True


def test_la_ayuda_de_jornada_nombra_los_defectos_con_la_ronda_cero_y_el_slot_de_3_min(capsys):
    with pytest.raises(SystemExit):
        captura.parse_args(["jornada", "--help"])
    ayuda = " ".join(capsys.readouterr().out.split())
    assert "por defecto 0,2,4,6,8,10,12" in ayuda and "0 = foto justo tras el impacto" in ayuda
    assert "por defecto 3)" in ayuda and "2.5" not in ayuda


def test_la_ayuda_de_impactos_dice_que_toma_las_fotos_de_la_ronda_cero(capsys):
    with pytest.raises(SystemExit):
        captura.parse_args(["impactos", "--help"])
    assert "+0 h" in " ".join(capsys.readouterr().out.split())


def test_opciones_de_prueba_ocultas_unidad_min_caras_y_semilla():
    a = captura.parse_args(["jornada", "--id", "P", "--lote", "L", "--mangos", "1-2",
                            "--unidad-min", "1", "--caras", "1,2", "--semilla", "7"])
    assert a.unidad_min == 1 and a.caras == "1,2" and a.semilla == 7


def test_las_opciones_ocultas_no_aparecen_en_la_ayuda(capsys):
    with pytest.raises(SystemExit):
        captura.parse_args(["jornada", "--help"])
    ayuda = capsys.readouterr().out
    assert "--unidad-min" not in ayuda and "--simular" not in ayuda
    assert "--mangos" in ayuda and "--tolerancia-min" in ayuda


def test_jornada_exige_id_lote_y_mangos():
    with pytest.raises(SystemExit) as exc:
        captura.parse_args(["jornada", "--id", "J01"])
    assert exc.value.code == 2


@pytest.mark.parametrize("argv", [["--auto", "sanas"], ["sanas", "--auto"]])
def test_auto_vale_antes_y_despues_del_subcomando(argv):
    assert captura.parse_args(argv).auto is True


@pytest.mark.parametrize("argv", [["--no-subir", "sanas"], ["sanas", "--no-subir"]])
def test_no_subir_vale_antes_y_despues_del_subcomando(argv):
    assert captura.parse_args(argv).no_subir is True


def test_sin_flags_globales_quedan_apagados():
    a = captura.parse_args(["sanas"])
    assert a.auto is False and a.no_subir is False


def test_ronda_con_forzar():
    assert captura.parse_args(["ronda", "--forzar"]).forzar is True
    assert captura.parse_args(["ronda"]).forzar is False


def test_los_subcomandos_de_la_especificacion_existen():
    for sub in ("jornada", "sanas", "impactos", "ronda", "estado", "subir", "enfoque"):
        extra = ["--id", "J", "--lote", "L", "--mangos", "1-2"] if sub == "jornada" else []
        assert captura.parse_args([sub] + extra).comando == sub


def test_enfoque_elige_la_camara():
    assert captura.parse_args(["enfoque"]).camara == "rgb"
    assert captura.parse_args(["enfoque", "--camara", "nir"]).camara == "nir"
    with pytest.raises(SystemExit):
        captura.parse_args(["enfoque", "--camara", "uv"])


def test_id_opcional_para_elegir_otra_jornada():
    assert captura.parse_args(["estado", "--id", "J05"]).id == "J05"
    assert captura.parse_args(["estado"]).id is None


def test_sin_subcomando_es_un_error():
    with pytest.raises(SystemExit):
        captura.parse_args([])


# --- recorrido completo con hardware simulado ----------------------------------------------------------------

class Entorno:
    """Reloj y esperas falsos para recorrer el protocolo completo sin esperar horas."""

    def __init__(self, tmp_path):
        self.t = datetime(2026, 10, 22, 8, 30, 0, tzinfo=LIMA)
        self.salida = []
        self.puts = []
        self.base = str(tmp_path / "capturas")

    def reloj(self):
        return self.t

    def dormir(self, s):
        self.t += timedelta(seconds=s)

    def put(self, url, ruta):
        self.puts.append((url, ruta))
        return 200

    def dic(self, **extra):
        d = dict(reloj=self.reloj, dormir=self.dormir, salida=self.salida.append, put=self.put,
                 env={}, ruta_par=None)
        d.update(extra)
        return d

    def correr(self, *argv, **extra):
        return captura.main(["--base", self.base, "--simular", "--auto", *argv], entorno=self.dic(**extra))


JORNADA = ["jornada", "--id", "J01", "--lote", "lote-A", "--mangos", "1-2", "--caras", "1,2",
           "--intervalos", "1,2", "--unidad-min", "1", "--slot-min", "0.5", "--tolerancia-min", "0.5",
           "--semilla", "3", "--no-subir"]


def test_recorrido_completo_jornada_sanas_impactos_ronda_estado(tmp_path):
    env = Entorno(tmp_path)
    assert env.correr(*JORNADA) == 0
    assert env.correr("sanas") == 0
    assert env.correr("impactos") == 0
    assert env.correr("ronda") == 0
    assert env.correr("estado") == 0
    j = e.Jornada.cargar(tmp_path / "capturas" / "J01")
    assert j.pendientes_sanas() == [] and j.pendientes_impactos() == []
    assert j.hechos() == {(1, 1), (2, 1), (1, 2), (2, 2)}
    texto = "\n".join(env.salida)
    assert "Siguiente paso" in texto and "No queda nada pendiente" in texto and "Jornada J01" in texto
    metas = list((tmp_path / "capturas" / "J01" / "news" / "meta").glob("*.json"))
    assert len(metas) == 4 + 8
    assert json.loads(metas[0].read_text())["schema"] == "3.0"


JORNADA_CERO = ["jornada", "--id", "J01", "--lote", "lote-A", "--mangos", "1-2", "--caras", "1,2",
                "--intervalos", "0,1", "--unidad-min", "1", "--slot-min", "0.5", "--tolerancia-min", "0.5",
                "--semilla", "3", "--no-subir"]


def metas_de(tmp_path):
    return [json.loads(p.read_text()) for p in (tmp_path / "capturas" / "J01" / "news" / "meta").glob("*.json")]


def test_jornada_acepta_el_cero_en_los_intervalos(tmp_path):
    env = Entorno(tmp_path)
    assert env.correr(*JORNADA_CERO) == 0
    assert e.Jornada.cargar(tmp_path / "capturas" / "J01").intervalos_h == [0, 1]


def test_jornada_con_los_intervalos_por_defecto_incluye_el_cero(tmp_path):
    env = Entorno(tmp_path)
    i = JORNADA_CERO.index("--intervalos")
    assert env.correr(*JORNADA_CERO[:i], *JORNADA_CERO[i + 2:]) == 0                  # sin --intervalos 0,1
    assert e.Jornada.cargar(tmp_path / "capturas" / "J01").intervalos_h == [0, 2, 4, 6, 8, 10, 12]


def test_recorrido_con_ronda_cero_impactos_toma_las_fotos_y_ronda_no_las_repite(tmp_path):
    env = Entorno(tmp_path)
    assert env.correr(*JORNADA_CERO) == 0
    assert env.correr("sanas") == 0
    assert env.correr("impactos") == 0
    ceros = [m for m in metas_de(tmp_path) if m["fase"] == "post_impacto"]
    assert sorted((m["mango_id"], m["cara"], m["ronda_h"]) for m in ceros) == [
        (1, 1, 0), (1, 2, 0), (2, 1, 0), (2, 2, 0)]
    assert env.correr("ronda") == 0
    posts = [(m["mango_id"], m["cara"], m["ronda_h"]) for m in metas_de(tmp_path) if m["fase"] == "post_impacto"]
    assert len(posts) == len(set(posts)) == 8 and {r for _, _, r in posts} == {0, 1}      # el +0 h no se repite
    assert e.Jornada.cargar(tmp_path / "capturas" / "J01").hechos() == {(1, 0), (2, 0), (1, 1), (2, 1)}
    assert env.correr("estado") == 0 and any("+0 h" in linea for linea in env.salida)


def test_impactos_con_ronda_cero_sale_con_codigo_2_si_la_autoverificacion_no_se_hizo(tmp_path):
    env = Entorno(tmp_path)
    assert env.correr(*JORNADA_CERO) == 0
    j = e.Jornada.cargar(tmp_path / "capturas" / "J01")
    j.datos.pop("autoverificacion_exposicion")
    j.guardar()
    assert env.correr("impactos") == 2
    assert any("autoverificación" in linea for linea in env.salida)


def test_impactos_con_ronda_cero_construye_el_hardware_real_y_sin_ronda_cero_no(tmp_path, monkeypatch):
    import hardware
    construidos = []

    def fabrica(config):
        construidos.append(config)
        return hardware.HardwareSimulado(config)

    monkeypatch.setattr(hardware, "Hardware", fabrica)
    env = Entorno(tmp_path)
    assert captura.main(["--base", env.base, "--simular", "--auto", *JORNADA_CERO], entorno=env.dic()) == 0
    assert captura.main(["--base", env.base, "--simular", "--auto", "sanas"], entorno=env.dic()) == 0
    assert construidos == []
    assert captura.main(["--base", env.base, "--auto", "impactos"], entorno=env.dic()) == 0
    assert len(construidos) == 1                                            # con +0 h necesita cámaras y luces
    assert len([m for m in metas_de(tmp_path) if m["fase"] == "post_impacto"]) == 4


def test_impactos_sin_ronda_cero_no_construye_el_hardware(tmp_path, monkeypatch):
    import hardware
    env = Entorno(tmp_path)
    assert env.correr(*JORNADA) == 0
    assert env.correr("sanas") == 0

    def no_debe_construirse(config):
        raise AssertionError("impactos sin ronda +0 h no usa cámaras ni luces")

    monkeypatch.setattr(hardware, "Hardware", no_debe_construirse)
    assert captura.main(["--base", env.base, "--auto", "impactos"], entorno=env.dic()) == 0
    assert e.Jornada.cargar(tmp_path / "capturas" / "J01").pendientes_impactos() == []


def test_la_configuracion_de_camaras_se_crea_en_la_primera_ejecucion(tmp_path):
    env = Entorno(tmp_path)
    assert env.correr(*JORNADA) == 0
    assert (tmp_path / "capturas" / "config_camaras.json").exists()


def test_error_de_preparacion_sale_con_codigo_2_y_mensaje(tmp_path):
    env = Entorno(tmp_path)
    assert env.correr("sanas") == 2
    assert any("jornada" in linea.lower() and linea.startswith("✗") for linea in env.salida)


def test_abort_por_luz_apagada_en_auto_sale_con_codigo_3(tmp_path):
    env = Entorno(tmp_path)
    assert env.correr(*JORNADA) == 0

    class BancoDormido:
        def __call__(self, hw_):
            hw_.luces.banco_dormido = True

    assert env.correr("sanas", preparar_hw=BancoDormido()) == 3
    assert any("luz apagada?" in linea for linea in env.salida)


def test_subir_sin_url_sale_con_codigo_2_y_dice_donde_ponerla(tmp_path):
    env = Entorno(tmp_path)
    assert env.correr(*JORNADA) == 0
    assert env.correr("subir", env={}, ruta_par=tmp_path / "no_existe") == 2
    assert any("OCI_PAR_URL" in linea for linea in env.salida)


def test_subir_con_la_url_del_entorno_sube_y_no_la_imprime(tmp_path):
    env = Entorno(tmp_path)
    assert env.correr(*JORNADA) == 0
    assert env.correr("sanas") == 0
    assert env.correr("subir", env={"OCI_PAR_URL": SECRETO}) == 0
    assert len(env.puts) == 10 + 12                      # calibracion + 4 caras x (rgb, nir, meta)
    assert all(u.startswith(SECRETO) for u, _ in env.puts)
    assert "SECRETO123" not in "\n".join(env.salida)


def test_la_url_con_permisos_abiertos_avisa_sin_mostrarla(tmp_path):
    import os
    env = Entorno(tmp_path)
    assert env.correr(*JORNADA) == 0
    archivo = tmp_path / "oci_par_url"
    archivo.write_text(SECRETO)
    os.chmod(archivo, 0o644)
    assert env.correr("subir", env={}, ruta_par=archivo) == 0
    texto = "\n".join(env.salida)
    assert "chmod 600" in texto and "SECRETO123" not in texto


def test_ctrl_c_deja_el_avance_guardado_y_sale_con_130(tmp_path):
    env = Entorno(tmp_path)
    assert env.correr(*JORNADA) == 0

    def dormir_interrumpido(s):
        raise KeyboardInterrupt

    assert env.correr("sanas", preparar_hw=lambda hw_: setattr(hw_.rgb, "capturar", dormir_interrumpido)) == 130
    assert any("avance" in linea.lower() for linea in env.salida)


def test_estado_imprime_sin_necesitar_hardware(tmp_path):
    env = Entorno(tmp_path)
    assert env.correr(*JORNADA) == 0
    assert captura.main(["--base", env.base, "estado"], entorno=env.dic()) == 0     # sin --simular ni camaras
    assert any("Jornada J01" in linea for linea in env.salida)


def test_noir_trabada_sale_con_codigo_4_y_el_mensaje_de_reinicio(tmp_path):
    env = Entorno(tmp_path)
    assert env.correr(*JORNADA, preparar_hw=lambda hw_: setattr(hw_.nir, "trabada_simulada", True)) == 4
    assert any("la NoIR quedó trabada: reiniciá la Pi (sudo reboot) y retomá con el mismo comando; "
               "el estado se conserva" in linea for linea in env.salida)


def test_un_error_de_camara_en_el_enfoque_sale_con_codigo_2_sin_traceback(tmp_path):
    env = Entorno(tmp_path)

    def falla(*a, **k):
        raise hardware_error("rgb: no se pudieron leer los controles V4L2: Cannot open device")

    from hardware import ErrorCamara as hardware_error
    assert env.correr("enfoque", "--camara", "rgb", preparar_hw=lambda hw_: setattr(hw_.rgb, "capturar", falla)) == 2
    assert any("Cannot open device" in linea for linea in env.salida)


# --- reiniciar ----------------------------------------------------------------------------------------------

def test_reiniciar_exige_una_etapa_valida():
    assert captura.parse_args(["reiniciar", "--desde", "sanas"]).desde == "sanas"
    assert captura.parse_args(["reiniciar", "--desde", "vacia", "--forzar"]).forzar is True
    for argv in (["reiniciar"], ["reiniciar", "--desde", "rondas"]):
        with pytest.raises(SystemExit):
            captura.parse_args(argv)


def carpeta_j01(tmp_path):
    return tmp_path / "capturas" / "J01"


def reinicios(tmp_path):
    return list((carpeta_j01(tmp_path) / "descartes").glob("reinicio-*"))


def test_reiniciar_desde_sanas_aparta_las_fotos_y_las_sanas_se_repiten_en_la_misma_jornada(tmp_path):
    env = Entorno(tmp_path)
    assert env.correr(*JORNADA) == 0
    assert env.correr("sanas") == 0
    viejas = {p.name for p in (carpeta_j01(tmp_path) / "news" / "meta").glob("*.json")}
    assert env.correr("reiniciar", "--desde", "sanas") == 0
    j = e.Jornada.cargar(carpeta_j01(tmp_path))
    assert len(j.pendientes_sanas()) == 4 and j.calibracion_hecha("vacia")
    assert not list((carpeta_j01(tmp_path) / "news").rglob("*.*"))
    (apartados,) = reinicios(tmp_path)
    assert {p.name for p in (apartados / "news" / "meta").glob("*.json")} == viejas
    assert len(list(apartados.rglob("*.jpg"))) == 8 and len(list((carpeta_j01(tmp_path) / "calib").glob("*"))) == 10
    assert j.datos["reinicios"][0]["desde"] == "sanas" and j.datos["reinicios"][0]["archivos"] == 12
    assert any("Siguiente paso: captura.py sanas" in linea for linea in env.salida)
    assert env.correr("sanas") == 0
    nuevas = {p.name for p in (carpeta_j01(tmp_path) / "news" / "meta").glob("*.json")}
    assert len(nuevas) == 4 and not nuevas & viejas      # timestamps nuevos: no pisa lo apartado


def test_reiniciar_desde_vacia_y_jornada_retoma_solo_la_bandeja_vacia(tmp_path):
    env = Entorno(tmp_path)
    assert env.correr(*JORNADA) == 0
    assert env.correr("sanas") == 0
    calib = carpeta_j01(tmp_path) / "calib"
    vacia_vieja = {p.name for p in calib.glob("calib-vacia-*")}
    assert env.correr("reiniciar", "--desde", "vacia") == 0
    assert not list(calib.glob("calib-vacia-*")) and len(list(calib.glob("calib-gris-*"))) == 3
    assert env.correr("sanas") == 2                                  # sin bandeja vacia no hay QC
    assert any("Falta la bandeja vacía" in linea for linea in env.salida)
    assert env.correr(*JORNADA) == 0
    assert len(list(calib.glob("calib-vacia-*"))) == 3 and not {p.name for p in calib.glob("calib-vacia-*")} & vacia_vieja
    assert len(list(calib.glob("calib-gris-*"))) == 3 and len(list(calib.glob("calib-aruco-*"))) == 3
    j = e.Jornada.cargar(carpeta_j01(tmp_path))
    assert j.referencia("nir") is not None and len(j.pendientes_sanas()) == 4
    assert env.correr("sanas") == 0


def test_reiniciar_con_impactos_exige_forzar(tmp_path):
    env = Entorno(tmp_path)
    assert env.correr(*JORNADA) == 0
    assert env.correr("sanas") == 0
    assert env.correr("impactos") == 0
    assert env.correr("reiniciar", "--desde", "sanas") == 2
    assert any("--forzar" in linea for linea in env.salida) and not reinicios(tmp_path)
    assert e.Jornada.cargar(carpeta_j01(tmp_path)).tiene_impactos()
    assert env.correr("reiniciar", "--desde", "impactos", "--forzar") == 0
    j = e.Jornada.cargar(carpeta_j01(tmp_path))
    assert not j.tiene_impactos() and j.pendientes_sanas() == []


def test_reiniciar_desde_impactos_aparta_solo_las_fotos_posteriores_al_impacto(tmp_path):
    env = Entorno(tmp_path)
    assert env.correr(*JORNADA_CERO) == 0
    assert env.correr("sanas") == 0
    assert env.correr("impactos") == 0
    assert env.correr("reiniciar", "--desde", "impactos", "--forzar") == 0
    assert sorted(m["fase"] for m in metas_de(tmp_path)) == ["sano"] * 4
    (apartados,) = reinicios(tmp_path)
    fases = [json.loads(p.read_text())["fase"] for p in (apartados / "news" / "meta").glob("*.json")]
    assert fases == ["post_impacto"] * 4
    assert env.correr("impactos") == 0 and len(metas_de(tmp_path)) == 8


def test_reiniciar_marca_perdido_lo_pendiente_y_avisa_lo_ya_subido(tmp_path):
    env = Entorno(tmp_path)
    assert env.correr(*JORNADA) == 0
    assert env.correr("subir", env={"OCI_PAR_URL": SECRETO}) == 0      # sube la calibracion (10)
    assert env.correr("sanas") == 0
    cola = subida.ColaSubida(carpeta_j01(tmp_path) / "pendientes.jsonl")
    subida.encolar_faltantes(cola, carpeta_j01(tmp_path))              # 12 de las sanas, pendientes
    assert env.correr("reiniciar", "--desde", "vacia") == 0
    resumen = subida.ColaSubida(carpeta_j01(tmp_path) / "pendientes.jsonl").resumen()
    assert resumen == {"pendientes": 0, "subidos": 10, "perdidos": 12}
    assert any("3 de esos archivos ya estaban subidos" in linea for linea in env.salida)
    env.puts.clear()
    assert env.correr("subir", env={"OCI_PAR_URL": SECRETO}) == 0
    assert env.puts == [] and "SECRETO123" not in "\n".join(env.salida)


def test_reiniciar_cancelado_con_q_no_toca_nada(tmp_path):
    env = Entorno(tmp_path)
    assert env.correr(*JORNADA) == 0
    assert env.correr("sanas") == 0
    antes = (carpeta_j01(tmp_path) / e.NOMBRE_ESTADO).read_text()
    codigo = captura.main(["--base", env.base, "reiniciar", "--desde", "sanas"],
                          entorno=env.dic(entrada=lambda texto: "q"))
    assert codigo == 0 and any("Cancelado" in linea for linea in env.salida)
    assert (carpeta_j01(tmp_path) / e.NOMBRE_ESTADO).read_text() == antes and not reinicios(tmp_path)
