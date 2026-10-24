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
    assert a.intervalos == "2,4,6,8,10,12"
    assert a.slot_min == 2.5 and a.tolerancia_min == 10
    assert a.temperatura_c is None and a.no_subir is False and a.auto is False
    assert a.caras == "1-4" and a.unidad_min == 60


def test_jornada_acepta_todas_las_opciones_de_la_especificacion():
    a = captura.parse_args(["jornada", "--id", "J02", "--lote", "L", "--mangos", "33-64",
                            "--temperatura-c", "24.5", "--intervalos", "1,2", "--slot-min", "3",
                            "--tolerancia-min", "5", "--no-subir"])
    assert a.temperatura_c == 24.5 and a.intervalos == "1,2" and a.slot_min == 3
    assert a.tolerancia_min == 5 and a.no_subir is True


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
