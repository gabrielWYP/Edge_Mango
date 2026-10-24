"""Subida a OCI: URL pre-autenticada (nunca expuesta), cola persistente y reintentos."""
import logging
import os
import threading
import time

import pytest

import subida as s

SECRETO = "https://objectstorage.sa-test.oraclecloud.com/p/SECRETO123/n/ns/b/bucket/o/"


# --- carga de la URL pre-autenticada -----------------------------------------------------------

def test_la_variable_de_entorno_tiene_prioridad_sobre_el_archivo(tmp_path):
    archivo = tmp_path / "oci_par_url"
    archivo.write_text("https://otra/o/\n")
    os.chmod(archivo, 0o600)
    par, avisos = s.cargar_par_url(env={"OCI_PAR_URL": SECRETO}, ruta_archivo=archivo)
    assert par.valor == SECRETO and avisos == []


def test_si_no_hay_variable_se_lee_el_archivo_y_se_limpian_los_espacios(tmp_path):
    archivo = tmp_path / "oci_par_url"
    archivo.write_text(f"  {SECRETO}\n\n")
    os.chmod(archivo, 0o600)
    par, avisos = s.cargar_par_url(env={}, ruta_archivo=archivo)
    assert par.valor == SECRETO and avisos == []


def test_archivo_con_permisos_abiertos_da_aviso_sin_mostrar_la_url(tmp_path):
    archivo = tmp_path / "oci_par_url"
    archivo.write_text(SECRETO)
    os.chmod(archivo, 0o644)
    par, avisos = s.cargar_par_url(env={}, ruta_archivo=archivo)
    assert par.valor == SECRETO
    assert len(avisos) == 1 and "chmod 600" in avisos[0]
    assert "SECRETO123" not in avisos[0]


def test_la_url_se_normaliza_para_terminar_en_barra(tmp_path):
    par, _ = s.cargar_par_url(env={"OCI_PAR_URL": SECRETO.rstrip("/")}, ruta_archivo=tmp_path / "no")
    assert par.valor.endswith("/o/")


def test_sin_url_configurada_el_error_dice_donde_ponerla(tmp_path):
    ruta = tmp_path / "oci_par_url"
    with pytest.raises(s.ParNoConfigurada) as exc:
        s.cargar_par_url(env={}, ruta_archivo=ruta)
    assert "OCI_PAR_URL" in str(exc.value) and str(ruta) in str(exc.value)


def test_variable_vacia_cuenta_como_ausente(tmp_path):
    with pytest.raises(s.ParNoConfigurada):
        s.cargar_par_url(env={"OCI_PAR_URL": "  "}, ruta_archivo=tmp_path / "no")


def test_la_url_nunca_aparece_en_str_repr_ni_logs(tmp_path, caplog, capsys):
    par, _ = s.cargar_par_url(env={"OCI_PAR_URL": SECRETO}, ruta_archivo=tmp_path / "no")
    with caplog.at_level(logging.DEBUG):
        logging.getLogger("x").info("par=%s %r", par, par)
    assert "SECRETO123" not in str(par) and "SECRETO123" not in repr(par)
    assert "SECRETO123" not in caplog.text
    assert "SECRETO123" not in capsys.readouterr().out


def test_la_ruta_por_defecto_del_archivo():
    assert str(s.RUTA_PAR_DEFECTO).endswith(".config/captura/oci_par_url")


# --- destinos -------------------------------------------------------------------------------------

def test_destino_replica_el_layout_del_bucket(tmp_path):
    raiz = tmp_path / "J01"
    assert s.destino_de(raiz / "news" / "rgb" / "mango_rgb_1.jpg", raiz) == "news/rgb/mango_rgb_1.jpg"
    assert s.destino_de(raiz / "news" / "nir_raw" / "mango_nir_1.dng", raiz) == "news/nir_raw/mango_nir_1.dng"
    assert s.destino_de(raiz / "news" / "meta" / "mango_meta_1.json", raiz) == "news/meta/mango_meta_1.json"
    assert s.destino_de(raiz / "calib" / "calib-gris-rgb-1.jpg", raiz) == "calib/calib-gris-rgb-1.jpg"


# --- cola persistente --------------------------------------------------------------------------------

def test_la_cola_se_persiste_y_se_recarga(tmp_path):
    ruta = tmp_path / "pendientes.jsonl"
    c = s.ColaSubida(ruta)
    assert c.encolar("/a/x.jpg", "news/rgb/x.jpg")
    assert c.encolar("/a/y.jpg", "news/nir/y.jpg")
    c2 = s.ColaSubida(ruta)
    assert [p["destino"] for p in c2.pendientes()] == ["news/rgb/x.jpg", "news/nir/y.jpg"]


def test_no_se_encola_dos_veces_el_mismo_destino(tmp_path):
    c = s.ColaSubida(tmp_path / "p.jsonl")
    assert c.encolar("/a/x.jpg", "news/rgb/x.jpg")
    assert not c.encolar("/a/x.jpg", "news/rgb/x.jpg")
    assert len(c.pendientes()) == 1


def test_lo_subido_no_vuelve_a_pendientes_tras_recargar(tmp_path):
    ruta = tmp_path / "p.jsonl"
    c = s.ColaSubida(ruta)
    c.encolar("/a/x.jpg", "news/rgb/x.jpg")
    c.encolar("/a/y.jpg", "news/nir/y.jpg")
    c.marcar_hecho("news/rgb/x.jpg")
    c2 = s.ColaSubida(ruta)
    assert [p["destino"] for p in c2.pendientes()] == ["news/nir/y.jpg"]
    assert not c2.encolar("/a/x.jpg", "news/rgb/x.jpg")  # ya subido
    assert c2.resumen() == {"pendientes": 1, "subidos": 1, "perdidos": 0}


def test_una_ultima_linea_cortada_por_una_caida_se_ignora(tmp_path):
    ruta = tmp_path / "p.jsonl"
    c = s.ColaSubida(ruta)
    c.encolar("/a/x.jpg", "news/rgb/x.jpg")
    with open(ruta, "a") as f:
        f.write('{"op": "add", "destino": "news/rgb/z.j')   # escritura a medias
    c2 = s.ColaSubida(ruta)
    assert [p["destino"] for p in c2.pendientes()] == ["news/rgb/x.jpg"]
    assert c2.encolar("/a/w.jpg", "news/rgb/w.jpg")        # y se puede seguir escribiendo
    assert len(s.ColaSubida(ruta).pendientes()) == 2


def test_encolar_faltantes_recorre_la_carpeta_de_la_jornada(tmp_path):
    raiz = tmp_path / "J01"
    for rel in ("news/rgb/mango_rgb_1.jpg", "news/nir/mango_nir_1.jpg", "news/meta/mango_meta_1.json",
                "calib/calib-vacia-rgb-0.jpg", "descartes/mala.jpg", "jornada.json"):
        (raiz / rel).parent.mkdir(parents=True, exist_ok=True)
        (raiz / rel).write_bytes(b"x")
    (raiz / "news" / "rgb" / "parcial.jpg.tmp").write_bytes(b"x")
    c = s.ColaSubida(raiz / "pendientes.jsonl")
    n = s.encolar_faltantes(c, raiz)
    assert n == 4
    destinos = {p["destino"] for p in c.pendientes()}
    assert destinos == {"news/rgb/mango_rgb_1.jpg", "news/nir/mango_nir_1.jpg",
                        "news/meta/mango_meta_1.json", "calib/calib-vacia-rgb-0.jpg"}
    assert s.encolar_faltantes(c, raiz) == 0   # idempotente


# --- subida con reintentos (HTTP simulado) -------------------------------------------------------------

class PutSimulado:
    def __init__(self, respuestas):
        self.respuestas = list(respuestas)
        self.llamadas = []

    def __call__(self, url, ruta):
        self.llamadas.append((url, ruta))
        r = self.respuestas.pop(0) if self.respuestas else 200
        if isinstance(r, Exception):
            raise r
        return r


def cola_con(tmp_path, *nombres):
    c = s.ColaSubida(tmp_path / "p.jsonl")
    for n in nombres:
        archivo = tmp_path / n
        archivo.write_bytes(b"datos")
        c.encolar(str(archivo), f"news/rgb/{n}")
    return c


def par_de_prueba():
    return s.ParUrl(SECRETO)


def test_procesar_sube_todo_y_construye_la_url_con_el_destino(tmp_path):
    c = cola_con(tmp_path, "a.jpg", "b.jpg")
    put = PutSimulado([200, 200])
    r = s.procesar(c, par_de_prueba(), put, esperar=lambda seg: None)
    assert r == {"subidos": 2, "pendientes": 0, "perdidos": 0, "errores": []}
    assert put.llamadas[0][0] == SECRETO + "news/rgb/a.jpg"
    assert c.pendientes() == []


def test_procesar_reintenta_con_espera_creciente_y_termina_subiendo(tmp_path):
    c = cola_con(tmp_path, "a.jpg")
    put = PutSimulado([500, ConnectionError("sin red"), 200])
    esperas = []
    r = s.procesar(c, par_de_prueba(), put, max_intentos=3, esperar=esperas.append)
    assert r["subidos"] == 1 and r["pendientes"] == 0
    assert len(put.llamadas) == 3
    assert len(esperas) == 2 and esperas[1] > esperas[0]


def test_procesar_agotados_los_intentos_deja_el_archivo_pendiente(tmp_path):
    c = cola_con(tmp_path, "a.jpg", "b.jpg")
    put = PutSimulado([503, 503, 503, 200])
    r = s.procesar(c, par_de_prueba(), put, max_intentos=3, esperar=lambda seg: None)
    assert r["subidos"] == 1 and r["pendientes"] == 1
    assert [p["destino"] for p in s.ColaSubida(tmp_path / "p.jsonl").pendientes()] == ["news/rgb/a.jpg"]


def test_los_errores_no_filtran_la_url(tmp_path):
    c = cola_con(tmp_path, "a.jpg")
    put = PutSimulado([ConnectionError(f"HTTPSConnectionPool: {SECRETO}news/rgb/a.jpg timeout")] * 2)
    r = s.procesar(c, par_de_prueba(), put, max_intentos=2, esperar=lambda seg: None)
    texto = " ".join(r["errores"]) + str(c.pendientes())
    assert "SECRETO123" not in texto and "oraclecloud" not in texto
    assert r["errores"]


def test_archivo_local_desaparecido_se_marca_perdido_sin_reintentar(tmp_path):
    c = cola_con(tmp_path, "a.jpg")
    (tmp_path / "a.jpg").unlink()
    put = PutSimulado([])
    r = s.procesar(c, par_de_prueba(), put, esperar=lambda seg: None)
    assert r["perdidos"] == 1 and put.llamadas == []
    assert c.pendientes() == []


def test_put_requests_usa_la_biblioteca_y_devuelve_el_codigo(tmp_path, monkeypatch):
    llamado = {}

    class Resp:
        status_code = 200

    def falso_put(url, data=None, timeout=None):
        llamado.update(url=url, datos=data.read(), timeout=timeout)
        return Resp()

    import requests
    monkeypatch.setattr(requests, "put", falso_put)
    archivo = tmp_path / "x.jpg"
    archivo.write_bytes(b"contenido")
    assert s.put_requests(SECRETO + "news/rgb/x.jpg", str(archivo)) == 200
    assert llamado["datos"] == b"contenido" and llamado["timeout"]


# --- hilo de fondo --------------------------------------------------------------------------------------

def esperar_hasta(cond, limite=5.0):
    fin = time.time() + limite
    while time.time() < fin:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_hilo_de_fondo_sube_lo_que_se_va_encolando(tmp_path):
    c = s.ColaSubida(tmp_path / "p.jsonl")
    put = PutSimulado([])
    sub = s.SubidorFondo(c, par_de_prueba(), put, intervalo_s=0.01, esperar=lambda seg: None)
    sub.iniciar()
    try:
        for n in ("a.jpg", "b.jpg", "c.jpg"):
            (tmp_path / n).write_bytes(b"x")
            c.encolar(str(tmp_path / n), f"news/rgb/{n}")
            sub.notificar()
        assert esperar_hasta(lambda: not c.pendientes())
    finally:
        sub.detener()
    assert len(put.llamadas) == 3


def test_hilo_de_fondo_reintenta_mas_tarde_si_falla(tmp_path):
    c = cola_con(tmp_path, "a.jpg")
    put = PutSimulado([503, 503, 200])
    sub = s.SubidorFondo(c, par_de_prueba(), put, max_intentos=1, intervalo_s=0.01, esperar=lambda seg: None)
    sub.iniciar()
    try:
        assert esperar_hasta(lambda: not c.pendientes())
    finally:
        sub.detener()
    assert len(put.llamadas) == 3


def test_detener_con_vaciar_termina_lo_pendiente_antes_de_salir(tmp_path):
    c = cola_con(tmp_path, "a.jpg", "b.jpg")
    put = PutSimulado([])
    sub = s.SubidorFondo(c, par_de_prueba(), put, intervalo_s=60, esperar=lambda seg: None)
    sub.iniciar()
    sub.detener(vaciar=True, limite_s=5)
    assert c.pendientes() == []


def test_el_hilo_es_daemon_para_no_bloquear_la_salida(tmp_path):
    c = s.ColaSubida(tmp_path / "p.jsonl")
    sub = s.SubidorFondo(c, par_de_prueba(), PutSimulado([]), intervalo_s=60)
    sub.iniciar()
    try:
        assert all(t.daemon for t in threading.enumerate() if t.name == "subidor-fondo")
    finally:
        sub.detener()
