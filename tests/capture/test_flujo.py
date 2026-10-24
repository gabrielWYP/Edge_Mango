"""Flujos completos (jornada, sanas, impactos, ronda, estado, subir, enfoque) con hardware simulado."""
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import configuracion
import consola
import estado as e
import flujo
import hardware as hw
import subida

LIMA = timezone(timedelta(hours=-5))
SECRETO = "https://objectstorage.sa-test.oraclecloud.com/p/SECRETO123/n/ns/b/bucket/o/"
PATRON_PIPELINE = re.compile(r"(\w+)_(rgb|nir)_(\d+)\.jpg")


class RelojFalso:
    """Reloj de pared controlable. `paso` avanza el reloj en cada lectura."""

    def __init__(self, inicio=None):
        self.t = inicio or datetime(2026, 10, 22, 8, 30, 0, tzinfo=LIMA)
        self.paso = timedelta(0)

    def __call__(self):
        actual = self.t
        self.t += self.paso
        return actual

    def avanzar(self, **kw):
        self.t += timedelta(**kw)

    def dormir(self, segundos):
        self.t += timedelta(seconds=segundos)


class PutSimulado:
    def __init__(self):
        self.llamadas = []

    def __call__(self, url, ruta):
        self.llamadas.append((url, ruta))
        return 200


def contexto(tmp_path, entradas=(), auto=True, subir=False, put=None, par=None, reloj=None, entrada=None):
    base = tmp_path / "capturas"
    config = configuracion.cargar_config(base)
    sim = hw.HardwareSimulado(config)
    reloj = reloj or RelojFalso()
    salida, prompts = [], []
    it = iter(entradas)

    def entrada_con_registro(texto):
        prompts.append(texto)                 # en modo interactivo el mensaje viaja por input()
        return entrada(texto) if entrada else next(it)

    ui = consola.Consola(entrada=entrada_con_registro, salida=salida.append,
                         salida_en_linea=lambda t: None, auto=auto, dormir=reloj.dormir)
    ctx = flujo.Contexto(base=base, config=config, hw=sim, ui=ui, reloj=reloj, dormir=reloj.dormir,
                         cronometro=lambda: 0.0, subir=subir, put=put, par=par)
    ctx.prompts = prompts
    return ctx, reloj, salida


def args_jornada(**kw):
    datos = dict(id="J01", lote="lote-A", mangos="1-2", temperatura_c=24.5, intervalos="1,2",
                 slot_min=0.5, tolerancia_min=0.5, no_subir=True, caras="1,2", unidad_min=1,
                 semilla=5, ignorar_ntp=False)
    datos.update(kw)
    return SimpleNamespace(**datos)


def args(**kw):
    datos = dict(id=None, forzar=False, camara="rgb", max_lecturas=None, valores="0,5,10")
    datos.update(kw)
    return SimpleNamespace(**datos)


def metas(raiz):
    return sorted((raiz / "news" / "meta").glob("mango_meta_*.json"))


def leer_meta(ruta):
    return json.loads(ruta.read_text(encoding="utf-8"))


def jornada_lista(tmp_path, **kw):
    """Jornada creada y calibrada en modo auto."""
    ctx, reloj, salida = contexto(tmp_path)
    flujo.cmd_jornada(ctx, args_jornada(**kw))
    return ctx, reloj, salida


# --- jornada -------------------------------------------------------------------------------------

def test_jornada_crea_el_estado_y_hace_la_calibracion_en_auto(tmp_path):
    ctx, _, _ = jornada_lista(tmp_path)
    raiz = ctx.base / "J01"
    assert (raiz / "jornada.json").exists()
    nombres = sorted(p.name for p in (raiz / "calib").iterdir())
    for tipo in ("vacia", "gris", "aruco"):
        for mod in ("rgb", "nir"):
            assert any(n.startswith(f"calib-{tipo}-{mod}-") and n.endswith(".jpg") for n in nombres), (tipo, mod)
        assert any(n.startswith(f"calib-{tipo}-meta-") and n.endswith(".json") for n in nombres)
    assert any(n.startswith("calib-wb-meta-") and n.endswith(".json") for n in nombres)   # barrido de balance
    assert not (raiz / "news").exists()                      # la calibracion nunca entra al dataset
    assert not any(PATRON_PIPELINE.search(n) for n in nombres)   # ni se empareja por el patron del pipeline
    j = e.Jornada.cargar(raiz)
    assert j.referencia("rgb")["borde_media"] == pytest.approx(100, abs=4)
    assert j.referencia("nir")["borde_media"] == pytest.approx(60, abs=4)
    assert e.leer_activa(ctx.base) == "J01"
    assert ctx.hw.luces.encendidas == set()
    assert not (raiz / "tmp").exists() or not list((raiz / "tmp").iterdir())


def test_la_calibracion_va_en_el_orden_gris_vacia_aruco_porque_el_balance_va_antes_de_la_referencia(tmp_path):
    ctx, _, _ = jornada_lista(tmp_path)
    ts = {t: e.Jornada.cargar(ctx.base / "J01").datos["calibracion"][t]["timestamp"]
          for t in ("gris", "vacia", "aruco")}
    assert ts["gris"] < ts["vacia"] < ts["aruco"]


def test_jornada_calibra_el_balance_de_blancos_y_lo_guarda_en_la_configuracion(tmp_path):
    ctx, _, salida = jornada_lista(tmp_path)
    j = e.Jornada.cargar(ctx.base / "J01")
    wb = j.balance_blancos()
    assert abs(wb["temperatura"] - 4500) <= 200          # la tarjeta simulada es neutra a 4500 K
    cfg = configuracion.cargar_config(ctx.base)
    assert cfg["rgb"]["controles"]["white_balance_temperature"] == wb["temperatura"]
    texto = "\n".join(salida)
    assert "white_balance_temperature=" in texto and "Balance de blancos" in texto
    nombre = next(p for p in (ctx.base / "J01" / "calib").iterdir() if p.name.startswith("calib-wb-meta-"))
    barrido = json.loads(nombre.read_text(encoding="utf-8"))
    assert barrido["mejor"] == wb["temperatura"] and len(barrido["muestras"]) >= 10
    assert {"temperatura", "r", "g", "b", "puntaje", "pct_recortado"} <= set(barrido["muestras"][0])


def test_el_barrido_de_balance_usa_tomas_rapidas_y_las_capturas_del_dataset_usan_el_valor_calibrado(tmp_path):
    ctx, _, _ = jornada_lista(tmp_path)
    barrido = [c for c in ctx.hw.rgb.historial if "white_balance_temperature" in c["extra"]]
    assert len(barrido) >= 10 and all(c["resolucion"] == "1600x1200" for c in barrido)
    wb = e.Jornada.cargar(ctx.base / "J01").balance_blancos()["temperatura"]
    flujo.cmd_sanas(ctx, args())
    meta = leer_meta(metas(ctx.base / "J01")[0])
    assert meta["rgb"]["controles_pedidos"]["white_balance_temperature"] == wb


def test_el_balance_se_mide_a_media_exposicion_y_el_dataset_usa_la_exposicion_completa(tmp_path):
    # El balance es un cociente entre canales y no depende de la exposicion. La exposicion de captura
    # es el techo para el mango, y a ese nivel una hoja blanca bajo la luz ya se recorta.
    ctx, _, _ = jornada_lista(tmp_path)
    e_captura = ctx.config["rgb"]["controles"]["exposure_time_absolute"]
    barrido = [c for c in ctx.hw.rgb.historial if "white_balance_temperature" in c["extra"]]
    assert {c["extra"]["exposure_time_absolute"] for c in barrido} == {e_captura // 2}
    meta_wb = leer_meta(next((ctx.base / "J01" / "calib").glob("calib-wb-meta-*.json")))
    assert meta_wb["exposicion_medicion"] == e_captura // 2
    flujo.cmd_sanas(ctx, args())
    meta = leer_meta(metas(ctx.base / "J01")[0])
    assert meta["rgb"]["controles"]["exposure_time_absolute"] == e_captura


def test_jornada_guia_la_calibracion_con_mensajes_coloca_x_y_presiona_enter(tmp_path):
    # Enter para: tarjeta gris, confirmar el recuadro, bandeja vacia, tablero ArUco
    ctx, _, salida = contexto(tmp_path, entradas=["", "", "", ""], auto=False)
    flujo.cmd_jornada(ctx, args_jornada())
    texto = "\n".join(ctx.prompts)
    assert "Coloca la tarjeta gris" in texto
    assert "Coloca la bandeja VACIA" in texto
    assert "Coloca el tablero ArUco" in texto
    assert "presiona Enter" in texto
    assert "recuadro" in texto.lower() and "confirmar" in texto.lower()
    assert any("vista previa" in linea.lower() for linea in salida)


def test_jornada_permite_reubicar_la_tarjeta_si_el_recuadro_no_esta_sobre_ella(tmp_path):
    ctx, _, _ = contexto(tmp_path, entradas=["", "n", "", "", "", ""], auto=False)
    flujo.cmd_jornada(ctx, args_jornada())
    texto = "\n".join(ctx.prompts)
    assert "Reubica" in texto
    assert e.Jornada.cargar(ctx.base / "J01").balance_blancos() is not None


def test_jornada_permite_omitir_la_tarjeta_gris_y_deja_el_balance_como_esta(tmp_path):
    ctx, _, salida = contexto(tmp_path, entradas=["s", "", ""], auto=False)
    flujo.cmd_jornada(ctx, args_jornada())
    nombres = [p.name for p in (ctx.base / "J01" / "calib").iterdir()]
    assert not any("gris" in n or "calib-wb" in n for n in nombres) and any("aruco" in n for n in nombres)
    assert e.Jornada.cargar(ctx.base / "J01").balance_blancos() is None
    assert any("balance de blancos" in linea.lower() and "sin calibrar" in linea.lower() for linea in salida)


def test_jornada_con_tarjeta_saturada_pide_bajar_la_exposicion(tmp_path, monkeypatch):
    import balance
    ctx, _, _ = contexto(tmp_path)
    monkeypatch.setattr(balance, "medir_archivo",
                        lambda ruta, frac=0.4, max_lado=640: {"r": 255.0, "g": 255.0, "b": 255.0, "pct_recortado": 95.0})
    with pytest.raises(flujo.ErrorPreparacion) as exc:
        flujo.cmd_jornada(ctx, args_jornada())
    assert "exposicion" in str(exc.value).lower()
    assert not e.Jornada.cargar(ctx.base / "J01").calibracion_hecha("gris")
    assert ctx.hw.luces.encendidas == set()


def test_jornada_guarda_zonas_y_semilla(tmp_path):
    ctx, _, salida = jornada_lista(tmp_path)
    j = e.Jornada.cargar(ctx.base / "J01")
    assert j.datos["semilla"] == 5 and j.datos["impacto"]["masa_bola_g"] == 66.0
    assert all(z in ("centro", "superior", "inferior", "izquierda", "derecha")
               for m in j.mangos for z in j.zonas_de(m).values())
    assert j.datos["unidad_min"] == 1 and j.datos["subir"] is False
    assert any("semilla" in linea.lower() for linea in salida)


def test_jornada_aborta_si_no_hay_camara_nir_y_no_crea_nada(tmp_path):
    ctx, _, _ = contexto(tmp_path)
    ctx.hw.nir.presente_simulada = False
    with pytest.raises(flujo.ErrorPreparacion) as exc:
        flujo.cmd_jornada(ctx, args_jornada())
    assert "NIR" in str(exc.value) and "cable" in str(exc.value)
    assert not (ctx.base / "J01").exists()


def test_jornada_aborta_si_el_reloj_no_esta_sincronizado(tmp_path):
    ctx, _, _ = contexto(tmp_path)
    ctx.hw.salud.reloj = False
    with pytest.raises(flujo.ErrorPreparacion) as exc:
        flujo.cmd_jornada(ctx, args_jornada())
    assert "reloj" in str(exc.value).lower()
    assert not (ctx.base / "J01").exists()


def test_jornada_con_ignorar_ntp_sigue_adelante_con_aviso(tmp_path):
    ctx, _, salida = contexto(tmp_path)
    ctx.hw.salud.reloj = False
    flujo.cmd_jornada(ctx, args_jornada(ignorar_ntp=True))
    assert (ctx.base / "J01" / "jornada.json").exists()
    assert any("reloj" in linea.lower() for linea in salida)


def test_jornada_existente_se_reanuda_sin_repetir_la_calibracion(tmp_path):
    ctx, _, _ = jornada_lista(tmp_path)
    antes = sorted(p.name for p in (ctx.base / "J01" / "calib").iterdir())
    ctx2, _, salida = contexto(tmp_path)
    flujo.cmd_jornada(ctx2, args_jornada())
    assert sorted(p.name for p in (ctx.base / "J01" / "calib").iterdir()) == antes
    assert any("ya existe" in linea.lower() for linea in salida)


def test_jornada_interrumpida_en_la_calibracion_continua_donde_quedo(tmp_path):
    ctx, _, salida = contexto(tmp_path, entradas=["", "", "q"], auto=False)
    flujo.cmd_jornada(ctx, args_jornada())       # hizo la gris (con el balance) y salio en la bandeja vacia
    assert any("avance" in linea.lower() and "guardado" in linea.lower() for linea in salida)
    j = e.Jornada.cargar(ctx.base / "J01")
    assert j.calibracion_hecha("gris") and not j.calibracion_hecha("vacia")
    wb = j.balance_blancos()
    ctx2, _, _ = contexto(tmp_path)
    flujo.cmd_jornada(ctx2, args_jornada())
    j = e.Jornada.cargar(ctx.base / "J01")
    assert all(j.calibracion_hecha(t) for t in ("vacia", "gris", "aruco"))
    assert j.balance_blancos() == wb             # no se repitio el barrido de balance
    assert len([p for p in (ctx.base / "J01" / "calib").iterdir() if p.name.startswith("calib-gris-rgb")]) == 1


# --- sanas ------------------------------------------------------------------------------------------

def test_sanas_captura_todo_con_nombres_del_pipeline_y_timestamps_unicos(tmp_path):
    ctx, reloj, _ = jornada_lista(tmp_path)
    flujo.cmd_sanas(ctx, args())
    raiz = ctx.base / "J01"
    rgb = sorted((raiz / "news" / "rgb").glob("*.jpg"))
    nir = sorted((raiz / "news" / "nir").glob("*.jpg"))
    assert len(rgb) == len(nir) == len(metas(raiz)) == 4        # 2 mangos x 2 caras
    ts_rgb = [PATRON_PIPELINE.fullmatch(p.name).group(3) for p in rgb]
    ts_nir = [PATRON_PIPELINE.fullmatch(p.name).group(3) for p in nir]
    assert ts_rgb == ts_nir and len(set(ts_rgb)) == 4
    assert all(PATRON_PIPELINE.fullmatch(p.name).group(1) == "mango" for p in rgb + nir)
    assert [p.stem.split("_")[-1] for p in metas(raiz)] == ts_rgb
    j = e.Jornada.cargar(raiz)
    assert j.pendientes_sanas() == []
    assert ctx.hw.luces.encendidas == set()


def test_sanas_el_orden_de_luces_es_blanca_rgb_apaga_nir_captura_apaga(tmp_path):
    ctx, _, _ = jornada_lista(tmp_path)
    ctx.hw.luces.historial.clear()
    flujo.cmd_sanas(ctx, args())
    por_cara = ctx.hw.luces.historial[:4]
    assert por_cara == [("on", "rgb"), ("off", "rgb"), ("on", "nir"), ("off", "nir")]


def test_sanas_metadatos_siguen_el_esquema_3_0(tmp_path):
    ctx, _, _ = jornada_lista(tmp_path)
    flujo.cmd_sanas(ctx, args())
    raiz = ctx.base / "J01"
    meta = leer_meta(metas(raiz)[0])
    assert meta["schema"] == "3.0" and meta["fase"] == "sano" and meta["ronda_h"] is None
    assert meta["impacto"] is None
    assert (meta["mango_id"], meta["cara"]) == (1, 1)
    assert meta["jornada"] == {"id": "J01", "lote": "lote-A", "temperatura_c": 24.5}
    assert meta["rgb"]["controles"]["gain"] == 0 and meta["rgb"]["config"]["controles"]["gain"] == 0
    assert meta["rgb"]["controles"]["exposure_time_absolute"] == 104
    assert meta["rgb"]["controles"]["focus_absolute"] == 300         # se registra aunque el foco sea manual
    assert meta["nir"]["rpicam_metadata"]["ExposureTime"] == 6000
    assert meta["nir"]["config"]["shutter_us"] == 6000 and meta["nir"]["config"]["gain"] == 1.0
    assert meta["calidad"]["rgb"]["estado"] == "ok" and meta["calidad"]["nir"]["estado"] == "ok"
    assert meta["salud"]["throttled"]["raw"] == "0x0"
    import hashlib
    rgb = raiz / "news" / "rgb" / f"mango_rgb_{meta['timestamp']}.jpg"
    assert meta["archivos"]["rgb"]["sha256"] == hashlib.sha256(rgb.read_bytes()).hexdigest()
    assert meta["archivos"]["nir"]["bytes"] > 0 and "raw" not in meta["archivos"]
    assert meta["subida"] == {"estado": "pendiente"}
    assert "total" in meta["tiempos_s"]
    assert "http" not in json.dumps(meta)


def test_sanas_con_raw_activado_guarda_el_dng_en_nir_raw(tmp_path):
    ctx, _, _ = jornada_lista(tmp_path)
    ctx.hw.nir.cfg["raw"] = True
    flujo.cmd_sanas(ctx, args())
    raiz = ctx.base / "J01"
    assert len(list((raiz / "news" / "nir_raw").glob("*.dng"))) == 4
    assert not list((raiz / "news" / "nir").glob("*.dng"))
    assert "raw" in leer_meta(metas(raiz)[0])["archivos"]


def test_sanas_no_deja_temporales_en_news(tmp_path):
    ctx, _, _ = jornada_lista(tmp_path)
    flujo.cmd_sanas(ctx, args())
    archivos = [p.name for p in (ctx.base / "J01" / "news").rglob("*") if p.is_file()]
    assert not [n for n in archivos if n.endswith(".tmp") or ".rpicam" in n]


def test_sanas_muestra_los_mensajes_del_operador_y_el_resultado_de_qc(tmp_path):
    ctx, _, salida = jornada_lista(tmp_path)
    flujo.cmd_sanas(ctx, args())
    texto = "\n".join(salida)
    assert "Mango 1 · cara 1 → coloca y presiona Enter" in texto
    assert "borde" in texto and "OK" in texto


def test_sanas_reanuda_donde_se_quedo(tmp_path):
    ctx, _, _ = jornada_lista(tmp_path)
    ctx1, _, _ = contexto(tmp_path, entradas=["", "", "q"], auto=False)
    flujo.cmd_sanas(ctx1, args())          # mango 1 completo y sale al llegar al mango 2
    j = e.Jornada.cargar(ctx.base / "J01")
    assert j.pendientes_sanas() == [(2, 1), (2, 2)]
    ctx2, _, salida = contexto(tmp_path, entradas=["", ""], auto=False)
    flujo.cmd_sanas(ctx2, args())
    assert e.Jornada.cargar(ctx.base / "J01").pendientes_sanas() == []
    assert len(metas(ctx.base / "J01")) == 4
    assert "Mango 1 · cara 1" not in "\n".join(salida)


def test_sanas_luz_apagada_ofrece_reintentar_y_descarta_la_toma_mala(tmp_path):
    ctx0, _, _ = jornada_lista(tmp_path)

    def entrada(texto):
        if "reintentar" in texto.lower():
            ctx.hw.luces.banco_dormido = False   # el operador desperto la bateria
        return ""

    ctx, _, salida = contexto(tmp_path, auto=False, entrada=entrada)
    ctx.hw.luces.banco_dormido = True
    flujo.cmd_sanas(ctx, args())
    texto = "\n".join(salida)
    assert "luz apagada?" in texto
    raiz = ctx0.base / "J01"
    assert len(list((raiz / "descartes").glob("*"))) >= 2        # rgb y nir de la toma mala
    j = e.Jornada.cargar(raiz)
    assert j.sana_hecha(1, 1)
    meta = leer_meta(metas(raiz)[0])
    assert meta["calidad"]["rgb"]["estado"] == "ok"               # el archivo bueno es el reintento


def test_sanas_en_auto_aborta_si_la_luz_esta_apagada(tmp_path):
    ctx, _, _ = jornada_lista(tmp_path)
    ctx.hw.luces.banco_dormido = True
    with pytest.raises(flujo.ErrorQC) as exc:
        flujo.cmd_sanas(ctx, args())
    assert "luz apagada?" in str(exc.value)
    assert e.Jornada.cargar(ctx.base / "J01").pendientes_sanas()[0] == (1, 1)
    assert not metas(ctx.base / "J01")
    assert ctx.hw.luces.encendidas == set()


def test_sanas_alerta_si_cambia_la_iluminacion_pero_guarda_la_foto(tmp_path):
    ctx, _, salida = jornada_lista(tmp_path)
    original = ctx.hw.rgb.capturar

    def mas_brillante(salida_, extra=None):
        info = original(salida_)
        from PIL import Image
        import numpy as np
        a = np.asarray(Image.open(salida_)).astype(np.float32) * 1.4
        Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(salida_, "JPEG", quality=90)
        return info

    ctx.hw.rgb.capturar = mas_brillante
    flujo.cmd_sanas(ctx, args())
    meta = leer_meta(metas(ctx.base / "J01")[0])
    assert any("la iluminación o la cámara cambió" in a for a in meta["calidad"]["alertas"])
    assert "ALERTA" in "\n".join(salida)


def test_sin_camara_nir_las_sanas_abortan(tmp_path):
    ctx, _, _ = jornada_lista(tmp_path)
    ctx.hw.nir.presente_simulada = False
    with pytest.raises(flujo.ErrorPreparacion):
        flujo.cmd_sanas(ctx, args())


def test_sin_jornada_activa_el_mensaje_es_claro(tmp_path):
    ctx, _, _ = contexto(tmp_path)
    with pytest.raises(flujo.ErrorPreparacion) as exc:
        flujo.cmd_sanas(ctx, args())
    assert "jornada" in str(exc.value).lower()


# --- impactos -----------------------------------------------------------------------------------------

def test_impactos_registra_la_hora_exacta_de_cada_cara_y_muestra_las_zonas(tmp_path):
    ctx, reloj, salida = jornada_lista(tmp_path)
    reloj.paso = timedelta(seconds=20)
    flujo.cmd_impactos(ctx, args())
    j = e.Jornada.cargar(ctx.base / "J01")
    assert j.pendientes_impactos() == []
    horas = [j.hora_impacto(m, c) for m in j.mangos for c in j.caras]
    assert horas == sorted(horas) and len(set(horas)) == len(horas)
    texto = "\n".join(salida)
    for m in j.mangos:
        for c in j.caras:
            assert f"Cara {c} · zona {j.zona(m, c)} → presiona Enter AL SOLTAR la bola" in texto
    assert j.datos["impactos"]["1"]["1"]["zona"] == j.zona(1, 1)


def test_impactos_muestra_el_ritmo_objetivo_y_avisa_si_el_operador_se_atrasa(tmp_path):
    ctx0, _, _ = jornada_lista(tmp_path, slot_min=2.5, tolerancia_min=10)
    reloj = ctx0.reloj
    llamadas = {"n": 0}

    def entrada(texto):
        llamadas["n"] += 1
        if llamadas["n"] == 3:      # tercera pulsacion = primera cara del mango 2
            reloj.avanzar(minutes=8)
        return ""

    ctx, _, salida = contexto(tmp_path, auto=False, entrada=entrada, reloj=reloj)
    flujo.cmd_impactos(ctx, args())
    texto = "\n".join(salida)
    assert "ritmo" in texto.lower() and "2.5 min" in texto
    assert any(linea.startswith("⚠") and "atrasado" in linea for linea in salida)


def test_impactos_reanuda_con_las_caras_que_faltan(tmp_path):
    ctx0, _, _ = jornada_lista(tmp_path)
    ctx1, _, _ = contexto(tmp_path, entradas=["", "q"], auto=False, reloj=ctx0.reloj)
    flujo.cmd_impactos(ctx1, args())
    j = e.Jornada.cargar(ctx0.base / "J01")
    assert j.pendientes_impactos() == [(1, 2), (2, 1), (2, 2)]
    ctx2, _, _ = contexto(tmp_path, reloj=ctx0.reloj)
    flujo.cmd_impactos(ctx2, args())
    assert e.Jornada.cargar(ctx0.base / "J01").pendientes_impactos() == []


# --- impactos con la ronda +0 h (las fotos justo despues del golpe) -------------------------------------------------

def jornada_cero(tmp_path, **kw):
    """Jornada calibrada con las rondas +0 h y +1 h, con las sanas ya tomadas."""
    datos = dict(intervalos="0,1")
    datos.update(kw)
    ctx, reloj, salida = jornada_lista(tmp_path, **datos)
    flujo.cmd_sanas(ctx, args())
    return ctx, reloj, salida


def post_impacto(raiz, ronda_h):
    return [m for m in map(leer_meta, metas(raiz)) if m["fase"] == "post_impacto" and m["ronda_h"] == ronda_h]


def acciones(salida):
    """Los avisos al operador de `impactos` (soltar la bola / colocar el mango), en orden."""
    return [linea for linea in salida if "AL SOLTAR" in linea or "coloca y presiona Enter" in linea]


def test_jornada_acepta_el_intervalo_cero(tmp_path):
    ctx, _, salida = jornada_lista(tmp_path, intervalos="0,2")
    assert e.Jornada.cargar(ctx.base / "J01").intervalos_h == [0, 2]
    assert any("intervalos [0, 2] h" in linea for linea in salida)


def test_impactos_con_ronda_cero_fotografia_cada_mango_justo_despues_de_sus_impactos(tmp_path, monkeypatch):
    ctx, reloj, _ = jornada_cero(tmp_path)
    eventos = []
    marcar_impacto, marcar_ronda = e.Jornada.marcar_impacto, e.Jornada.marcar_ronda

    def impacto(self, mango, cara, hora_iso):
        eventos.append(("impacto", mango, cara))
        return marcar_impacto(self, mango, cara, hora_iso)

    def ronda(self, mango, ronda_h, cara, *a, **k):
        eventos.append(("foto", mango, ronda_h, cara))
        return marcar_ronda(self, mango, ronda_h, cara, *a, **k)

    monkeypatch.setattr(e.Jornada, "marcar_impacto", impacto)
    monkeypatch.setattr(e.Jornada, "marcar_ronda", ronda)
    reloj.paso = timedelta(seconds=2)
    flujo.cmd_impactos(ctx, args())
    assert eventos == [("impacto", 1, 1), ("impacto", 1, 2), ("foto", 1, 0, 1), ("foto", 1, 0, 2),
                       ("impacto", 2, 1), ("impacto", 2, 2), ("foto", 2, 0, 1), ("foto", 2, 0, 2)]


def test_impactos_con_ronda_cero_guarda_las_fotos_del_cero_en_el_dataset_y_marca_la_ronda(tmp_path):
    ctx, reloj, salida = jornada_cero(tmp_path)
    reloj.paso = timedelta(seconds=2)
    flujo.cmd_impactos(ctx, args())
    raiz = ctx.base / "J01"
    j = e.Jornada.cargar(raiz)
    assert j.pendientes_impactos() == []
    assert [j.estado_ronda(m, 0) for m in j.mangos] == ["completa", "completa"]
    assert (1, 0) in j.hechos() and (2, 0) in j.hechos() and (1, 1) not in j.hechos()
    ceros = post_impacto(raiz, 0)
    assert sorted((m["mango_id"], m["cara"]) for m in ceros) == [(1, 1), (1, 2), (2, 1), (2, 2)]
    assert len(metas(raiz)) == 4 + 4                                       # 4 sanas + 4 fotos +0 h
    for m in ceros:
        imp = m["impacto"]
        assert imp["en_tolerancia"] is True and 0 <= imp["horas_desde_impacto"] < 0.5 / 60
        assert imp["zona"] == j.zona(m["mango_id"], m["cara"])
        assert imp["hora_iso"] == j.datos["impactos"][str(m["mango_id"])][str(m["cara"])]["hora_iso"]
        assert m["calidad"]["rgb"]["estado"] == "ok" and m["calidad"]["nir"]["estado"] == "ok"
    texto = "\n".join(salida)
    for mango in (1, 2):
        for cara in (1, 2):
            assert f"Mango {mango} · +0 h · cara {cara} → coloca y presiona Enter" in texto
    assert ctx.hw.luces.encendidas == set()


def test_impactos_con_ronda_cero_mantiene_los_mismos_avisos_de_impacto_y_el_ritmo(tmp_path):
    ctx, _, salida = jornada_cero(tmp_path)
    flujo.cmd_impactos(ctx, args())
    j = e.Jornada.cargar(ctx.base / "J01")
    texto = "\n".join(salida)
    for m in j.mangos:
        assert f"Mango {m} ({m} de 2) · zonas sorteadas → " in texto
        for c in j.caras:
            assert f"Cara {c} · zona {j.zona(m, c)} → presiona Enter AL SOLTAR la bola" in texto
    assert "Ritmo:" in texto and "un mango cada 0.5 min" in texto


def test_impactos_con_ronda_cero_toma_los_pares_con_luz_blanca_y_luego_nir(tmp_path):
    ctx, _, _ = jornada_cero(tmp_path)
    ctx.hw.luces.historial.clear()
    flujo.cmd_impactos(ctx, args())
    assert ctx.hw.luces.historial == [("on", "rgb"), ("off", "rgb"), ("on", "nir"), ("off", "nir")] * 4


def test_impactos_con_ronda_cero_reanuda_en_las_fotos_a_medias_de_un_mango(tmp_path):
    ctx0, reloj, _ = jornada_cero(tmp_path)
    ctx1, _, salida1 = contexto(tmp_path, entradas=["", "", "", "q"], auto=False, reloj=reloj)
    flujo.cmd_impactos(ctx1, args())                   # impactos del mango 1, foto de la cara 1 y sale en la cara 2
    raiz = ctx0.base / "J01"
    j = e.Jornada.cargar(raiz)
    assert j.pendientes_impactos() == [(2, 1), (2, 2)]
    assert j.caras_pendientes_ronda(1, 0) == [2] and j.n_caras_hechas_ronda(1, 0) == 1
    assert flujo.MENSAJE_SALIR in salida1 and ctx1.hw.luces.encendidas == set()
    ctx2, _, salida2 = contexto(tmp_path, reloj=reloj)
    flujo.cmd_impactos(ctx2, args())
    assert [a.split(" → ")[0] for a in acciones(salida2)] == [
        "Mango 1 · +0 h · cara 2",                      # primero lo que quedó a medias, sin repetir la cara 1
        f"Cara 1 · zona {j.zona(2, 1)}", f"Cara 2 · zona {j.zona(2, 2)}",
        "Mango 2 · +0 h · cara 1", "Mango 2 · +0 h · cara 2"]
    assert sorted((m["mango_id"], m["cara"]) for m in post_impacto(raiz, 0)) == [(1, 1), (1, 2), (2, 1), (2, 2)]
    assert e.Jornada.cargar(raiz).pendientes_impactos() == []


def test_impactos_con_ronda_cero_reanuda_en_los_impactos_a_medias_y_fotografia_al_completarlos(tmp_path):
    ctx0, reloj, _ = jornada_cero(tmp_path)
    ctx1, _, _ = contexto(tmp_path, entradas=["", "q"], auto=False, reloj=reloj)
    flujo.cmd_impactos(ctx1, args())                   # solo la cara 1 del mango 1
    raiz = ctx0.base / "J01"
    assert e.Jornada.cargar(raiz).pendientes_impactos() == [(1, 2), (2, 1), (2, 2)]
    assert post_impacto(raiz, 0) == []                 # no se fotografía hasta que las 4 caras del mango están golpeadas
    ctx2, _, salida2 = contexto(tmp_path, reloj=reloj)
    flujo.cmd_impactos(ctx2, args())
    j = e.Jornada.cargar(raiz)
    assert [a.split(" → ")[0].split(" · zona")[0] for a in acciones(salida2)] == [
        "Cara 2", "Mango 1 · +0 h · cara 1", "Mango 1 · +0 h · cara 2",
        "Cara 1", "Cara 2", "Mango 2 · +0 h · cara 1", "Mango 2 · +0 h · cara 2"]
    assert j.hechos() >= {(1, 0), (2, 0)}


def test_impactos_con_ronda_cero_al_reanudar_con_la_ventana_vencida_ofrece_capturar_o_omitir(tmp_path):
    ctx0, reloj, _ = jornada_cero(tmp_path)
    ctx1, _, _ = contexto(tmp_path, entradas=["", "", "q"], auto=False, reloj=reloj)
    flujo.cmd_impactos(ctx1, args())                   # impactos del mango 1 y sale ante la primera foto
    raiz = ctx0.base / "J01"
    assert e.Jornada.cargar(raiz).n_caras_hechas_ronda(1, 0) == 0
    reloj.avanzar(minutes=30)                          # la ventana +0 h (±0,5 min) del mango 1 ya pasó
    ctx2, _, salida2 = contexto(tmp_path, entradas=["s", "", "", "", ""], auto=False, reloj=reloj)
    flujo.cmd_impactos(ctx2, args())
    j = e.Jornada.cargar(raiz)
    assert "Enter = capturar igual · s = omitir esta ronda del mango · q = salir" in "\n".join(ctx2.prompts)
    assert any(linea.startswith("⚠") and "Mango 1 ronda +0 h" in linea and "FUERA de tolerancia" in linea
               for linea in salida2)
    assert j.estado_ronda(1, 0) == "omitida" and j.n_caras_hechas_ronda(1, 0) == 0
    assert j.estado_ronda(2, 0) == "completa"                # el mango 2 sí se fotografía, dentro de su ventana
    assert j.pendientes_impactos() == [] and (1, 0) in j.hechos()
    assert [m["mango_id"] for m in post_impacto(raiz, 0)] == [2, 2]


def test_impactos_con_ronda_cero_al_reanudar_tarde_puede_capturar_igual_y_queda_fuera_de_tolerancia(tmp_path):
    ctx0, reloj, _ = jornada_cero(tmp_path)
    ctx1, _, _ = contexto(tmp_path, entradas=["", "", "q"], auto=False, reloj=reloj)
    flujo.cmd_impactos(ctx1, args())
    reloj.avanzar(minutes=30)
    ctx2, _, _ = contexto(tmp_path, auto=False, entrada=lambda texto: "", reloj=reloj)
    flujo.cmd_impactos(ctx2, args())
    raiz = ctx0.base / "J01"
    j = e.Jornada.cargar(raiz)
    assert j.estado_ronda(1, 0) == "fuera_de_tolerancia" and j.ronda_completa(1, 0)
    mango1 = [m for m in post_impacto(raiz, 0) if m["mango_id"] == 1]
    assert len(mango1) == 2 and all(m["impacto"]["en_tolerancia"] is False for m in mango1)
    assert any("fuera de tolerancia" in a for a in mango1[0]["calidad"]["alertas"])
    assert j.estado_ronda(2, 0) == "completa"


def test_impactos_con_ronda_cero_q_ante_la_ventana_vencida_sale_sin_tocar_nada(tmp_path):
    ctx0, reloj, _ = jornada_cero(tmp_path)
    ctx1, _, _ = contexto(tmp_path, entradas=["", "", "q"], auto=False, reloj=reloj)
    flujo.cmd_impactos(ctx1, args())
    reloj.avanzar(minutes=30)
    ctx2, _, salida2 = contexto(tmp_path, entradas=["q"], auto=False, reloj=reloj)
    flujo.cmd_impactos(ctx2, args())
    j = e.Jornada.cargar(ctx0.base / "J01")
    assert flujo.MENSAJE_SALIR in salida2 and j.estado_ronda(1, 0) == "pendiente"
    assert j.pendientes_impactos() == [(2, 1), (2, 2)] and ctx2.hw.luces.encendidas == set()


def test_impactos_con_ronda_cero_omitir_una_cara_con_error_la_deja_pendiente_para_reanudar(tmp_path):
    ctx0, reloj, _ = jornada_cero(tmp_path)

    def entrada(texto):
        if "reintentar" in texto.lower():
            ctx.hw.luces.banco_dormido = False         # el operador omite la cara y despierta la batería
            return "s"
        return ""

    ctx, _, salida = contexto(tmp_path, auto=False, entrada=entrada, reloj=reloj)
    ctx.hw.luces.banco_dormido = True
    flujo.cmd_impactos(ctx, args())
    raiz = ctx0.base / "J01"
    j = e.Jornada.cargar(raiz)
    assert "luz apagada?" in "\n".join(salida)
    assert j.caras_pendientes_ronda(1, 0) == [1] and j.pendientes_impactos() == []
    assert j.estado_ronda(2, 0) == "completa" and (1, 0) not in j.hechos()
    ctx2, _, salida2 = contexto(tmp_path, reloj=reloj)
    flujo.cmd_impactos(ctx2, args())                   # la cara omitida se ofrece de nuevo
    assert [a.split(" → ")[0] for a in acciones(salida2)] == ["Mango 1 · +0 h · cara 1"]
    assert e.Jornada.cargar(raiz).ronda_completa(1, 0)


def test_impactos_con_ronda_cero_termina_diciendo_el_siguiente_paso_y_cuando_vence_la_primera_ronda(tmp_path):
    ctx, _, salida = jornada_cero(tmp_path, intervalos="0,2", unidad_min=60, slot_min=3, tolerancia_min=10)
    flujo.cmd_impactos(ctx, args())
    j = e.Jornada.cargar(ctx.base / "J01")
    primera = next(item for item in j.calendario() if item.ronda_h == 2)
    assert primera.vence.strftime("%H:%M") == "10:30"
    ultima = [linea for linea in salida if "Siguiente paso: captura.py ronda" in linea]
    assert len(ultima) == 1
    assert "+0 h" in ultima[0] and "Mango 1 · ronda +2 h · vence 10:30" in ultima[0]
    assert salida[-1] == ultima[0]


def test_impactos_con_ronda_cero_ya_completo_no_captura_de_nuevo_y_dice_el_siguiente_paso(tmp_path):
    ctx, _, salida = jornada_cero(tmp_path)
    flujo.cmd_impactos(ctx, args())
    raiz = ctx.base / "J01"
    antes, tomas = len(metas(raiz)), ctx.hw.nir.tomas
    del salida[:]
    flujo.cmd_impactos(ctx, args())
    assert len(metas(raiz)) == antes and ctx.hw.nir.tomas == tomas
    texto = "\n".join(salida)
    assert "Ya están" in texto and "Siguiente paso: captura.py ronda" in texto and "ronda +1 h" in texto


def test_impactos_con_solo_la_ronda_cero_remite_a_subir(tmp_path):
    ctx, _, salida = jornada_cero(tmp_path, intervalos="0")
    flujo.cmd_impactos(ctx, args())
    assert salida[-1].endswith("Siguiente paso: captura.py subir")
    assert e.Jornada.cargar(ctx.base / "J01").siguiente_pendiente() is None


def test_impactos_con_ronda_cero_se_niega_si_la_autoverificacion_fallo(tmp_path):
    ctx, _, _ = contexto(tmp_path)
    ctx.hw.rgb.ignora_exposicion = True
    with pytest.raises(flujo.ErrorPreparacion):
        flujo.cmd_jornada(ctx, args_jornada(intervalos="0,1"))
    with pytest.raises(flujo.ErrorPreparacion) as exc:
        flujo.cmd_impactos(ctx, args())
    assert "autoverificación" in str(exc.value) and "jornada" in str(exc.value)
    assert len(e.Jornada.cargar(ctx.base / "J01").pendientes_impactos()) == 4          # no registró nada


def test_impactos_con_ronda_cero_aborta_sin_camara_nir_antes_de_registrar_impactos(tmp_path):
    ctx, _, _ = jornada_cero(tmp_path)
    ctx.hw.nir.presente_simulada = False
    with pytest.raises(flujo.ErrorPreparacion) as exc:
        flujo.cmd_impactos(ctx, args())
    assert "NIR" in str(exc.value) and "cable" in str(exc.value)
    assert len(e.Jornada.cargar(ctx.base / "J01").pendientes_impactos()) == 4


def test_impactos_con_ronda_cero_avisa_si_otro_proceso_usa_las_camaras(tmp_path):
    ctx0, reloj, _ = jornada_cero(tmp_path)
    ctx, _, salida = contexto(tmp_path, reloj=reloj)
    ctx.hw.salud.ocupantes_simulados = [{"pid": 9, "nombre": "pipewire"}]
    flujo.cmd_impactos(ctx, args())
    assert any("pipewire" in linea and linea.startswith("⚠") for linea in salida)


def test_impactos_con_ronda_cero_conserva_lo_hecho_y_apaga_las_luces_si_la_noir_se_traba(tmp_path):
    ctx, _, _ = jornada_cero(tmp_path)
    ctx.hw.nir.trabada_simulada = True
    with pytest.raises(hw.NIRTrabada):
        flujo.cmd_impactos(ctx, args())
    j = e.Jornada.cargar(ctx.base / "J01")
    assert j.caras_impactadas(1) == [1, 2] and j.caras_pendientes_ronda(1, 0) == [1, 2]
    assert j.pendientes_impactos() == [(2, 1), (2, 2)]
    assert ctx.hw.luces.encendidas == set()


def test_impactos_con_ronda_cero_sube_las_fotos_del_cero_y_cierra_la_cola(tmp_path):
    put = PutSimulado()
    ctx, _, _ = contexto(tmp_path, subir=True, put=put, par=subida.ParUrl(SECRETO))
    flujo.cmd_jornada(ctx, args_jornada(no_subir=False, intervalos="0,1"))
    flujo.cmd_sanas(ctx, args())
    antes = len(put.llamadas)
    flujo.cmd_impactos(ctx, args())
    nuevos = [u[len(SECRETO):] for u, _ in put.llamadas[antes:]]
    assert [len([d for d in nuevos if d.startswith(f"news/{carpeta}/")]) for carpeta in ("rgb", "nir", "meta")] == [4, 4, 4]
    assert subida.ColaSubida(ctx.base / "J01" / "pendientes.jsonl").pendientes() == []


def test_impactos_sin_ronda_cero_no_usa_camaras_ni_pide_la_verificacion(tmp_path):
    ctx, _, salida = jornada_lista(tmp_path)                   # intervalos 1,2: el protocolo anterior
    j = e.Jornada.cargar(ctx.base / "J01")
    j.datos.pop("autoverificacion_exposicion")                 # una jornada vieja puede no tenerla
    j.guardar()
    ctx.hw.nir.presente_simulada = False
    ctx.hw.salud.ocupantes_simulados = [{"pid": 9, "nombre": "pipewire"}]
    antes = (len(ctx.hw.rgb.historial), ctx.hw.nir.tomas, list(ctx.hw.luces.historial))
    del salida[:]
    flujo.cmd_impactos(ctx, args())
    assert (len(ctx.hw.rgb.historial), ctx.hw.nir.tomas, ctx.hw.luces.historial) == antes
    raiz = ctx.base / "J01"
    assert e.Jornada.cargar(raiz).pendientes_impactos() == [] and metas(raiz) == []
    texto = "\n".join(salida)
    assert "Otro proceso" not in texto and "Impactos completos. Siguiente paso: captura.py ronda" in texto
    assert all("+0 h" not in linea for linea in salida)


def test_impactos_sin_ronda_cero_q_sigue_saliendo_con_el_mismo_mensaje(tmp_path):
    ctx0, reloj, _ = jornada_lista(tmp_path)
    ctx, _, salida = contexto(tmp_path, entradas=["", "q"], auto=False, reloj=reloj)
    flujo.cmd_impactos(ctx, args())
    assert flujo.MENSAJE_SALIR in salida
    assert e.Jornada.cargar(ctx0.base / "J01").pendientes_impactos() == [(1, 2), (2, 1), (2, 2)]


def test_ronda_no_repite_las_caras_del_cero_que_ya_tomo_impactos(tmp_path):
    ctx, reloj, salida = jornada_cero(tmp_path)
    reloj.paso = timedelta(seconds=2)
    flujo.cmd_impactos(ctx, args())
    reloj.paso = timedelta(0)
    del salida[:]
    flujo.cmd_ronda(ctx, args())
    raiz = ctx.base / "J01"
    posts = [(m["mango_id"], m["cara"], m["ronda_h"]) for m in map(leer_meta, metas(raiz))
             if m["fase"] == "post_impacto"]
    assert len(posts) == len(set(posts)) == 8
    assert sorted(r for _, _, r in posts) == [0, 0, 0, 0, 1, 1, 1, 1]
    assert e.Jornada.cargar(raiz).hechos() == {(1, 0), (2, 0), (1, 1), (2, 1)}
    texto = "\n".join(salida)
    assert "ronda +0 h" not in texto and "+0 h · cara" not in texto and "ronda +1 h" in texto


def test_ronda_ofrece_el_cero_pendiente_si_impactos_se_interrumpio_y_solo_las_caras_que_faltan(tmp_path):
    ctx0, reloj, _ = jornada_cero(tmp_path)
    ctx1, _, _ = contexto(tmp_path, entradas=["", "", "", "q"], auto=False, reloj=reloj)
    flujo.cmd_impactos(ctx1, args())                   # el mango 1 queda con la cara 1 del +0 h hecha
    reloj.avanzar(minutes=30)
    ctx2, _, salida2 = contexto(tmp_path, entradas=["", "", "q"], auto=False, reloj=reloj)
    flujo.cmd_ronda(ctx2, args())                      # +0 h del mango 1 atrasado: captura igual; luego sale en +1 h
    raiz = ctx0.base / "J01"
    j = e.Jornada.cargar(raiz)
    assert "ronda +0 h" in "\n".join(salida2) and "FUERA de tolerancia" in "\n".join(salida2)
    assert j.ronda_completa(1, 0)
    assert sorted(m["cara"] for m in post_impacto(raiz, 0) if m["mango_id"] == 1) == [1, 2]     # la cara 1 no se repitió
    assert j.estado_ronda(1, 0) == "fuera_de_tolerancia"


def test_impactos_con_ronda_cero_tras_un_cero_parcial_de_ronda_solo_fotografia_las_caras_nuevas(tmp_path):
    ctx0, reloj, _ = jornada_cero(tmp_path)
    ctx1, _, _ = contexto(tmp_path, entradas=["", "q"], auto=False, reloj=reloj)
    flujo.cmd_impactos(ctx1, args())                   # solo la cara 1 del mango 1 golpeada
    ctx2, _, _ = contexto(tmp_path, entradas=["", "", "q"], auto=False, reloj=reloj)
    flujo.cmd_ronda(ctx2, args())                      # ronda toma el +0 h de esa cara y sale ante el +1 h
    raiz = ctx0.base / "J01"
    assert e.Jornada.cargar(raiz).ronda_completa(1, 0)             # "completa" solo respecto de las caras golpeadas
    ctx3, _, salida3 = contexto(tmp_path, reloj=reloj)
    flujo.cmd_impactos(ctx3, args())
    assert [a.split(" → ")[0].split(" · zona")[0] for a in acciones(salida3)] == [
        "Cara 2", "Mango 1 · +0 h · cara 2", "Cara 1", "Cara 2", "Mango 2 · +0 h · cara 1", "Mango 2 · +0 h · cara 2"]
    assert sorted((m["mango_id"], m["cara"]) for m in post_impacto(raiz, 0)) == [(1, 1), (1, 2), (2, 1), (2, 2)]


def test_estado_rotula_la_ronda_cero_y_la_marca_hecha_tras_los_impactos(tmp_path):
    ctx, reloj, salida = jornada_cero(tmp_path)
    flujo.cmd_impactos(ctx, args())
    del salida[:]
    flujo.cmd_estado(ctx, args())
    cabecera = next(linea for linea in salida if linea.startswith("Mango"))
    assert cabecera.split() == ["Mango", "sano", "imp", "+0", "h", "+1", "h"]
    fila = next(linea for linea in salida if linea.split()[:1] == ["1"])
    assert fila.split() == ["1", "✓", "✓", "✓", "⏳08:31"]
    assert "Siguiente: Mango 1 · ronda +1 h" in "\n".join(salida)


# --- rondas ---------------------------------------------------------------------------------------------

def jornada_con_impactos(tmp_path, **kw):
    ctx, reloj, salida = jornada_lista(tmp_path, **kw)
    flujo.cmd_sanas(ctx, args())
    reloj.paso = timedelta(seconds=2)
    flujo.cmd_impactos(ctx, args())
    reloj.paso = timedelta(0)
    return ctx, reloj, salida


def test_ronda_espera_con_cuenta_regresiva_y_captura_a_la_hora_exacta(tmp_path):
    ctx, reloj, salida = jornada_con_impactos(tmp_path)
    flujo.cmd_ronda(ctx, args())
    raiz = ctx.base / "J01"
    j = e.Jornada.cargar(raiz)
    assert j.hechos() == {(1, 1), (2, 1), (1, 2), (2, 2)}
    texto = "\n".join(salida)
    assert "ronda +1 h" in texto and "vence" in texto and "faltan" in texto
    assert "batería USB" in texto          # tras una espera larga la batería de las luces pudo dormirse
    posts = [leer_meta(p) for p in metas(raiz) if leer_meta(p)["fase"] == "post_impacto"]
    assert len(posts) == 8                                        # 2 mangos x 2 caras x 2 rondas
    for m in posts:
        imp = m["impacto"]
        assert imp["en_tolerancia"] is True
        assert imp["horas_desde_impacto"] == pytest.approx(m["ronda_h"] / 60, abs=0.5 / 60 + 1e-6)
        assert imp["zona"] == j.zona(m["mango_id"], m["cara"])
        assert imp["hora_iso"] == j.datos["impactos"][str(m["mango_id"])][str(m["cara"])]["hora_iso"]
        assert imp["masa_bola_g"] == 66.0 and imp["altura_cm"] == 50.0
    todas = [leer_meta(p)["timestamp"] for p in metas(raiz)]
    assert len(set(todas)) == len(todas) == 12
    assert "No queda nada pendiente" in texto


def test_ronda_pide_enter_por_cara_con_el_mensaje_del_protocolo(tmp_path):
    ctx, reloj, salida = jornada_con_impactos(tmp_path)
    reloj.avanzar(minutes=1)          # ya es hora de la ronda +1
    flujo.cmd_ronda(ctx, args())
    assert "Mango 1 · cara 2 → coloca y presiona Enter" in "\n".join(salida)


def test_ronda_con_forzar_captura_de_inmediato_y_marca_fuera_de_tolerancia(tmp_path):
    ctx, reloj, _ = jornada_con_impactos(tmp_path)
    flujo.cmd_ronda(ctx, args(forzar=True))
    raiz = ctx.base / "J01"
    posts = [leer_meta(p) for p in metas(raiz) if leer_meta(p)["fase"] == "post_impacto"]
    assert posts and all(m["impacto"]["en_tolerancia"] is False for m in posts)
    assert any("fuera de tolerancia" in a for a in posts[0]["calidad"]["alertas"])
    assert e.Jornada.cargar(raiz).estado_ronda(1, 1) == "fuera_de_tolerancia"


def test_ronda_interactiva_ofrece_esperar_o_salir_si_esta_temprano(tmp_path):
    ctx0, reloj, _ = jornada_con_impactos(tmp_path)
    ctx, _, salida = contexto(tmp_path, entradas=["q"], auto=False, reloj=reloj)
    flujo.cmd_ronda(ctx, args())
    texto = "\n".join(salida)
    assert "--forzar" in texto and "esperar" in texto.lower()
    j = e.Jornada.cargar(ctx0.base / "J01")
    assert j.n_caras_hechas_ronda(1, 1) == 0


def test_ronda_interactiva_puede_omitir_un_item_atrasado(tmp_path):
    ctx0, reloj, _ = jornada_con_impactos(tmp_path)
    reloj.avanzar(minutes=30)         # todo quedo atrasado
    ctx, _, salida = contexto(tmp_path, entradas=["s", "q"], auto=False, reloj=reloj)
    flujo.cmd_ronda(ctx, args())
    j = e.Jornada.cargar(ctx0.base / "J01")
    assert j.estado_ronda(1, 1) == "omitida"
    assert j.n_caras_hechas_ronda(2, 1) == 0
    assert "atrasado" in "\n".join(salida)


def test_ronda_atrasada_se_captura_y_queda_marcada_fuera_de_tolerancia(tmp_path):
    ctx0, reloj, _ = jornada_con_impactos(tmp_path)
    reloj.avanzar(minutes=30)
    ctx, _, _ = contexto(tmp_path, entradas=["", "", "", "q"], auto=False, reloj=reloj)
    flujo.cmd_ronda(ctx, args())
    j = e.Jornada.cargar(ctx0.base / "J01")
    assert j.ronda_completa(1, 1) and j.estado_ronda(1, 1) == "fuera_de_tolerancia"


def test_ronda_sin_impactos_explica_que_faltan(tmp_path):
    ctx, _, salida = jornada_lista(tmp_path)
    flujo.cmd_ronda(ctx, args())
    assert "impactos" in "\n".join(salida).lower()


def test_ronda_reanuda_sin_repetir_lo_hecho(tmp_path):
    ctx0, reloj, _ = jornada_con_impactos(tmp_path)
    reloj.avanzar(minutes=1)
    ctx1, _, _ = contexto(tmp_path, entradas=["", "", "q"], auto=False, reloj=reloj)
    flujo.cmd_ronda(ctx1, args())      # primera caras del mango 1 y sale
    j = e.Jornada.cargar(ctx0.base / "J01")
    assert j.n_caras_hechas_ronda(1, 1) == 1
    ctx2, _, salida = contexto(tmp_path, reloj=reloj)
    flujo.cmd_ronda(ctx2, args())
    assert e.Jornada.cargar(ctx0.base / "J01").hechos() == {(1, 1), (2, 1), (1, 2), (2, 2)}
    # la ronda +1 h del mango 1 tiene exactamente una foto por cara: la cara 1 no se repitio
    primeras = [leer_meta(p) for p in metas(ctx0.base / "J01")]
    ronda1 = sorted(m["cara"] for m in primeras
                    if m["fase"] == "post_impacto" and m["mango_id"] == 1 and m["ronda_h"] == 1)
    assert ronda1 == [1, 2]


# --- estado --------------------------------------------------------------------------------------------------

def test_estado_imprime_la_tabla_y_el_siguiente_vencimiento(tmp_path):
    ctx, reloj, salida = jornada_con_impactos(tmp_path)
    del salida[:]
    flujo.cmd_estado(ctx, args())
    texto = "\n".join(salida)
    assert "Jornada J01" in texto and "Siguiente: Mango 1 · ronda +1 h" in texto
    assert "✓" in texto and "⏳" in texto and "Descanso" in texto


# --- subida ----------------------------------------------------------------------------------------------------

def test_las_capturas_se_encolan_y_el_hilo_las_sube_sin_mostrar_la_url(tmp_path):
    put = PutSimulado()
    ctx, reloj, salida = contexto(tmp_path, subir=True, put=put, par=subida.ParUrl(SECRETO))
    flujo.cmd_jornada(ctx, args_jornada(no_subir=False))
    flujo.cmd_sanas(ctx, args())
    cola = subida.ColaSubida(ctx.base / "J01" / "pendientes.jsonl")
    assert cola.pendientes() == []
    destinos = [u[len(SECRETO):] for u, _ in put.llamadas]
    assert len([d for d in destinos if d.startswith("news/rgb/")]) == 4
    assert len([d for d in destinos if d.startswith("news/nir/")]) == 4
    assert len([d for d in destinos if d.startswith("news/meta/")]) == 4
    assert len([d for d in destinos if d.startswith("calib/")]) == 10     # 3 tipos x (rgb, nir, meta) + barrido de balance
    assert "SECRETO123" not in "\n".join(salida)


def test_con_no_subir_no_se_encola_nada(tmp_path):
    put = PutSimulado()
    ctx, _, _ = contexto(tmp_path, subir=True, put=put, par=subida.ParUrl(SECRETO))
    flujo.cmd_jornada(ctx, args_jornada(no_subir=True))
    flujo.cmd_sanas(ctx, args())
    assert put.llamadas == []
    assert not (ctx.base / "J01" / "pendientes.jsonl").exists() or \
        subida.ColaSubida(ctx.base / "J01" / "pendientes.jsonl").pendientes() == []


def test_sin_url_configurada_se_captura_igual_y_se_avisa(tmp_path):
    ctx, _, salida = contexto(tmp_path, subir=True, put=PutSimulado(), par=None)
    flujo.cmd_jornada(ctx, args_jornada(no_subir=False))
    flujo.cmd_sanas(ctx, args())
    assert len(metas(ctx.base / "J01")) == 4
    assert any("URL" in linea and "subir" in linea for linea in salida)
    cola = subida.ColaSubida(ctx.base / "J01" / "pendientes.jsonl")
    assert len(cola.pendientes()) == 22     # 10 de calibracion + 12 de las sanas, a la espera de `subir`


def test_subir_sube_lo_pendiente_y_lo_capturado_con_no_subir(tmp_path):
    ctx0, _, _ = jornada_lista(tmp_path)          # no_subir=True: nada encolado
    flujo.cmd_sanas(ctx0, args())
    put = PutSimulado()
    ctx, _, salida = contexto(tmp_path, subir=True, put=put, par=subida.ParUrl(SECRETO))
    flujo.cmd_subir(ctx, args())
    destinos = sorted(u[len(SECRETO):] for u, _ in put.llamadas)
    assert len(destinos) == 22                    # 12 de las sanas + 10 de calibracion
    assert any(d.startswith("calib/") for d in destinos) and any(d.startswith("news/meta/") for d in destinos)
    texto = "\n".join(salida)
    assert "22" in texto and "SECRETO123" not in texto
    ctx3, _, salida3 = contexto(tmp_path, subir=True, put=PutSimulado(), par=subida.ParUrl(SECRETO))
    flujo.cmd_subir(ctx3, args())                 # segunda vez: no hay nada que subir
    assert "Nada pendiente" in "\n".join(salida3)


def test_subir_sin_url_da_un_error_claro(tmp_path):
    ctx0, _, _ = jornada_lista(tmp_path)
    ctx, _, _ = contexto(tmp_path, subir=True, put=PutSimulado(), par=None)
    with pytest.raises(flujo.ErrorPreparacion) as exc:
        flujo.cmd_subir(ctx, args())
    assert "OCI_PAR_URL" in str(exc.value)


# --- enfoque (el foco es manual: lectura de nitidez en vivo) ----------------------------------------------------

def test_enfoque_rgb_en_vivo_usa_la_luz_blanca_y_tomas_rapidas(tmp_path):
    ctx, _, salida = contexto(tmp_path)
    flujo.cmd_enfoque(ctx, args(camara="rgb", max_lecturas=3))
    texto = "\n".join(salida)
    assert texto.count("nitidez") >= 3 and "maximo visto" in texto
    assert ctx.hw.rgb.ultima["resolucion"] == "1600x1200" and ctx.hw.rgb.ultima["descartar"] == 2
    assert ("on", "rgb") in ctx.hw.luces.historial and ("off", "rgb") in ctx.hw.luces.historial
    assert ("on", "nir") not in ctx.hw.luces.historial
    assert ctx.hw.luces.encendidas == set()
    assert not list(ctx.base.glob("tmp_enfoque*"))


def test_enfoque_nir_en_vivo_usa_la_luz_nir(tmp_path):
    ctx, _, salida = contexto(tmp_path)
    flujo.cmd_enfoque(ctx, args(camara="nir", max_lecturas=3))
    texto = "\n".join(salida)
    assert texto.count("nitidez") >= 3
    assert ("on", "nir") in ctx.hw.luces.historial and ("off", "nir") in ctx.hw.luces.historial
    assert ("on", "rgb") not in ctx.hw.luces.historial
    assert ctx.hw.luces.encendidas == set()


def test_enfoque_apaga_las_luces_aunque_se_interrumpa_con_ctrl_c(tmp_path):
    ctx, _, _ = contexto(tmp_path)

    def interrumpida(*a, **k):
        raise KeyboardInterrupt

    ctx.hw.rgb.capturar = interrumpida
    flujo.cmd_enfoque(ctx, args(camara="rgb"))
    assert ctx.hw.luces.encendidas == set() and ("off", "rgb") in ctx.hw.luces.historial


def test_enfoque_nir_sin_camara_aborta(tmp_path):
    ctx, _, _ = contexto(tmp_path)
    ctx.hw.nir.presente_simulada = False
    with pytest.raises(flujo.ErrorPreparacion):
        flujo.cmd_enfoque(ctx, args(camara="nir", max_lecturas=1))


# --- descarte de cuadros (cuantos cuadros hay que saltar para que los ajustes hagan efecto) -----------------------

def test_descarte_mide_el_tiempo_de_cada_valor_y_restaura_la_configuracion(tmp_path):
    ctx, _, salida = contexto(tmp_path)
    original = ctx.hw.rgb.cfg["cuadros_descartados"]
    flujo.cmd_descarte(ctx, args(valores="0,5,10"))
    texto = "\n".join(salida)
    for v in (0, 5, 10):
        assert f"descartar {v} cuadros" in texto
    assert "borde" in texto and "KB" in texto
    assert ctx.hw.rgb.cfg["cuadros_descartados"] == original
    assert ctx.hw.luces.encendidas == set()
    assert not list((ctx.base).glob("tmp_descarte*"))


def test_descarte_con_valores_invalidos(tmp_path):
    ctx, _, _ = contexto(tmp_path)
    with pytest.raises(flujo.ErrorPreparacion):
        flujo.cmd_descarte(ctx, args(valores="a,b"))


# --- controles de la camara RGB: diferencias y fallos -----------------------------------------------------------

def test_los_controles_distintos_de_lo_pedido_generan_una_alerta_en_la_foto(tmp_path):
    ctx, _, salida = jornada_lista(tmp_path)
    original = ctx.hw.rgb.capturar

    def con_diferencias(salida_, **kw):
        info = original(salida_, **kw)
        info["diferencias"] = {"gain": {"pedido": 0, "leido": 20}}
        return info

    ctx.hw.rgb.capturar = con_diferencias
    flujo.cmd_sanas(ctx, args())
    meta = leer_meta(metas(ctx.base / "J01")[0])
    assert any("distintos de lo pedido" in a and "gain" in a for a in meta["calidad"]["alertas"])
    assert any(linea.startswith("⚠") and "gain" in linea for linea in salida)


def test_un_control_que_no_se_pudo_fijar_queda_registrado_y_avisado_salvo_el_foco(tmp_path):
    ctx, _, salida = jornada_lista(tmp_path)
    original = ctx.hw.rgb.capturar

    def con_fallos(salida_, **kw):
        info = original(salida_, **kw)
        info["controles_fallidos"] = {"focus_absolute": "Invalid argument", "sharpness": "rc=255"}
        return info

    ctx.hw.rgb.capturar = con_fallos
    flujo.cmd_sanas(ctx, args())
    meta = leer_meta(metas(ctx.base / "J01")[0])
    assert set(meta["rgb"]["controles_fallidos"]) == {"focus_absolute", "sharpness"}      # todo queda registrado
    alertas = " ".join(meta["calidad"]["alertas"])
    assert "sharpness" in alertas and "focus_absolute" not in alertas                     # el foco no importa


# --- NoIR trabada: prueba rapida al empezar y abort sin reintentos --------------------------------------------

MENSAJE_NOIR_TRABADA = ("la NoIR quedó trabada: reiniciá la Pi (sudo reboot) y retomá con el mismo comando; "
                        "el estado se conserva")


def test_jornada_hace_una_toma_rapida_de_prueba_de_la_noir_antes_de_calibrar(tmp_path):
    ctx, _, _ = contexto(tmp_path)
    eventos = []
    original = ctx.hw.nir.probar
    ctx.hw.nir.probar = lambda carpeta: (eventos.append(len(ctx.hw.rgb.historial)), original(carpeta))[1]
    flujo.cmd_jornada(ctx, args_jornada())
    assert eventos == [0]            # una sola vez, antes de la primera toma RGB de la calibracion


def test_jornada_con_la_noir_trabada_aborta_antes_de_crear_nada(tmp_path):
    ctx, _, _ = contexto(tmp_path)
    ctx.hw.nir.trabada_simulada = True
    with pytest.raises(hw.NIRTrabada) as exc:
        flujo.cmd_jornada(ctx, args_jornada())
    assert MENSAJE_NOIR_TRABADA in str(exc.value)
    assert not (ctx.base / "J01").exists() and ctx.hw.rgb.historial == []
    assert ctx.hw.luces.encendidas == set()


def test_una_noir_trabada_a_mitad_de_las_sanas_aborta_conserva_el_estado_y_no_reintenta(tmp_path):
    ctx0, _, _ = jornada_lista(tmp_path)
    ctx, _, salida = contexto(tmp_path, auto=False, entrada=lambda texto: "")
    nir = ctx.hw.nir
    original = nir.capturar
    llamadas = []

    def capturar_y_trabar(*a, **k):
        llamadas.append(1)
        if nir.tomas >= 2:                 # tras dos caras buenas, la camara queda trabada
            nir.trabada_simulada = True
        return original(*a, **k)

    nir.capturar = capturar_y_trabar
    with pytest.raises(hw.NIRTrabada) as exc:
        flujo.cmd_sanas(ctx, args())
    assert MENSAJE_NOIR_TRABADA in str(exc.value)
    assert len(llamadas) == 3                                  # no hubo un cuarto intento
    assert all("reintentar" not in p.lower() for p in ctx.prompts)
    j = e.Jornada.cargar(ctx0.base / "J01")
    assert j.pendientes_sanas() == [(2, 1), (2, 2)]            # lo hecho se conserva
    assert ctx.hw.luces.encendidas == set()


# --- autoverificacion de la exposicion manual (trampa del firmware de la Arducam) --------------------------------

MENSAJE_EXPOSICION = ("la cámara RGB no está respetando la exposición manual: desconectá y reconectá la "
                      "alimentación o reiniciá la Pi")


def test_jornada_verifica_que_la_camara_respete_la_exposicion_manual_y_guarda_la_razon(tmp_path):
    ctx, _, salida = jornada_lista(tmp_path)
    ver = e.Jornada.cargar(ctx.base / "J01").autoverificacion_exposicion()
    assert ver["ok"] is True and ver["razon"] >= 1.5
    assert (ver["exposicion_alta"], ver["exposicion_baja"]) == (104, 34)         # E y E/3
    assert ver["media_alta"] > ver["media_baja"] > 0
    ultimas = ctx.hw.rgb.historial[-2:]
    assert [c["extra"]["exposure_time_absolute"] for c in ultimas] == [104, 34]
    assert all(c["resolucion"] == "1600x1200" for c in ultimas)                  # tomas rapidas de baja resolucion
    assert "Exposición manual verificada" in "\n".join(salida)


def test_la_autoverificacion_va_despues_de_la_calibracion_con_la_luz_blanca(tmp_path):
    ctx, _, _ = jornada_lista(tmp_path)
    j = e.Jornada.cargar(ctx.base / "J01")
    assert j.autoverificacion_exposicion()["timestamp"] > j.datos["calibracion"]["aruco"]["timestamp"]
    assert ctx.hw.luces.encendidas == set()


def test_la_jornada_aborta_si_la_camara_ignora_la_exposicion_manual(tmp_path):
    ctx, _, _ = contexto(tmp_path)
    ctx.hw.rgb.ignora_exposicion = True
    with pytest.raises(flujo.ErrorPreparacion) as exc:
        flujo.cmd_jornada(ctx, args_jornada())
    assert MENSAJE_EXPOSICION in str(exc.value)
    j = e.Jornada.cargar(ctx.base / "J01")
    ver = j.autoverificacion_exposicion()
    assert ver["ok"] is False and ver["razon"] == pytest.approx(1.0, abs=0.1)
    assert not j.calibracion_hecha("vacia") and j.balance_blancos() is None     # calibrada con la exposicion mala
    assert ctx.hw.luces.encendidas == set()


def test_sanas_y_rondas_se_niegan_si_la_autoverificacion_fallo(tmp_path):
    ctx, _, _ = contexto(tmp_path)
    ctx.hw.rgb.ignora_exposicion = True
    with pytest.raises(flujo.ErrorPreparacion):
        flujo.cmd_jornada(ctx, args_jornada())
    for comando in (flujo.cmd_sanas, flujo.cmd_ronda):
        with pytest.raises(flujo.ErrorPreparacion) as exc:
            comando(ctx, args())
        assert "autoverificación" in str(exc.value) and "jornada" in str(exc.value)


def test_la_autoverificacion_se_repite_al_reanudar_y_la_calibracion_se_rehace(tmp_path):
    ctx, _, _ = contexto(tmp_path)
    ctx.hw.rgb.ignora_exposicion = True
    with pytest.raises(flujo.ErrorPreparacion):
        flujo.cmd_jornada(ctx, args_jornada())
    ctx2, _, _ = contexto(tmp_path)                      # tras reiniciar la alimentacion
    flujo.cmd_jornada(ctx2, args_jornada())
    j = e.Jornada.cargar(ctx.base / "J01")
    assert j.autoverificacion_exposicion()["ok"] is True
    assert all(j.calibracion_hecha(t) for t in ("gris", "vacia", "aruco"))
    flujo.cmd_sanas(ctx2, args())                        # ya se puede capturar
    assert j.pendientes_sanas() != [] and e.Jornada.cargar(ctx.base / "J01").pendientes_sanas() == []


# --- procesos ajenos usando las camaras ------------------------------------------------------------------------------

def test_el_arranque_avisa_si_otro_proceso_tiene_abiertos_los_nodos_de_video_o_medios(tmp_path):
    ctx, _, salida = contexto(tmp_path)
    ctx.hw.salud.ocupantes_simulados = [{"pid": 7, "nombre": "wireplumber"}]
    flujo.cmd_jornada(ctx, args_jornada())                # avisa pero no aborta
    aviso = next(linea for linea in salida if "wireplumber" in linea)
    assert aviso.startswith("⚠") and "pid 7" in aviso and "README" in aviso
    assert (ctx.base / "J01" / "jornada.json").exists()


@pytest.mark.parametrize("comando,argumentos", [("cmd_sanas", {}), ("cmd_enfoque", {"camara": "rgb", "max_lecturas": 1}),
                                                 ("cmd_ronda", {})])
def test_el_aviso_de_procesos_ajenos_vale_en_todos_los_comandos_con_camara(tmp_path, comando, argumentos):
    ctx0, _, _ = jornada_lista(tmp_path)
    ctx, _, salida = contexto(tmp_path)
    ctx.hw.salud.ocupantes_simulados = [{"pid": 9, "nombre": "pipewire"}]
    getattr(flujo, comando)(ctx, args(**argumentos))
    assert any("pipewire" in linea and linea.startswith("⚠") for linea in salida)


def test_sin_ocupantes_o_sin_fuser_no_hay_aviso(tmp_path):
    ctx, _, salida = contexto(tmp_path)
    flujo.cmd_jornada(ctx, args_jornada())
    assert not any("Otro proceso" in linea for linea in salida)
    ctx2, _, salida2 = contexto(tmp_path, reloj=ctx.reloj)
    ctx2.hw.salud.ocupantes = lambda: None               # fuser no esta instalado: no se pudo verificar
    flujo.cmd_sanas(ctx2, args())
    assert not any("Otro proceso" in linea for linea in salida2)
