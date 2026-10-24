"""config_camaras.json: se crea con valores razonables y se completa si faltan claves."""
import json

import pytest

import camaras
import configuracion as cfg


def test_primera_ejecucion_crea_el_archivo_con_los_defectos(tmp_path):
    c = cfg.cargar_config(tmp_path)
    ruta = tmp_path / "config_camaras.json"
    assert ruta.exists()
    assert json.loads(ruta.read_text()) == c
    assert c["rgb"]["resolucion"] == "3264x2448"
    assert c["nir"]["width"] == 3280 and c["nir"]["height"] == 2464 and c["nir"]["mode"] == "3280:2464"
    assert c["nir"]["denoise"] == "cdn_off" and c["nir"]["awbgains"] == "1,1"
    assert c["nir"]["raw"] is False
    assert c["luces"]["rgb_gpio"] == 27 and c["luces"]["nir_gpio"] == 22


def test_los_controles_rgb_por_defecto_son_los_pedidos_por_el_protocolo(tmp_path):
    ctrls = cfg.cargar_config(tmp_path)["rgb"]["controles"]
    assert ctrls["auto_exposure"] == 1                      # manual
    assert ctrls["exposure_dynamic_framerate"] == 0
    assert ctrls["white_balance_automatic"] == 0
    assert ctrls["focus_automatic_continuous"] == 0
    assert ctrls["backlight_compensation"] == 0
    assert ctrls["power_line_frequency"] == 0
    for nombre in ("exposure_time_absolute", "gain", "white_balance_temperature",
                   "focus_absolute", "sharpness"):
        assert nombre in ctrls
    # mango a la altura nueva, sin reflector: 10,4 ms dejan 0,1 % del mango recortado (techo medido)
    assert ctrls["exposure_time_absolute"] == 104 and ctrls["gain"] == 0
    assert ctrls["white_balance_temperature"] == 6500      # 5000 daba un tinte cian fuerte
    assert ctrls["sharpness"] == 3                         # valor de fabrica: con 0 se veia blanda y granulosa
    # barrido NIR a la altura nueva: 20 ms x2 quemaba el 66 % del mango; 6 ms x1 deja 0,1 %
    assert cfg.DEFECTOS["nir"]["shutter_us"] == 6000 and cfg.DEFECTOS["nir"]["gain"] == 1.0
    assert 0 <= ctrls["gain"] <= 100
    assert 2800 <= ctrls["white_balance_temperature"] <= 6500
    assert 1 <= ctrls["focus_absolute"] <= 1023
    assert 0 <= ctrls["sharpness"] <= 6


def test_el_dispositivo_por_defecto_es_el_by_id(tmp_path):
    assert cfg.cargar_config(tmp_path)["rgb"]["device_by_id"] == camaras.DEVICE_RGB_DEFECTO
    assert "/dev/video" not in cfg.cargar_config(tmp_path)["rgb"]["device_by_id"]


def test_se_completan_las_claves_que_faltan_sin_pisar_las_del_usuario(tmp_path):
    (tmp_path / "config_camaras.json").write_text(
        json.dumps({"rgb": {"controles": {"gain": 35}}, "nir": {"shutter_us": 30000}}))
    c = cfg.cargar_config(tmp_path)
    assert c["rgb"]["controles"]["gain"] == 35
    assert c["rgb"]["controles"]["focus_absolute"] == cfg.DEFECTOS["rgb"]["controles"]["focus_absolute"]
    assert c["nir"]["shutter_us"] == 30000 and c["nir"]["gain"] == cfg.DEFECTOS["nir"]["gain"]
    assert "qc" in c


def test_json_corrupto_da_un_error_claro_y_no_se_sobrescribe(tmp_path):
    ruta = tmp_path / "config_camaras.json"
    ruta.write_text("{roto")
    with pytest.raises(cfg.ConfigInvalida) as exc:
        cfg.cargar_config(tmp_path)
    assert "config_camaras.json" in str(exc.value)
    assert ruta.read_text() == "{roto"


def test_los_defectos_no_se_comparten_entre_llamadas(tmp_path):
    a = cfg.cargar_config(tmp_path / "a")
    a["rgb"]["controles"]["gain"] = 99
    b = cfg.cargar_config(tmp_path / "b")
    assert b["rgb"]["controles"]["gain"] != 99


def test_guardar_un_control_rgb_calibrado(tmp_path):
    cfg.cargar_config(tmp_path)
    cfg.guardar_control_rgb(tmp_path, "white_balance_temperature", 3900)
    c = cfg.cargar_config(tmp_path)
    assert c["rgb"]["controles"]["white_balance_temperature"] == 3900
    assert c["rgb"]["controles"]["gain"] == 0          # lo demas no se toca


def test_el_foco_ya_no_se_calibra_por_barrido():
    """El foco de la RGB es manual (focus_absolute no hace nada): no hay barrido ni guardar_foco."""
    assert not hasattr(cfg, "guardar_foco")


def test_orden_de_aplicacion_de_controles_pone_los_modos_automaticos_primero():
    ctrls = {"gain": 20, "exposure_time_absolute": 150, "auto_exposure": 1, "extra": 1}
    assert list(camaras.ordenar_controles(ctrls)) == ["auto_exposure", "exposure_time_absolute", "gain", "extra"]


def test_el_valor_anterior_de_exposicion_ya_no_es_el_defecto(tmp_path):
    assert cfg.cargar_config(tmp_path)["rgb"]["controles"]["exposure_time_absolute"] != 130


def test_rpicam_tiene_un_timeout_de_25_s(tmp_path):
    assert cfg.cargar_config(tmp_path)["nir"]["timeout_s"] == 25


@pytest.mark.parametrize("control", ["backlight_compensation", "power_line_frequency"])
def test_los_controles_obligatorios_no_se_pueden_configurar_distinto_de_cero(tmp_path, control):
    """backlight_compensation=1 de fabrica es un realce adaptativo que anula la exposicion fija."""
    (tmp_path / "config_camaras.json").write_text(json.dumps({"rgb": {"controles": {control: 1}}}))
    with pytest.raises(cfg.ConfigInvalida) as exc:
        cfg.cargar_config(tmp_path)
    assert control in str(exc.value) and "0" in str(exc.value)
