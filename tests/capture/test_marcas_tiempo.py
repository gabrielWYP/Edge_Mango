"""Los timestamps de archivo deben ser unicos por cara: el pipeline empareja por ellos."""
import marcas_tiempo as mt


def test_siguiente_timestamp_normal():
    assert mt.siguiente_timestamp(100, None) == 100
    assert mt.siguiente_timestamp(200, 100) == 200


def test_siguiente_timestamp_nunca_repite_ni_retrocede():
    assert mt.siguiente_timestamp(100, 100) == 101
    assert mt.siguiente_timestamp(100, 105) == 106  # el reloj retrocedio


def test_generador_con_reloj_congelado_da_consecutivos():
    gen = mt.GeneradorTimestamp(reloj=lambda: 1_000)
    assert [gen.siguiente() for _ in range(3)] == [1_000, 1_001, 1_002]


def test_generador_sigue_al_reloj_cuando_avanza():
    t = iter([10, 10, 50])
    gen = mt.GeneradorTimestamp(reloj=lambda: next(t))
    assert [gen.siguiente() for _ in range(3)] == [10, 11, 50]


def test_generador_respeta_el_ultimo_timestamp_guardado():
    gen = mt.GeneradorTimestamp(reloj=lambda: 100, ultimo=250)
    assert gen.siguiente() == 251


def test_ultimo_timestamp_en_carpeta_lee_los_nombres_de_archivo(tmp_path):
    (tmp_path / "news" / "rgb").mkdir(parents=True)
    (tmp_path / "news" / "nir").mkdir(parents=True)
    (tmp_path / "news" / "rgb" / "mango_rgb_1700000005.jpg").write_bytes(b"x")
    (tmp_path / "news" / "nir" / "mango_nir_1700000009.jpg").write_bytes(b"x")
    (tmp_path / "news" / "rgb" / "otro.txt").write_bytes(b"x")
    assert mt.ultimo_timestamp_en(tmp_path) == 1700000009


def test_ultimo_timestamp_en_carpeta_vacia(tmp_path):
    assert mt.ultimo_timestamp_en(tmp_path) is None
    assert mt.ultimo_timestamp_en(tmp_path / "no_existe") is None
