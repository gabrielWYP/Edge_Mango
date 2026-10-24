"""Interaccion con el operador: Enter para continuar, cuenta regresiva y modo --auto."""
import consola as c


def consola(entradas=(), auto=False, dormir=lambda s: None):
    salida = []
    it = iter(entradas)
    ui = c.Consola(entrada=lambda texto: next(it), salida=salida.append,
                   salida_en_linea=salida.append, auto=auto, dormir=dormir)
    return ui, salida


def test_esperar_devuelve_lo_que_escribe_el_operador():
    ui, _ = consola(["", " q "])
    assert ui.esperar("Coloca y presiona Enter") == ""
    assert ui.esperar("Otra vez") == "q"


def test_en_modo_auto_no_pide_nada_pero_muestra_el_mensaje():
    ui, salida = consola([], auto=True)
    assert ui.esperar("Mango 1 · cara 1 → coloca y presiona Enter") == ""
    assert any("Mango 1" in linea for linea in salida)


def test_decir_aviso_y_error_tienen_prefijos_distintos():
    ui, salida = consola([])
    ui.decir("hola")
    ui.aviso("cuidado")
    ui.error("mal")
    assert salida[0] == "hola" and salida[1].startswith("⚠") and salida[2].startswith("✗")


def test_cuenta_regresiva_duerme_un_segundo_por_vuelta():
    dormidos = []
    ui, salida = consola([], dormir=dormidos.append)
    assert ui.cuenta_regresiva(3, "Esperando") is True
    assert dormidos == [1, 1, 1]
    assert any("Esperando" in linea for linea in salida)


def test_cuenta_regresiva_interrumpida_con_ctrl_c():
    def dormir(s):
        raise KeyboardInterrupt

    ui, _ = consola([], dormir=dormir)
    assert ui.cuenta_regresiva(5, "Esperando") is False


def test_formato_de_la_cuenta_regresiva():
    assert c.formato_cuenta(75) == "01:15"
    assert c.formato_cuenta(3725) == "1:02:05"
    assert c.formato_cuenta(0) == "00:00"
