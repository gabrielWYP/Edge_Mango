"""Interaccion con el operador por consola.

Todo (entrada, salida, espera) es inyectable para poder probar los flujos sin teclado.
Con `auto=True` no se pide ningun Enter: sirve para pruebas y ensayos.
"""
import sys
import time


def formato_cuenta(segundos):
    s = int(segundos)
    h, resto = divmod(s, 3600)
    m, s = divmod(resto, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _escribir_en_linea(texto):
    sys.stdout.write(f"\r{texto}   ")
    sys.stdout.flush()


class Consola:
    def __init__(self, entrada=input, salida=print, salida_en_linea=_escribir_en_linea,
                 auto=False, dormir=time.sleep):
        self._entrada = entrada
        self._salida = salida
        self._en_linea = salida_en_linea
        self.auto = auto
        self._dormir = dormir

    def decir(self, texto=""):
        self._salida(texto)

    def aviso(self, texto):
        self._salida(f"⚠ {texto}")

    def error(self, texto):
        self._salida(f"✗ {texto}")

    def esperar(self, texto):
        """Muestra el mensaje y espera Enter. Devuelve lo escrito (en `auto`, siempre '')."""
        if self.auto:
            self._salida(texto)
            return ""
        return self._entrada(texto + " ").strip().lower()

    def cuenta_regresiva(self, segundos, mensaje):
        """Cuenta hacia atras mostrando el tiempo. False si el operador la interrumpe (Ctrl-C)."""
        restante = int(round(segundos))
        try:
            while restante > 0:
                self._en_linea(f"{mensaje}: {formato_cuenta(restante)}")
                self._dormir(1)
                restante -= 1
        except KeyboardInterrupt:
            self._salida("")
            return False
        self._salida("")
        return True
