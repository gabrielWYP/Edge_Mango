#!/usr/bin/env python3
"""Captura dual RGB/NIR con protocolo de golpes cronometrado (Raspberry Pi).

Reemplaza a `captura_dual.py` (una foto por ejecucion) por un flujo guiado por jornadas:

  captura.py jornada --id J01 --lote L1 --mangos 1-32   # verifica camaras y reloj, calibra
  captura.py sanas                                      # fotos antes del impacto, en orden
  captura.py impactos                                   # hora de cada impacto y fotos +0 h (justo tras el golpe)
  captura.py ronda                                      # fotos a +2, +4, ... +12 h del impacto
  captura.py estado                                     # tabla de avance y proximo vencimiento
  captura.py subir                                      # sube a OCI lo que falte
  captura.py enfoque [--camara rgb|nir]                 # nitidez en vivo para enfocar a mano

Opciones globales (antes o despues del subcomando): --auto (sin Enter, para pruebas) y
--no-subir. El procedimiento completo, los mensajes del QC y el esquema de metadatos estan en
capture/README.md.

Los modulos de esta carpeta se despliegan planos junto a este archivo.
"""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import camaras
import configuracion
import consola
import estado as est
import flujo
import hardware
import protocolo
import subida

COMANDOS_CON_HARDWARE = {"jornada", "sanas", "ronda", "enfoque", "descarte"}
# `impactos` solo usa camaras y luces si la jornada incluye la ronda +0 h (ver `_impactos_toma_fotos`).
CODIGO_PREPARACION, CODIGO_QC, CODIGO_NOIR_TRABADA, CODIGO_CTRL_C = 2, 3, 4, 130


def _globales(parser, subcomando):
    """Opciones globales. En los subcomandos no fijan valor por defecto para no pisar las del padre."""
    def por_defecto(valor):
        return argparse.SUPPRESS if subcomando else valor

    parser.add_argument("--auto", action="store_true", default=por_defecto(False),
                        help="sin pedir Enter (pruebas y ensayos)")
    parser.add_argument("--no-subir", action="store_true", default=por_defecto(False),
                        help="no subir a OCI (los archivos quedan para `captura.py subir`)")
    parser.add_argument("--base", default=por_defecto(str(Path.home() / "capturas")),
                        help=argparse.SUPPRESS)
    parser.add_argument("--simular", action="store_true", default=por_defecto(False),
                        help=argparse.SUPPRESS)


def construir_parser():
    p = argparse.ArgumentParser(prog="captura.py", description=__doc__.split("\n\n")[0])
    _globales(p, subcomando=False)
    sub = p.add_subparsers(dest="comando", required=True, metavar="subcomando")

    def nuevo(nombre, ayuda):
        sp = sub.add_parser(nombre, help=ayuda, description=ayuda)
        _globales(sp, subcomando=True)
        return sp

    def con_id(sp):
        sp.add_argument("--id", help="jornada a usar (por defecto, la activa)")

    j = nuevo("jornada", "crea una jornada, verifica camaras y reloj, y guia la calibracion")
    j.add_argument("--id", required=True, help="identificador de la jornada (carpeta ~/capturas/<id>)")
    j.add_argument("--lote", required=True, help="texto del lote de mangos")
    j.add_argument("--mangos", required=True, help="IDs de mango, por ejemplo 1-32 o 1-5,8")
    j.add_argument("--temperatura-c", type=float, help="temperatura ambiente (se guarda en los metadatos)")
    intervalos_defecto = ",".join(str(h) for h in protocolo.INTERVALOS_DEFECTO_H)
    j.add_argument("--intervalos", default=intervalos_defecto,
                   help=f"horas despues del impacto de cada ronda (por defecto {intervalos_defecto}; "
                        "0 = foto justo tras el impacto, se toma en `impactos`)")
    j.add_argument("--slot-min", type=float, default=protocolo.SLOT_MIN_DEFECTO,
                   help=f"minutos por mango en cada tanda (por defecto {protocolo.SLOT_MIN_DEFECTO:g})")
    j.add_argument("--tolerancia-min", type=float, default=protocolo.TOLERANCIA_MIN_DEFECTO,
                   help="tolerancia de cada ronda en minutos (por defecto 10)")
    j.add_argument("--caras", default="1-4", help="caras a fotografiar (por defecto 1-4)")
    j.add_argument("--semilla", type=int, help="semilla del sorteo de zonas (por defecto, aleatoria)")
    j.add_argument("--unidad-min", type=float, default=protocolo.UNIDAD_MIN_DEFECTO,
                   help=argparse.SUPPRESS)          # oculta: minutos por "hora" (1 = prueba rapida)
    j.add_argument("--ignorar-ntp", action="store_true", help="seguir aunque el reloj no este sincronizado")

    con_id(nuevo("sanas", "fotos de cada cara antes del impacto, en orden (reanuda donde quedo)"))
    con_id(nuevo("impactos", "registra la hora de cada impacto, muestra las zonas sorteadas y toma las "
                              "fotos +0 h (si la jornada las incluye)"))
    r = nuevo("ronda", "fotos de la siguiente ronda que vence, hasta salir con q")
    con_id(r)
    r.add_argument("--forzar", action="store_true", help="capturar aunque falte mas que la tolerancia")
    con_id(nuevo("estado", "tabla de mangos x momentos, proximo vencimiento y descanso"))
    con_id(nuevo("subir", "sube a OCI los archivos pendientes, con reintentos"))
    e = nuevo("enfoque", "nitidez en vivo para enfocar a mano el lente (Ctrl-C para terminar)")
    e.add_argument("--camara", choices=["rgb", "nir"], default="rgb",
                   help="camara a enfocar: rgb con luz blanca, nir con luz de 850 nm")
    e.add_argument("--max-lecturas", type=int, help=argparse.SUPPRESS)
    d = nuevo("descarte", "mide cuantos cuadros descartar tras abrir la camara RGB (avanzado)")
    d.add_argument("--valores", default="0,2,5,10,30", help="cuadros a probar, por ejemplo 0,2,5,10,30")
    return p


def parse_args(argv=None):
    return construir_parser().parse_args(argv)


COMANDOS = {
    "jornada": flujo.cmd_jornada, "sanas": flujo.cmd_sanas, "impactos": flujo.cmd_impactos,
    "ronda": flujo.cmd_ronda, "estado": flujo.cmd_estado, "subir": flujo.cmd_subir,
    "enfoque": flujo.cmd_enfoque, "descarte": flujo.cmd_descarte,
}


def _impactos_toma_fotos(args, base):
    """True si `impactos` debe abrir camaras y luces: la jornada incluye la ronda +0 h.

    Si la jornada no existe o no se puede leer devuelve False: `impactos` informa el problema solo.
    """
    try:
        id_jornada = getattr(args, "id", None) or est.leer_activa(base)
        return bool(id_jornada) and est.Jornada.cargar(base / id_jornada).tiene_ronda_cero()
    except (est.JornadaNoEncontrada, OSError, ValueError, KeyError):
        return False


def _reloj_local():
    return datetime.now(timezone.utc).astimezone()


def main(argv=None, entorno=None):
    """Punto de entrada. `entorno` permite inyectar reloj, esperas, salida y red (pruebas)."""
    entorno = entorno or {}
    args = parse_args(argv)
    salida = entorno.get("salida")
    ui_kw = {"auto": args.auto}
    if salida is not None:
        ui_kw["salida"] = salida
        ui_kw["salida_en_linea"] = lambda texto: None
    if "entrada" in entorno:
        ui_kw["entrada"] = entorno["entrada"]
    if "dormir" in entorno:
        ui_kw["dormir"] = entorno["dormir"]
    ui = consola.Consola(**ui_kw)
    hw = None
    try:
        base = Path(args.base).expanduser()
        config = configuracion.cargar_config(base)
        impactos_con_fotos = args.comando == "impactos" and _impactos_toma_fotos(args, base)
        if args.comando in COMANDOS_CON_HARDWARE or impactos_con_fotos:
            hw = hardware.HardwareSimulado(config) if args.simular else hardware.Hardware(config)
            if entorno.get("preparar_hw"):
                entorno["preparar_hw"](hw)
        par = None
        if (args.comando in ("subir", "jornada", "sanas", "ronda") or impactos_con_fotos) and not (
                args.no_subir and args.comando != "subir"):
            try:
                par, avisos = subida.cargar_par_url(env=entorno.get("env"),
                                                    ruta_archivo=entorno.get("ruta_par") or subida.RUTA_PAR_DEFECTO)
                for aviso in avisos:
                    ui.aviso(aviso)
            except subida.ParNoConfigurada:
                par = None      # `subir` lo informa; las capturas avisan y dejan los archivos en cola
        ctx = flujo.Contexto(
            base=base, config=config, hw=hw, ui=ui, reloj=entorno.get("reloj", _reloj_local),
            dormir=entorno.get("dormir", time.sleep), subir=not args.no_subir,
            put=entorno.get("put"), par=par)
        COMANDOS[args.comando](ctx, args)
        return 0
    except (flujo.ErrorPreparacion, configuracion.ConfigInvalida, camaras.DispositivoNoEncontrado) as e:
        ui.error(str(e))
        return CODIGO_PREPARACION
    except hardware.NIRTrabada as e:
        ui.error(str(e))
        return CODIGO_NOIR_TRABADA
    except hardware.ErrorCamara as e:
        ui.error(str(e))
        return CODIGO_PREPARACION
    except flujo.ErrorQC as e:
        ui.error(f"Captura abortada (modo --auto): {e}")
        return CODIGO_QC
    except KeyboardInterrupt:
        ui.decir("")
        ui.decir("Interrumpido con Ctrl-C. El avance quedó guardado; vuelve a ejecutar el comando para continuar.")
        return CODIGO_CTRL_C
    finally:
        if hw is not None:
            hw.cerrar()


if __name__ == "__main__":
    sys.exit(main())
