"""Configuracion de pytest para las pruebas de la herramienta de captura.

Los modulos de `capture/` se despliegan planos en la Raspberry Pi (se ejecuta
`python3 captura.py` desde una sola carpeta), asi que aqui se agrega esa carpeta
al `sys.path` para importarlos con los mismos nombres que usan en la Pi.
"""
import sys
from pathlib import Path

RAIZ_CAPTURA = Path(__file__).resolve().parents[2] / "capture"
if str(RAIZ_CAPTURA) not in sys.path:
    sys.path.insert(0, str(RAIZ_CAPTURA))
