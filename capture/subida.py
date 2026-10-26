"""Subida a OCI Object Storage con cola persistente y reintentos.

La URL pre-autenticada (PAR) es una credencial: solo se lee de la variable de entorno
`OCI_PAR_URL` o del archivo `~/.config/captura/oci_par_url` (con permisos 600). Nunca
va en el codigo, ni se imprime, ni se escribe en logs, metadatos o mensajes de error
(`ParUrl` oculta su valor en str/repr y los errores se sanean antes de guardarse).

La cola `pendientes.jsonl` es un registro de eventos (solo se agrega al final):
  {"op": "add", "destino": ..., "local": ...}, {"op": "done", ...}, {"op": "lost", ...},
  {"op": "fail", ...}
Al recargarla se reconstruye el estado, de modo que tras una caida o un reinicio solo se
sube lo que falta. Una ultima linea cortada por un apagon se ignora.
"""
import json
import os
import re
import stat
import threading
import time
from pathlib import Path

RUTA_PAR_DEFECTO = Path.home() / ".config" / "captura" / "oci_par_url"
ENV_PAR = "OCI_PAR_URL"
SUBCARPETAS_SUBIBLES = ("news", "calib")
TIMEOUT_HTTP_S = 60


class ParNoConfigurada(Exception):
    pass


class ParUrl:
    """Envuelve la URL pre-autenticada para que no se imprima por accidente."""

    def __init__(self, valor):
        self._valor = valor

    @property
    def valor(self):
        return self._valor

    def __str__(self):
        return "<URL pre-autenticada oculta>"

    __repr__ = __str__


def cargar_par_url(env=None, ruta_archivo=RUTA_PAR_DEFECTO):
    """Devuelve (ParUrl, avisos). Prioridad: variable de entorno y luego archivo."""
    env = os.environ if env is None else env
    avisos = []
    valor = (env.get(ENV_PAR) or "").strip()
    if not valor:
        ruta = Path(ruta_archivo)
        if ruta.exists():
            valor = ruta.read_text(encoding="utf-8").strip()
            if stat.S_IMODE(ruta.stat().st_mode) & 0o077:
                avisos.append(f"{ruta} es legible por otros usuarios; ejecuta: chmod 600 {ruta}")
    if not valor:
        raise ParNoConfigurada(
            f"Falta la URL pre-autenticada de OCI. Define la variable de entorno {ENV_PAR} o "
            f"guardala en {ruta_archivo} (chmod 600). Debe terminar en '/o/'.")
    if not valor.endswith("/"):
        valor += "/"
    return ParUrl(valor), avisos


def sanear(texto, par=None):
    """Quita cualquier URL (y la PAR en particular) de un texto antes de mostrarlo o guardarlo."""
    texto = str(texto)
    if par is not None and par.valor:
        texto = texto.replace(par.valor.rstrip("/"), "<PAR>")
    return re.sub(r"https?://\S+", "<URL>", texto)[:300]


def destino_de(ruta_local, raiz_jornada):
    """Clave en el bucket: replica el layout local (news/rgb/..., calib/...)."""
    return Path(ruta_local).relative_to(raiz_jornada).as_posix()


class ColaSubida:
    def __init__(self, ruta):
        self.ruta = Path(ruta)
        self._lock = threading.RLock()
        self._items = {}   # destino -> dict (en orden de llegada)
        self._cargar()

    def _cargar(self):
        if not self.ruta.exists():
            return
        with open(self.ruta, encoding="utf-8") as f:
            for linea in f:
                try:
                    ev = json.loads(linea)
                except ValueError:
                    continue   # linea cortada por una caida
                self._aplicar(ev)
        with open(self.ruta, "rb+") as f:   # una linea cortada no debe pegarse al siguiente evento
            if f.seek(0, os.SEEK_END) > 0:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    f.write(b"\n")

    def _aplicar(self, ev):
        destino = ev.get("destino")
        op = ev.get("op")
        if op == "add" and destino not in self._items:
            self._items[destino] = {"destino": destino, "local": ev["local"],
                                    "estado": "pendiente", "intentos": 0, "ultimo_error": None}
        elif op in ("done", "lost") and destino in self._items:
            self._items[destino]["estado"] = "hecho" if op == "done" else "perdido"
        elif op == "fail" and destino in self._items:
            self._items[destino]["intentos"] += 1
            self._items[destino]["ultimo_error"] = ev.get("error")

    def _agregar(self, ev):
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ruta, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            f.flush()
        self._aplicar(ev)

    def encolar(self, local, destino):
        with self._lock:
            if destino in self._items:
                return False
            self._agregar({"op": "add", "destino": destino, "local": str(local), "t": int(time.time())})
            return True

    def marcar_hecho(self, destino):
        with self._lock:
            self._agregar({"op": "done", "destino": destino})

    def marcar_perdido(self, destino, motivo):
        with self._lock:
            self._agregar({"op": "lost", "destino": destino, "motivo": motivo})

    def registrar_fallo(self, destino, error):
        with self._lock:
            self._agregar({"op": "fail", "destino": destino, "error": error})

    def pendientes(self):
        with self._lock:
            return [dict(i) for i in self._items.values() if i["estado"] == "pendiente"]

    def estado_de(self, destino):
        """'pendiente' | 'hecho' | 'perdido', o None si el destino nunca se encolo."""
        with self._lock:
            item = self._items.get(destino)
            return item["estado"] if item else None

    def resumen(self):
        with self._lock:
            estados = [i["estado"] for i in self._items.values()]
        return {"pendientes": estados.count("pendiente"), "subidos": estados.count("hecho"),
                "perdidos": estados.count("perdido")}


def encolar_faltantes(cola, raiz_jornada):
    """Encola los archivos de news/ y calib/ que aun no estan en la cola. Devuelve cuantos."""
    raiz = Path(raiz_jornada)
    nuevos = 0
    for sub in SUBCARPETAS_SUBIBLES:
        carpeta = raiz / sub
        if not carpeta.is_dir():
            continue
        for ruta in sorted(carpeta.rglob("*")):
            if ruta.is_file() and not ruta.name.endswith(".tmp"):
                if cola.encolar(ruta, destino_de(ruta, raiz)):
                    nuevos += 1
    return nuevos


def put_requests(url, ruta_local, timeout=TIMEOUT_HTTP_S):
    """PUT del archivo al bucket. Devuelve el codigo HTTP."""
    import requests
    with open(ruta_local, "rb") as f:
        return requests.put(url, data=f, timeout=timeout).status_code


def procesar(cola, par, put=put_requests, max_intentos=3, esperar=time.sleep,
             espera_base_s=2.0, max_fallos_seguidos=3):
    """Intenta subir todo lo pendiente, con reintentos y espera creciente por archivo.

    Si fallan `max_fallos_seguidos` archivos seguidos se corta la pasada (probablemente no hay
    red): lo pendiente se reintenta en la siguiente pasada.
    """
    subidos = perdidos = fallos_seguidos = 0
    errores = []
    for item in cola.pendientes():
        destino, local = item["destino"], item["local"]
        if not os.path.exists(local):
            cola.marcar_perdido(destino, "el archivo local ya no existe")
            perdidos += 1
            continue
        error = None
        for intento in range(1, max_intentos + 1):
            try:
                codigo = put(par.valor + destino, local)
                if 200 <= codigo < 300:
                    error = None
                    break
                error = f"HTTP {codigo}"
            except Exception as e:   # red caida, timeout, DNS...
                error = sanear(f"{type(e).__name__}: {e}", par)
            if intento < max_intentos:
                esperar(min(espera_base_s * 2 ** (intento - 1), 30))
        if error is None:
            cola.marcar_hecho(destino)
            subidos += 1
            fallos_seguidos = 0
        else:
            cola.registrar_fallo(destino, error)
            errores.append(f"{destino}: {error}")
            fallos_seguidos += 1
            if fallos_seguidos >= max_fallos_seguidos:
                break
    return {"subidos": subidos, "pendientes": cola.resumen()["pendientes"],
            "perdidos": perdidos, "errores": errores}


class SubidorFondo:
    """Hilo que sube la cola mientras el operador sigue capturando (la captura nunca espera)."""

    def __init__(self, cola, par, put=put_requests, max_intentos=3, intervalo_s=30,
                 esperar=time.sleep):
        self.cola, self.par, self.put = cola, par, put
        self.max_intentos, self.intervalo_s, self._esperar = max_intentos, intervalo_s, esperar
        self.ultimo_resultado = None
        self._despertar = threading.Event()
        self._parar = threading.Event()
        self._vaciar = False
        self._hilo = threading.Thread(target=self._bucle, name="subidor-fondo", daemon=True)

    def _pasada(self):
        self.ultimo_resultado = procesar(self.cola, self.par, self.put,
                                         max_intentos=self.max_intentos, esperar=self._esperar)

    def _bucle(self):
        while not self._parar.is_set():
            self._pasada()
            self._despertar.wait(self.intervalo_s)   # despierta con notificar() o para reintentar
            self._despertar.clear()
        if self._vaciar:
            self._pasada()

    def iniciar(self):
        self._hilo.start()

    def notificar(self):
        self._despertar.set()

    def detener(self, vaciar=False, limite_s=30):
        self._vaciar = vaciar
        self._parar.set()
        self._despertar.set()
        self._hilo.join(limite_s)
