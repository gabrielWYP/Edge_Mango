"""Textos de estado para la consola: tabla de mangos x momentos y descripcion de la proxima ronda."""
import protocolo


def duracion_texto(minutos):
    m = int(round(minutos))
    return f"{m // 60} h {m % 60:02d} min" if m >= 60 else f"{m} min"


def etiqueta_ronda(ronda_h):
    return f"+{int(ronda_h) if ronda_h == int(ronda_h) else ronda_h} h"


def descripcion_item(item, ahora, tolerancia_min):
    """'Mango 7 · ronda +4 h · vence 14:32 (faltan 3 min)'."""
    minutos = protocolo.minutos_hasta(ahora, item.vence)
    if minutos >= 1:
        relativo = f"faltan {duracion_texto(minutos)}"
    elif minutos > -1:
        relativo = "es ahora"
    else:
        relativo = f"atrasado {duracion_texto(-minutos)}"
    return (f"Mango {item.mango} · ronda {etiqueta_ronda(item.ronda_h)} · "
            f"vence {item.vence.strftime('%H:%M')} ({relativo})")


def _avance(hechas, total):
    if hechas >= total and total:
        return "✓", ""
    if hechas:
        return "·", f"{hechas}/{total}"
    return "-", ""


def celdas(jornada, ahora):
    """{(mango, momento): (simbolo, texto)} para sano, impacto y cada ronda."""
    vencen = {(i.mango, i.ronda_h): i.vence for i in jornada.calendario()}
    tol = jornada.tolerancia_min
    n_caras = len(jornada.caras)
    resultado = {}
    for m in jornada.mangos:
        resultado[(m, "sano")] = _avance(sum(jornada.sana_hecha(m, c) for c in jornada.caras), n_caras)
        resultado[(m, "impacto")] = _avance(sum(jornada.impacto_hecho(m, c) for c in jornada.caras), n_caras)
        for r in jornada.intervalos_h:
            estado = jornada.estado_ronda(m, r)
            vence = vencen.get((m, r))
            if estado == "omitida":
                celda = ("✗", "omit.")
            elif estado == "fuera_de_tolerancia":
                celda = ("✗", "")
            elif estado == "completa":
                celda = ("✓", "")
            elif vence is None:
                celda = ("-", "")
            elif protocolo.clasificar_ventana(ahora, vence, tol) == "tarde":
                celda = ("✗", "")
            elif estado == "parcial":
                celda = ("·", f"{jornada.n_caras_hechas_ronda(m, r)}/{len(jornada.caras_impactadas(m))}")
            else:
                celda = ("⏳", vence.strftime("%H:%M"))
            resultado[(m, r)] = celda
    return resultado


def _texto_celda(celda):
    simbolo, texto = celda
    return f"{simbolo}{texto}"


def tabla_estado(jornada, ahora, cola=None):
    d = jornada.datos
    tol = jornada.tolerancia_min
    cab = (f"Jornada {d['id']} · lote {d['lote']} · {len(jornada.mangos)} mangos · "
           f"slot {d['slot_min']:g} min · tolerancia ±{tol:g} min")
    lineas = [cab]
    if jornada.unidad_min != protocolo.UNIDAD_MIN_DEFECTO:
        lineas.append(f"** PRUEBA: 1 'hora' del protocolo = {jornada.unidad_min:g} min **")
    lineas.append("Leyenda: ✓ hecho · ⏳ hora de vencimiento · ✗ perdido o fuera de tolerancia · "
                  "· parcial (caras hechas/total) · - sin datos")
    momentos = ["sano", "impacto"] + list(jornada.intervalos_h)
    nombres = ["sano", "imp"] + [etiqueta_ronda(r) for r in jornada.intervalos_h]
    ancho = 8
    lineas.append("Mango  " + "".join(n.ljust(ancho) for n in nombres))
    tabla = celdas(jornada, ahora)
    for m in jornada.mangos:
        lineas.append(f"{m:>5}  " + "".join(_texto_celda(tabla[(m, mo)]).ljust(ancho) for mo in momentos))

    lineas.append("")
    siguiente = jornada.siguiente_pendiente()
    if siguiente is not None:
        lineas.append("Siguiente: " + descripcion_item(siguiente, ahora, tol))
        minutos = protocolo.minutos_hasta(ahora, siguiente.vence)
        if minutos >= 1:
            lineas.append(f"Descanso: faltan {duracion_texto(minutos)} para el siguiente vencimiento.")
        else:
            lineas.append("Descanso: ninguno, ya hay una ronda lista para capturar.")
    elif not jornada.impactos_iso():
        lineas.append("Aun no hay impactos registrados: faltan los impactos para programar las rondas.")
    else:
        lineas.append("Nada pendiente en las rondas.")
    sin_impacto = len(jornada.pendientes_impactos())
    if jornada.impactos_iso() and sin_impacto:
        lineas.append(f"Impactos sin registrar: {sin_impacto} caras.")
    if cola is not None:
        lineas.append(f"Subida: {cola['pendientes']} pendientes · {cola['subidos']} subidos · "
                      f"{cola['perdidos']} perdidos")
    return lineas
