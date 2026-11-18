"""Motor de simulación HÍBRIDO que evalúa un horario producido por el AG.

 * Eventos discretos (motor propio con cola de prioridad): llegada de estudiantes a cada salón,
   salida al terminar la clase y traslados docentes/estudiantes entre sedes con tiempo aleatorio.
 * Dinámica continua (EDO integrada con RK4 propio): concentración de CO2 de cada salón,
   modelo de mezcla perfecta  dC/dt = G*N(t)/V + (Q/V)*(C_ext - C),  con N(t) cambiando por los eventos.
Todos los parámetros físicos/estocásticos de `ParamsSim` son SUPUESTOS a calibrar (no son datos medidos).
"""
from __future__ import annotations
import heapq, math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
from .modelo import Problema
from .rng import GeneradorU01


@dataclass
class ParamsSim:
    p_asist: float = 0.85                 # prob. de asistencia de cada matriculado        (supuesto)
    llegada: Tuple[float, float, float] = (-10.0, -2.0, 8.0)   # Triangular(min,moda,max) minutos vs inicio (supuesto)
    g_co2: float = 5.2e-6                 # m3/s de CO2 exhalado por persona sedentaria     (supuesto típico, verificar)
    ach: float = 6.0                      # renovaciones de aire por hora                    (supuesto)
    vol_por_puesto: float = 5.0           # m3 de aire por puesto                            (supuesto)
    c_ext: float = 420.0                  # ppm exterior                                     (supuesto)
    c_umbral: float = 1000.0              # ppm de referencia para "exceso"                  (supuesto)
    dt_max: float = 60.0                  # s, paso máximo RK4
    traslado: Tuple[float, float, float] = (20.0, 40.0, 90.0)  # Triangular(min,moda,max) minutos entre sedes (supuesto, medir)


# ---------------- continuo ---------------------------------------------------
def rk4(f: Callable[[float], float], y: float, h: float) -> float:
    """Un paso RK4 para y' = f(y) (autónoma entre eventos: N constante)."""
    k1 = f(y); k2 = f(y + 0.5 * h * k1); k3 = f(y + 0.5 * h * k2); k4 = f(y + h * k3)
    return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def co2_estacionario(n: float, vol: float, ach: float, g: float, c_ext: float) -> float:
    """Valor de equilibrio analítico: C_ss = C_ext + G*N / Q   (Q = ACH*V/3600), en ppm."""
    return c_ext + g * 1e6 * n / (ach / 3600.0 * vol)


def co2_analitico(c0: float, t: float, n: float, vol: float, ach: float, g: float, c_ext: float) -> float:
    css = co2_estacionario(n, vol, ach, g, c_ext)
    return css + (c0 - css) * math.exp(-(ach / 3600.0) * t)


# ---------------- discreto ---------------------------------------------------
class MotorEventos:
    """Reloj de simulación + cola de eventos (heap)."""
    def __init__(self, t0: float = 0.0):
        self.t, self._q, self._n = t0, [], 0

    def programar(self, t: float, accion: Callable[[], None]) -> None:
        self._n += 1
        heapq.heappush(self._q, (t, self._n, accion))

    def correr(self, antes_de_evento: Optional[Callable[[float], None]] = None) -> None:
        while self._q:
            t, _, accion = heapq.heappop(self._q)
            if antes_de_evento:
                antes_de_evento(t)       # el estado continuo avanza hasta el instante del evento
            self.t = t
            accion()


class SimuladorHorario:
    def __init__(self, prob: Problema, params: Optional[ParamsSim] = None):
        self.p, self.par = prob, params or ParamsSim()

    # --- una sala durante un día: híbrido DES + EDO ---------------------------
    def _sala_dia(self, cl_ids: List[int], k: int, rng: GeneradorU01, traza: Optional[list] = None) -> Dict:
        par, cl, sal = self.par, self.p.d.clases, self.p.d.salones[k]
        vol = sal.capacidad * par.vol_por_puesto
        a = par.g_co2 * 1e6 / vol                 # ppm/s por persona
        lam = par.ach / 3600.0                    # 1/s
        est = {"N": 0, "C": par.c_ext, "t": None}
        pico: Dict[int, float] = {c: 0.0 for c in cl_ids}
        ocup_s = [0.0]; sobre_s = [0.0]
        vent = [(cl[c].ini * 60.0, cl[c].fin * 60.0, c) for c in cl_ids]
        motor = MotorEventos()
        for c in cl_ids:
            n_att = rng.binomial(cl[c].cupo, par.p_asist)
            for _ in range(n_att):
                off = rng.triangular(*par.llegada) * 60.0
                motor.programar(cl[c].ini * 60.0 + off, lambda: est.__setitem__("N", est["N"] + 1))
            motor.programar(cl[c].fin * 60.0, lambda n=n_att: est.__setitem__("N", est["N"] - n))
        est["t"] = min(v[0] for v in vent) - 900.0
        if traza is not None:
            traza.append((est["t"], est["C"], est["N"]))

        def avanzar(t_nuevo: float) -> None:
            while est["t"] < t_nuevo - 1e-9:
                h = min(par.dt_max, t_nuevo - est["t"])
                n = est["N"]
                est["C"] = rk4(lambda C: a * n + lam * (par.c_ext - C), est["C"], h)
                est["t"] += h
                if n > 0:
                    ocup_s[0] += h
                    if est["C"] > par.c_umbral:
                        sobre_s[0] += h
                    for (i, f, c) in vent:
                        if i <= est["t"] <= f and est["C"] > pico[c]:
                            pico[c] = est["C"]
                if traza is not None:
                    traza.append((est["t"], est["C"], est["N"]))

        motor.correr(antes_de_evento=avanzar)
        return {"pico": pico, "ocup_s": ocup_s[0], "sobre_s": sobre_s[0]}

    # --- traslados entre sedes: eventos discretos ------------------------------
    def _traslados(self, x: List[int], rng: GeneradorU01) -> Dict:
        par, cl, sd = self.par, self.p.d.clases, self.p.sede_salon
        res = {"viajes_doc": 0, "tarde_doc": 0, "viajes_est": 0, "tarde_est": 0, "min_tarde": 0.0}
        for clave, tipo in ((lambda c: c.docente, "doc"), (lambda c: (c.programa, c.semestre, c.grupo), "est")):
            por: Dict = {}
            for c in cl:
                if x[c.id] >= 0:
                    por.setdefault((clave(c), c.dia), []).append(c)
            for lst in por.values():
                lst.sort(key=lambda c: c.ini)
                for p_, q_ in zip(lst, lst[1:]):
                    if sd[x[p_.id]] == sd[x[q_.id]]:
                        continue
                    sep = q_.ini - p_.fin
                    n_viajeros = 1 if tipo == "doc" else q_.cupo
                    motor = MotorEventos(p_.fin * 60.0)
                    for _ in range(n_viajeros):
                        dur = rng.triangular(*par.traslado)
                        res["viajes_" + tipo] += 1
                        def llega(d=dur):
                            if d > sep:
                                res["tarde_" + tipo] += 1
                                res["min_tarde"] += d - sep
                        motor.programar(p_.fin * 60.0 + dur * 60.0, llega)
                    motor.correr()
        return res

    # --- una réplica semanal ----------------------------------------------------
    def replica(self, x: List[int], rng: GeneradorU01, traza_sala: Optional[Tuple[int, int]] = None):
        cl = self.p.d.clases
        por_sala: Dict[Tuple[int, int], List[int]] = {}
        for c in cl:
            if x[c.id] >= 0:
                por_sala.setdefault((x[c.id], c.dia), []).append(c.id)
        picos, ocup, sobre, traza = [], 0.0, 0.0, None
        for (k, dia), ids in sorted(por_sala.items()):
            tz = [] if traza_sala == (k, dia) else None
            r = self._sala_dia(ids, k, rng, tz)
            picos += list(r["pico"].values()); ocup += r["ocup_s"]; sobre += r["sobre_s"]
            if tz is not None:
                traza = tz
        tr = self._traslados(x, rng)
        m = {"co2_pico_max": max(picos), "co2_pico_medio": sum(picos) / len(picos),
             "pct_ocup_sobre_umbral": 100.0 * sobre / ocup if ocup else 0.0,
             "min_sobre_umbral": sobre / 60.0, **tr,
             "pct_tarde_doc": 100.0 * tr["tarde_doc"] / tr["viajes_doc"] if tr["viajes_doc"] else 0.0,
             "pct_tarde_est": 100.0 * tr["tarde_est"] / tr["viajes_est"] if tr["viajes_est"] else 0.0}
        return (m, traza) if traza_sala is not None else m

    def traza_dia(self, x: List[int], dia: int, rng: GeneradorU01) -> Dict[int, list]:
        """Trayectorias (t[s], CO2[ppm], N) de cada sala con clases en `dia`, para animar en la GUI."""
        por_sala: Dict[int, List[int]] = {}
        for c in self.p.d.clases:
            if x[c.id] >= 0 and c.dia == dia:
                por_sala.setdefault(x[c.id], []).append(c.id)
        out = {}
        for k, ids in sorted(por_sala.items()):
            tz: list = []
            self._sala_dia(ids, k, rng, tz)
            out[k] = tz
        return out

    def replicas(self, x: List[int], n: int, rng_base: GeneradorU01) -> List[Dict]:
        """n réplicas independientes; la réplica r usa el sub-flujo r (números aleatorios comunes entre escenarios)."""
        return [self.replica(x, rng_base.flujo(r)) for r in range(n)]


T_CRIT = {9: 2.262, 14: 2.145, 19: 2.093, 29: 2.045, 49: 2.010}


def resumen(vals: List[float]) -> Dict[str, float]:
    n = len(vals)
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    t = T_CRIT.get(n - 1, 1.96)
    return {"media": m, "sd": sd, "ic95": t * sd / math.sqrt(n) if n > 1 else 0.0, "n": n}
