"""Modelo matemático: restricciones duras/blandas y función fitness F = N_asig + R - P.

Cromosoma: lista `x` de longitud n; x[c] = índice de salón de la clase c (o -1 = sin asignar).
El horario (día, inicio, fin) de cada clase es dato fijo (restricción 7 del documento).
La sede de la clase es la del salón elegido, de modo que "cambiar sede" = elegir un salón de otra sede.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from .datos import Datos

NOASIG = -1


@dataclass
class Pesos:
    """Pesos de recompensa (valores del documento) y de penalización (SUPUESTO: el documento no los fija)."""
    r_b: float = 0.1     # recompensa por clase en Balsas            (documento)
    r_g: float = 0.5     # recompensa por clase adicional agrupada   (documento)
    r_d: float = 0.05    # recompensa por hueco pequeño (<=60 min)   (documento)
    w_s: float = 10.0    # solapamiento en salón                      (supuesto)
    w_d: float = 10.0    # traslado docente < gap                     (supuesto)
    w_e: float = 10.0    # traslado grupo < gap                       (supuesto)
    w_cap: float = 10.0
    w_tipo: float = 10.0
    w_acc: float = 10.0
    w_exact: float = 10.0
    w_ocup: float = 0.0  # EXTENSIÓN propia (apagada por defecto): penaliza ocupación > umbral_ocup
    umbral_ocup: float = 0.9
    hueco_max: int = 60  # minutos (documento)


class Problema:
    """Precalcula la estructura de conflictos para evaluar rápido."""

    def __init__(self, datos: Datos, gap_traslado: int = 120, pesos: Pesos | None = None):
        self.d = datos
        self.n = len(datos.clases)
        self.gap = gap_traslado
        self.w = pesos or Pesos()
        self.dom = datos.dominios()
        cl = datos.clases
        self.sede_salon = [s.sede for s in datos.salones]
        # pares que se solapan en el tiempo (comparten salón => violación 1)
        self.par_solape: List[Tuple[int, int]] = []
        # pares del mismo docente / mismo grupo, mismo día, separados por menos de `gap` (sin solape)
        self.par_traslado_doc: List[Tuple[int, int]] = []
        self.par_traslado_gru: List[Tuple[int, int]] = []
        # bloques consecutivos del mismo grupo/día con hueco <= hueco_max
        self.par_hueco: List[Tuple[int, int]] = []
        self.grupo_dia: Dict[Tuple, List[int]] = {}
        for i in range(self.n):
            a = cl[i]
            self.grupo_dia.setdefault(((a.programa, a.semestre, a.grupo), a.dia), []).append(i)
            for j in range(i + 1, self.n):
                b = cl[j]
                if a.dia != b.dia:
                    continue
                if a.ini < b.fin and b.ini < a.fin:
                    self.par_solape.append((i, j))
                    continue
                p, q = (a, b) if a.fin <= b.ini else (b, a)
                sep = q.ini - p.fin
                ip, iq = (i, j) if p is a else (j, i)
                if a.docente == b.docente and sep < self.gap:
                    self.par_traslado_doc.append((ip, iq))
                if (a.programa, a.semestre, a.grupo) == (b.programa, b.semestre, b.grupo):
                    if sep < self.gap:
                        self.par_traslado_gru.append((ip, iq))
        for lst in self.grupo_dia.values():
            lst.sort(key=lambda k: cl[k].ini)
            for p, q in zip(lst, lst[1:]):
                if cl[q].ini - cl[p].fin <= self.w.hueco_max:
                    self.par_hueco.append((p, q))
        # vecindad por clase (para mutación validada y dirigida)
        self.vec_solape = [[] for _ in range(self.n)]
        self.vec_trasl = [[] for _ in range(self.n)]
        for i, j in self.par_solape:
            self.vec_solape[i].append(j); self.vec_solape[j].append(i)
        for i, j in self.par_traslado_doc + self.par_traslado_gru:
            self.vec_trasl[i].append(j); self.vec_trasl[j].append(i)

    # ---------- evaluación ------------------------------------------------
    def violaciones(self, x: List[int]) -> Dict[str, int]:
        sal, sd = self.d.salones, self.sede_salon
        v_s = sum(1 for i, j in self.par_solape if x[i] >= 0 and x[i] == x[j])
        v_d = sum(1 for i, j in self.par_traslado_doc if x[i] >= 0 and x[j] >= 0 and sd[x[i]] != sd[x[j]])
        v_e = sum(1 for i, j in self.par_traslado_gru if x[i] >= 0 and x[j] >= 0 and sd[x[i]] != sd[x[j]])
        v_cap = v_tipo = v_acc = 0
        for c, k in zip(self.d.clases, x):
            if k < 0:
                continue
            s = sal[k]
            v_cap += s.capacidad < c.cupo
            v_tipo += s.tipo != c.tipo_req
            v_acc += (c.req_acc and not s.accesible)
        # v_exact: el bloque horario es dato fijo e inmutable en esta representación -> 0 por construcción
        return {"s": v_s, "d": v_d, "e": v_e, "cap": v_cap, "tipo": v_tipo, "acc": v_acc, "exact": 0}

    def recompensas(self, x: List[int]) -> Dict[str, float]:
        cl, sd = self.d.clases, self.sede_salon
        n_b = sum(1 for k in x if k >= 0 and sd[k] == "Balsas")
        n_g = 0
        for (_, _), lst in self.grupo_dia.items():
            por_sede: Dict[str, int] = {}
            for i in lst:
                if x[i] >= 0:
                    por_sede[sd[x[i]]] = por_sede.get(sd[x[i]], 0) + 1
            n_g += sum(m - 1 for m in por_sede.values())
        # D(i): huecos <= 60 min entre clases consecutivas del grupo EN LA MISMA SEDE.
        # (Sin la condición de sede, D sería constante porque los horarios son fijos: ver informe.)
        d_i = sum(1 for p, q in self.par_hueco if x[p] >= 0 and x[q] >= 0 and sd[x[p]] == sd[x[q]])
        return {"N_b": n_b, "N_g": n_g, "D": d_i}

    def exceso_ocupacion(self, x: List[int]) -> float:
        sal, u = self.d.salones, self.w.umbral_ocup
        return sum(max(0.0, c.cupo / sal[k].capacidad - u) for c, k in zip(self.d.clases, x) if k >= 0)

    def evaluar(self, x: List[int]) -> Dict:
        w = self.w
        v = self.violaciones(x)
        r = self.recompensas(x)
        n_asig = sum(1 for k in x if k >= 0)
        R = w.r_b * r["N_b"] + w.r_g * r["N_g"] + w.r_d * r["D"]
        P = (w.w_s * v["s"] + w.w_d * v["d"] + w.w_e * v["e"] + w.w_cap * v["cap"]
             + w.w_tipo * v["tipo"] + w.w_acc * v["acc"] + w.w_exact * v["exact"]
             + w.w_ocup * self.exceso_ocupacion(x))
        return {"fitness": n_asig + R - P, "n_asig": n_asig, "R": R, "P": P, "viol": v,
                "n_viol": sum(v.values()), **r}

    def fitness(self, x: List[int]) -> float:
        return self.evaluar(x)["fitness"]

    # ---------- validador independiente (otra implementación, O(n^2)) --------
    def validar(self, x: List[int]) -> List[str]:
        """Reglas del documento escritas de forma directa, sin usar las listas precalculadas."""
        cl, sal = self.d.clases, self.d.salones
        errs: List[str] = []
        for i in range(self.n):
            if x[i] < 0:
                continue
            a, sa = cl[i], sal[x[i]]
            if sa.capacidad < a.cupo: errs.append(f"[cap] clase {i} cupo {a.cupo} > cap {sa.capacidad}")
            if sa.tipo != a.tipo_req: errs.append(f"[tipo] clase {i} requiere {a.tipo_req}, salón {sa.tipo}")
            if a.req_acc and not sa.accesible: errs.append(f"[acc] clase {i} requiere salón accesible")
            for j in range(i + 1, self.n):
                if x[j] < 0:
                    continue
                b, sb = cl[j], sal[x[j]]
                if a.dia != b.dia:
                    continue
                if a.ini < b.fin and b.ini < a.fin and x[i] == x[j]:
                    errs.append(f"[solape] clases {i} y {j} en {sa.id}")
                if not (a.ini < b.fin and b.ini < a.fin) and sa.sede != sb.sede:
                    sep = b.ini - a.fin if a.fin <= b.ini else a.ini - b.fin
                    if sep < self.gap:
                        if a.docente == b.docente: errs.append(f"[traslado-doc] {a.docente}: clases {i},{j} sep {sep} min")
                        if (a.programa, a.semestre, a.grupo) == (b.programa, b.semestre, b.grupo):
                            errs.append(f"[traslado-grupo] clases {i},{j} sep {sep} min")
        return errs

    # ---------- violaciones locales de un gen (mutación validada/dirigida) ---
    def viol_gen(self, x: List[int], c: int, k: int) -> int:
        """Nº de conflictos duros de la clase c si se le asignara el salón k."""
        if k < 0:
            return 0
        sd = self.sede_salon
        v = sum(1 for j in self.vec_solape[c] if x[j] == k)
        v += sum(1 for j in self.vec_trasl[c] if x[j] >= 0 and sd[x[j]] != sd[k])
        return v
