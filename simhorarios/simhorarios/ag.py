"""Algoritmo genético adaptativo (selección, cruce multipunto, mutación adaptativa, elitismo)."""
from __future__ import annotations
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from .modelo import Problema, NOASIG
from .rng import GeneradorU01


@dataclass
class ConfigAG:
    pob: int = 50                 # documento: 50
    gens: int = 50                # documento: 50
    prop_heur: float = 0.3        # documento: 0.3 (fracción de individuos que prefieren Balsas)
    estanc_max: int = 15          # documento: 15
    pm_base: float = 0.1          # documento: 0.1
    pm_alta: float = 0.5          # documento: 0.5
    pc: float = 0.9               # SUPUESTO: el documento no da prob. de cruce
    elite: int = 2                # el documento menciona elitismo, sin tamaño
    seleccion: str = "ruleta"     # "ruleta" (prob. proporcional al fitness, como el documento) | "torneo"
    k_torneo: int = 3
    kmax_cruce: int = 3           # nº de puntos de cruce ~ U{1..kmax}
    n_genes_mut: int = 1          # genes alterados por mutación
    mut_validada: bool = True     # acepta el cambio solo si no empeora los conflictos locales
    intentos_mut: int = 5
    mut_dirigida: bool = False    # EXTENSIÓN: con prob. 0.5 muta un gen que está en conflicto
    paro_sin_mejora: Optional[int] = None


@dataclass
class ResultadoAG:
    x: List[int]
    eval: Dict
    historia: List[Dict]
    segundos: float
    evaluaciones: int
    config: Dict = field(default_factory=dict)


class AlgoritmoGenetico:
    def __init__(self, prob: Problema, cfg: ConfigAG, rng: GeneradorU01):
        self.p, self.cfg, self.rng = prob, cfg, rng
        self.n_eval = 0

    # ---------- población inicial ------------------------------------------
    def _individuo(self, heuristico: bool) -> List[int]:
        p, r = self.p, self.rng
        x = []
        for c in range(p.n):
            dom = p.dom[c]
            if heuristico:
                bal = [k for k in dom if p.sede_salon[k] == "Balsas"]
                x.append(r.choice(bal) if bal else r.choice(dom))
            else:
                x.append(r.choice(dom))
        return x

    def _f(self, x):
        self.n_eval += 1
        return self.p.fitness(x)

    # ---------- operadores --------------------------------------------------
    def _seleccionar(self, pop, fit):
        r, cfg = self.rng, self.cfg
        if cfg.seleccion == "torneo":
            idx = [r.randint(0, len(pop) - 1) for _ in range(cfg.k_torneo)]
            return pop[max(idx, key=lambda i: fit[i])]
        fmin = min(fit)
        pesos = [f - fmin + 1e-6 for f in fit]
        u, acc = r.random() * sum(pesos), 0.0
        for ind, w in zip(pop, pesos):
            acc += w
            if acc >= u:
                return ind
        return pop[-1]

    def _cruzar(self, a, b):
        n, r = self.p.n, self.rng
        k = r.randint(1, min(self.cfg.kmax_cruce, n - 1))
        cortes = sorted(r.sample(range(1, n), k))
        h1, h2, prev, swap = [], [], 0, False
        for c in cortes + [n]:
            s1, s2 = (b, a) if swap else (a, b)
            h1 += s1[prev:c]; h2 += s2[prev:c]
            prev, swap = c, not swap
        return h1, h2

    def _mutar(self, x):
        p, r, cfg = self.p, self.rng, self.cfg
        x = x[:]
        for _ in range(cfg.n_genes_mut):
            c = r.randint(0, p.n - 1)
            if cfg.mut_dirigida and r.random() < 0.5:
                en_conf = [j for j in range(p.n) if p.viol_gen(x, j, x[j]) > 0]
                if en_conf:
                    c = r.choice(en_conf)
            dom = p.dom[c]
            if len(dom) < 2:
                continue
            actual = x[c]
            tipo = r.choice(["salon", "sede"])
            sede_act = p.sede_salon[actual]
            if tipo == "salon":
                cand = [k for k in dom if k != actual and p.sede_salon[k] == sede_act] or [k for k in dom if k != actual]
            else:
                cand = [k for k in dom if p.sede_salon[k] != sede_act] or [k for k in dom if k != actual]
            base = p.viol_gen(x, c, actual)
            for _ in range(cfg.intentos_mut if cfg.mut_validada else 1):
                k = r.choice(cand)
                if not cfg.mut_validada or p.viol_gen(x, c, k) <= base:
                    x[c] = k
                    break
        return x

    # ---------- ciclo evolutivo ---------------------------------------------
    def ejecutar(self) -> ResultadoAG:
        cfg, p, r = self.cfg, self.p, self.rng
        t0 = time.perf_counter()
        n_h = int(round(cfg.prop_heur * cfg.pob))
        pop = [self._individuo(i < n_h) for i in range(cfg.pob)]
        fit = [self._f(x) for x in pop]
        pm, mejor_f, sin_mejora = cfg.pm_base, -1e18, 0
        mejor_x, hist = pop[0], []
        for g in range(cfg.gens + 1):
            ib = max(range(len(pop)), key=lambda i: fit[i])
            if fit[ib] > mejor_f + 1e-9:
                mejor_f, mejor_x, sin_mejora = fit[ib], pop[ib][:], 0
                pm = cfg.pm_base
            else:
                sin_mejora += 1
                if sin_mejora >= cfg.estanc_max:
                    pm = cfg.pm_alta                      # adaptación por estancamiento
            ev = p.evaluar(mejor_x)
            div = sum(sum(a != b for a, b in zip(ind, mejor_x)) for ind in pop) / (len(pop) * p.n)
            hist.append({"gen": g, "mejor": mejor_f, "media": sum(fit) / len(fit), "peor": min(fit),
                         "pm": pm, "n_viol": ev["n_viol"], "diversidad": div})
            if g == cfg.gens or (cfg.paro_sin_mejora and sin_mejora >= cfg.paro_sin_mejora):
                break
            orden = sorted(range(len(pop)), key=lambda i: -fit[i])
            nueva = [pop[i][:] for i in orden[:cfg.elite]]
            nfit = [fit[i] for i in orden[:cfg.elite]]
            while len(nueva) < cfg.pob:
                a, b = self._seleccionar(pop, fit), self._seleccionar(pop, fit)
                h1, h2 = self._cruzar(a, b) if r.random() < cfg.pc else (a[:], b[:])
                for h in (h1, h2):
                    if len(nueva) >= cfg.pob:
                        break
                    if r.random() < pm:
                        h = self._mutar(h)
                    nueva.append(h); nfit.append(self._f(h))
            pop, fit = nueva, nfit
        return ResultadoAG(mejor_x, p.evaluar(mejor_x), hist, time.perf_counter() - t0,
                           self.n_eval, asdict(cfg))


# ---------- líneas base (no son datos reales de la institución) -------------
def base_aleatoria(p: Problema, rng: GeneradorU01) -> List[int]:
    return [rng.choice(p.dom[c]) for c in range(p.n)]


def base_voraz(p: Problema) -> List[int]:
    """Primer ajuste: clases por hora de inicio; salón factible libre de menor capacidad.
    Ignora reglas de traslado, como lo haría una asignación manual ingenua. Sin cupo libre -> sin asignar."""
    cl, sal = p.d.clases, p.d.salones
    x = [NOASIG] * p.n
    for c in sorted(range(p.n), key=lambda i: (cl[i].dia, cl[i].ini)):
        for k in sorted(p.dom[c], key=lambda k: (sal[k].capacidad, k)):
            if all(x[j] != k for j in p.vec_solape[c]):
                x[c] = k
                break
    return x
