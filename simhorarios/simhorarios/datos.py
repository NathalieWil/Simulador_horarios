"""Datos del problema. TODO lo generado aquí es SINTÉTICO (no son datos reales de la Univalle).

Para usar datos reales basta con cargarlos con `cargar_csv` respetando el esquema:
  salones.csv : id, sede, capacidad, tipo, accesible(0/1)
  clases.csv  : id, codigo, nombre, programa, semestre, grupo, docente, dia(0-4),
                inicio(HH:MM), fin(HH:MM), cupo, tipo_req, req_acc(0/1)
"""
from __future__ import annotations
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple
from .rng import GeneradorU01

DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie"]      # SUPUESTO: semana de 5 días
SEDES = ("Bolívar", "Balsas")                     # nombres de sedes tomados del documento del AG


@dataclass(frozen=True)
class Salon:
    id: str
    sede: str
    capacidad: int
    tipo: str          # "aula" | "laboratorio"
    accesible: bool


@dataclass(frozen=True)
class Clase:
    id: int
    codigo: str
    nombre: str
    programa: str
    semestre: int
    grupo: int
    docente: str
    dia: int
    ini: int           # minutos desde 00:00
    fin: int
    cupo: int
    tipo_req: str
    req_acc: bool


@dataclass
class Datos:
    salones: List[Salon]
    clases: List[Clase]
    sintetico: bool = True

    def dominios(self) -> List[List[int]]:
        """Salones factibles por clase (capacidad, tipo, accesibilidad: restricciones 4-6)."""
        dom = []
        for c in self.clases:
            dom.append([k for k, s in enumerate(self.salones)
                        if s.capacidad >= c.cupo and s.tipo == c.tipo_req
                        and (s.accesible or not c.req_acc)])
        return dom

    def validar_entrada(self) -> List[str]:
        """Consistencia de la entrada: un docente o un grupo no pueden tener dos clases a la vez."""
        errs = []
        for clave, nombre in ((lambda c: c.docente, "docente"),
                              (lambda c: (c.programa, c.semestre, c.grupo), "grupo")):
            por: Dict = {}
            for c in self.clases:
                por.setdefault((clave(c), c.dia), []).append(c)
            for (k, d), lst in por.items():
                lst.sort(key=lambda c: c.ini)
                for a, b in zip(lst, lst[1:]):
                    if b.ini < a.fin:
                        errs.append(f"cruce de {nombre} {k} día {d}: clases {a.id} y {b.id}")
        for c, dom in zip(self.clases, self.dominios()):
            if not dom:
                errs.append(f"clase {c.id} sin salón factible (cupo/tipo/accesibilidad)")
        return errs


def hhmm(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


def _min(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def generar_dataset(semilla: int = 2026, n_clases: int = 70, n_docentes: int = 14,
                    aulas_bol: int = 7, labs_bol: int = 1,
                    aulas_bal: int = 5, labs_bal: int = 1) -> Datos:
    g = GeneradorU01(semilla)
    caps = [20, 25, 30, 35, 40, 45, 50]
    salones: List[Salon] = []
    for sede, na, nl, pref in (("Bolívar", aulas_bol, labs_bol, "BOL"),
                               ("Balsas", aulas_bal, labs_bal, "BAL")):
        for i in range(na):
            salones.append(Salon(f"{pref}-A{i+1:02d}", sede, g.choice(caps), "aula", g.random() < 0.4))
        for i in range(nl):
            salones.append(Salon(f"{pref}-L{i+1:02d}", sede, g.choice([20, 25, 30]), "laboratorio", g.random() < 0.5))
    # nos aseguramos de que exista al menos un aula accesible de capacidad alta en cada sede
    for sede in SEDES:
        idx = [k for k, s in enumerate(salones) if s.sede == sede and s.tipo == "aula"]
        k = max(idx, key=lambda k: salones[k].capacidad)
        s = salones[k]
        salones[k] = Salon(s.id, s.sede, max(s.capacidad, 45), s.tipo, True)

    programas = ["P1", "P2", "P3", "P4"]
    cohortes = [(p, sem, 1) for p in programas for sem in (1, 3, 5)]
    tam = {c: g.randint(15, 42) for c in cohortes}
    docentes = [f"Doc{i+1:02d}" for i in range(n_docentes)]
    ocup_doc: Dict[Tuple[str, int], List[Tuple[int, int]]] = {}
    ocup_coh: Dict[Tuple[Tuple, int], List[Tuple[int, int]]] = {}

    def libre(tab, k, ini, fin):
        return all(fin <= a or ini >= b for a, b in tab.get(k, []))

    clases: List[Clase] = []
    intentos = 0
    while len(clases) < n_clases and intentos < 20000:
        intentos += 1
        coh = g.choice(cohortes)
        doc = g.choice(docentes)
        dia = g.randint(0, 4)
        dur = g.choice([120, 120, 120, 180, 60])
        ini = g.randint(7, 18) * 60
        fin = ini + dur
        if fin > 21 * 60 or not libre(ocup_doc, (doc, dia), ini, fin) or not libre(ocup_coh, (coh, dia), ini, fin):
            continue
        lab = g.random() < 0.10
        cupo = min(tam[coh], 30) if lab else tam[coh]
        req_acc = g.random() < 0.08
        tipo = "laboratorio" if lab else "aula"
        # garantizamos factibilidad de la instancia sintética (ajuste de cupo/accesibilidad)
        if not any(s.tipo == tipo and s.capacidad >= cupo and (s.accesible or not req_acc) for s in salones):
            req_acc = False
        if not any(s.tipo == tipo and s.capacidad >= cupo for s in salones):
            cupo = max(s.capacidad for s in salones if s.tipo == tipo)
        # condición necesaria de factibilidad (Hall) en cada instante afectado por la clase nueva
        ok = True
        instantes = {ini} | {c.ini for c in clases if c.dia == dia and ini <= c.ini < fin}
        for t in instantes:
            activas = [(c.cupo, c.tipo_req, c.req_acc) for c in clases if c.dia == dia and c.ini <= t < c.fin]
            activas.append((cupo, tipo, req_acc))
            for ti_ in ("aula", "laboratorio"):          # Hall por tipo de salón
                grupo = [(cu, ra) for (cu, ti, ra) in activas if ti == ti_]
                union = {k for (cu, ra) in grupo for k, sl in enumerate(salones)
                         if sl.tipo == ti_ and sl.capacidad >= cu and (sl.accesible or not ra)}
                if len(grupo) > len(union):
                    ok = False
            if not ok:
                break
        if not ok:
            continue
        n = len(clases)
        clases.append(Clase(n, f"{coh[0]}-S{coh[1]}-{n:03d}", f"Asignatura {n:03d}", coh[0], coh[1], coh[2],
                            doc, dia, ini, fin, cupo, tipo, req_acc))
        ocup_doc.setdefault((doc, dia), []).append((ini, fin))
        ocup_coh.setdefault((coh, dia), []).append((ini, fin))
    d = Datos(salones, clases, True)
    assert not d.validar_entrada(), d.validar_entrada()
    return d


def guardar_csv(d: Datos, ruta_salones: str, ruta_clases: str) -> None:
    with open(ruta_salones, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "sede", "capacidad", "tipo", "accesible"])
        for s in d.salones:
            w.writerow([s.id, s.sede, s.capacidad, s.tipo, int(s.accesible)])
    with open(ruta_clases, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "codigo", "nombre", "programa", "semestre", "grupo", "docente", "dia",
                    "inicio", "fin", "cupo", "tipo_req", "req_acc"])
        for c in d.clases:
            w.writerow([c.id, c.codigo, c.nombre, c.programa, c.semestre, c.grupo, c.docente, c.dia,
                        hhmm(c.ini), hhmm(c.fin), c.cupo, c.tipo_req, int(c.req_acc)])


def cargar_csv(ruta_salones: str, ruta_clases: str) -> Datos:
    with open(ruta_salones, encoding="utf-8") as f:
        salones = [Salon(r["id"], r["sede"], int(r["capacidad"]), r["tipo"], r["accesible"] == "1")
                   for r in csv.DictReader(f)]
    with open(ruta_clases, encoding="utf-8") as f:
        clases = [Clase(int(r["id"]), r["codigo"], r["nombre"], r["programa"], int(r["semestre"]),
                        int(r["grupo"]), r["docente"], int(r["dia"]), _min(r["inicio"]), _min(r["fin"]),
                        int(r["cupo"]), r["tipo_req"], r["req_acc"] == "1")
                  for r in csv.DictReader(f)]
    return Datos(salones, clases, sintetico=False)
