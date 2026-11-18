"""Escenarios experimentales reproducibles. Todo número reportado sale de esta ejecución (semillas fijas)."""
from __future__ import annotations
import csv, json, os, statistics as st, time
from dataclasses import replace, asdict
from typing import Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .datos import generar_dataset, guardar_csv, DIAS, hhmm
from .modelo import Problema, Pesos
from .ag import AlgoritmoGenetico, ConfigAG, base_aleatoria, base_voraz
from .rng import GeneradorU01, auditar
from .simulacion import SimuladorHorario, ParamsSim, resumen, co2_estacionario

SEMILLA_DATOS = 2026
SEMILLAS_AG = list(range(101, 111))      # 10 corridas independientes por escenario
N_REPLICAS = 30
SEMILLA_SIM = 777


def _ms(vals):
    return (st.mean(vals), st.pstdev(vals) if len(vals) < 2 else st.stdev(vals))


def correr_ag(P: Problema, cfg: ConfigAG, semillas=SEMILLAS_AG):
    return [AlgoritmoGenetico(P, cfg, GeneradorU01(s)).ejecutar() for s in semillas]


def fila_ag(nombre: str, res: List) -> Dict:
    f = [r.eval["fitness"] for r in res]
    return {"escenario": nombre, "n_corridas": len(res),
            "fitness_media": _ms(f)[0], "fitness_sd": _ms(f)[1], "fitness_min": min(f), "fitness_max": max(f),
            "tasa_factible_%": 100 * sum(r.eval["n_viol"] == 0 for r in res) / len(res),
            "viol_media": st.mean(r.eval["n_viol"] for r in res),
            "N_b_media": st.mean(r.eval["N_b"] for r in res), "N_g_media": st.mean(r.eval["N_g"] for r in res),
            "D_media": st.mean(r.eval["D"] for r in res),
            "seg_media": st.mean(r.segundos for r in res), "evals_media": st.mean(r.evaluaciones for r in res)}


def fila_baseline(nombre, P, xs):
    ev = [P.evaluar(x) for x in xs]
    f = [e["fitness"] for e in ev]
    return {"escenario": nombre, "n_corridas": len(xs), "fitness_media": _ms(f)[0],
            "fitness_sd": _ms(f)[1] if len(f) > 1 else 0.0, "fitness_min": min(f), "fitness_max": max(f),
            "tasa_factible_%": 100 * sum(e["n_viol"] == 0 for e in ev) / len(ev),
            "viol_media": st.mean(e["n_viol"] for e in ev), "N_b_media": st.mean(e["N_b"] for e in ev),
            "N_g_media": st.mean(e["N_g"] for e in ev), "D_media": st.mean(e["D"] for e in ev),
            "seg_media": 0.0, "evals_media": 0}


def simular(P, x, params=None, n=N_REPLICAS):
    S = SimuladorHorario(P, params or ParamsSim())
    reps = S.replicas(x, n, GeneradorU01(SEMILLA_SIM))
    return {k: resumen([r[k] for r in reps]) for k in reps[0]}


def guardar_filas(ruta, filas):
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        for r in filas:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def mejor(res):
    return max(res, key=lambda r: r.eval["fitness"])


def gantt(P, x, ruta, titulo):
    d = P.d
    fig, axs = plt.subplots(1, 5, figsize=(20, 5), sharey=True)
    col = {"Bolívar": "#4c72b0", "Balsas": "#dd8452"}
    for dia, ax in enumerate(axs):
        for c in d.clases:
            if c.dia == dia and x[c.id] >= 0:
                s = d.salones[x[c.id]]
                ax.broken_barh([(c.ini / 60, (c.fin - c.ini) / 60)], (x[c.id] - 0.4, 0.8),
                               facecolors=col[s.sede], edgecolor="white")
        ax.set_title(DIAS[dia]); ax.set_xlim(6.5, 21.5); ax.grid(alpha=.3)
        ax.set_xlabel("hora")
    axs[0].set_yticks(range(len(d.salones))); axs[0].set_yticklabels([s.id for s in d.salones], fontsize=8)
    axs[0].invert_yaxis()
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in col.values()], labels=list(col), loc="upper right")
    fig.suptitle(titulo); fig.tight_layout(); fig.savefig(ruta, dpi=110); plt.close(fig)


def main(salida: str = "resultados", rapido: bool = False):
    os.makedirs(salida, exist_ok=True)
    t_ini = time.time()
    semillas = SEMILLAS_AG[:3] if rapido else SEMILLAS_AG
    nrep = 10 if rapido else N_REPLICAS
    out: Dict = {"semilla_datos": SEMILLA_DATOS, "semillas_ag": semillas, "n_replicas": nrep,
                 "nota": "Datos SINTÉTICOS; parámetros físicos de ParamsSim son SUPUESTOS."}

    # --- 0. auditoría del generador U(0,1) (Fase 1) -----------------------------
    out["auditoria_rng"] = auditar(GeneradorU01(12345), 100_000, 10)

    # --- datos -------------------------------------------------------------------
    datos = generar_dataset(SEMILLA_DATOS, 70)
    guardar_csv(datos, f"{salida}/salones_sinteticos.csv", f"{salida}/clases_sinteticas.csv")
    P = Problema(datos, gap_traslado=120)
    out["instancia"] = {"clases": P.n, "salones": len(datos.salones), "pares_solape": len(P.par_solape),
                        "pares_traslado_doc": len(P.par_traslado_doc), "pares_traslado_grupo": len(P.par_traslado_gru),
                        "pares_hueco<=60": len(P.par_hueco)}

    # --- E1-E5: baselines y variantes del AG (gap=120, como el documento) ------
    cfgs = {
        "AG-documento (50x50, ruleta)": ConfigAG(),
        "AG ampliado (100x200, ruleta)": ConfigAG(pob=100, gens=200),
        "AG ampliado (100x200, torneo)": ConfigAG(pob=100, gens=200, seleccion="torneo"),
        "AG ampliado torneo + mutación dirigida*": ConfigAG(pob=100, gens=200, seleccion="torneo", mut_dirigida=True),
    }
    filas = [fila_baseline("Base aleatoria", P, [base_aleatoria(P, GeneradorU01(s)) for s in semillas]),
             fila_baseline("Base voraz (primer ajuste)", P, [base_voraz(P)])]
    resultados_ag = {}
    for nom, cfg in cfgs.items():
        resultados_ag[nom] = correr_ag(P, cfg, semillas)
        filas.append(fila_ag(nom, resultados_ag[nom]))
    guardar_filas(f"{salida}/E1_comparacion_ag.csv", filas)
    out["E1_comparacion"] = filas

    nom_mejor = max(cfgs, key=lambda n: fila_ag(n, resultados_ag[n])["fitness_media"])
    cfg_mejor = cfgs[nom_mejor]
    out["config_mejor_por_fitness_medio"] = nom_mejor
    r_mejor = mejor(resultados_ag[nom_mejor])
    x_ag = r_mejor.x
    out["mejor_solucion"] = {"eval": r_mejor.eval, "validador_independiente": P.validar(x_ag)}
    with open(f"{salida}/asignacion_mejor.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["clase", "docente", "programa", "semestre", "dia", "inicio", "fin", "cupo", "salon", "sede", "capacidad"])
        for c in datos.clases:
            s = datos.salones[x_ag[c.id]]
            w.writerow([c.id, c.docente, c.programa, c.semestre, DIAS[c.dia], hhmm(c.ini), hhmm(c.fin), c.cupo, s.id, s.sede, s.capacidad])

    # gráficos E1
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for nom, res in resultados_ag.items():
        L = max(len(r.historia) for r in res)
        M = [[r.historia[min(g, len(r.historia) - 1)]["mejor"] for r in res] for g in range(L)]
        mu = [st.mean(m) for m in M]; sd = [st.stdev(m) if len(m) > 1 else 0 for m in M]
        ax[0].plot(range(L), mu, label=nom); ax[0].fill_between(range(L), [a - b for a, b in zip(mu, sd)], [a + b for a, b in zip(mu, sd)], alpha=.15)
    ax[0].set_ylim(40, None); ax[0].set_xlabel("generación"); ax[0].set_ylabel("mejor fitness (media ± sd)"); ax[0].set_title("Convergencia"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    hist = r_mejor.historia
    ax[1].plot([h["gen"] for h in hist], [h["diversidad"] for h in hist], label="diversidad (dist. Hamming media al mejor)")
    ax2 = ax[1].twinx(); ax2.step([h["gen"] for h in hist], [h["pm"] for h in hist], color="C3", label="p_m(t)"); ax2.set_ylabel("p_m", color="C3")
    ax[1].set_xlabel("generación"); ax[1].set_title(f"Mejor corrida de «{nom_mejor[:28]}…»", fontsize=9); ax[1].grid(alpha=.3)
    data = [[r.eval["fitness"] for r in resultados_ag[n]] for n in cfgs]
    ax[2].boxplot(data, labels=["doc", "amp-ruleta", "amp-torneo", "amp-torneo\n+dirigida"]); ax[2].set_title("Fitness final por escenario"); ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{salida}/fig1_convergencia.png", dpi=110); plt.close(fig)
    gantt(P, x_ag, f"{salida}/fig2_horario_mejor.png", f"Mejor horario del AG (datos sintéticos) — fitness {r_mejor.eval['fitness']:.2f}")

    # --- E2: sensibilidad al tiempo mínimo de traslado entre sedes (gap) --------
    filas_gap, sims_gap, x_gap = [], {}, {}
    for gap in (120, 60, 0):
        Pg = Problema(datos, gap_traslado=gap)
        resg = correr_ag(Pg, cfg_mejor, semillas)
        bg = mejor(resg)
        x_gap[gap] = (Pg, bg.x)
        sm = simular(Pg, bg.x, n=nrep)
        fila = {**fila_ag(f"gap={gap} min", resg), "tras_est_total": sm["viajes_est"]["media"],
                "tras_doc_total": sm["viajes_doc"]["media"], "pct_tarde_est": sm["pct_tarde_est"]["media"],
                "pct_tarde_est_ic95": sm["pct_tarde_est"]["ic95"], "pct_tarde_doc": sm["pct_tarde_doc"]["media"],
                "min_tarde_medio": sm["min_tarde"]["media"], "N_b_mejor": bg.eval["N_b"], "viol_mejor": bg.eval["n_viol"]}
        filas_gap.append(fila); sims_gap[gap] = sm
    guardar_filas(f"{salida}/E2_sensibilidad_gap.csv", filas_gap)
    out["E2_gap"] = filas_gap

    # --- E3: sensibilidad a la ventilación (ACH) sobre horarios fijos ----------
    filas_ach = []
    x_vor = base_voraz(P)
    x_rnd = base_aleatoria(P, GeneradorU01(semillas[0]))
    for ach in (6.0, 4.0, 3.0, 2.0):
        for nom, x in (("AG (mejor)", x_ag), ("Voraz", x_vor)):   # el aleatorio es infactible: no se simula
            sm = simular(P, x, ParamsSim(ach=ach), n=nrep)
            filas_ach.append({"ACH": ach, "horario": nom, "co2_pico_max": sm["co2_pico_max"]["media"],
                              "co2_pico_medio": sm["co2_pico_medio"]["media"],
                              "pct_ocup_sobre_umbral": sm["pct_ocup_sobre_umbral"]["media"],
                              "pct_ic95": sm["pct_ocup_sobre_umbral"]["ic95"]})
    guardar_filas(f"{salida}/E3_sensibilidad_ventilacion.csv", filas_ach)
    out["E3_ach"] = filas_ach

    # --- E4: extensión w_ocup (fitness informado por el modelo de aire) ---------
    filas_oc = []
    for w_oc in (0.0, 5.0):
        Pw = Problema(datos, 120, Pesos(w_ocup=w_oc))
        resw = correr_ag(Pw, cfg_mejor, semillas)
        bw = mejor(resw)
        sm = simular(P, bw.x, ParamsSim(ach=3.0), n=nrep)
        ev = P.evaluar(bw.x)
        filas_oc.append({"w_ocup": w_oc, "n_viol": ev["n_viol"], "N_b": ev["N_b"], "N_g": ev["N_g"], "D": ev["D"],
                         "exceso_ocupacion": P.exceso_ocupacion(bw.x),
                         "co2_pico_max": sm["co2_pico_max"]["media"], "pct_ocup_sobre_umbral": sm["pct_ocup_sobre_umbral"]["media"],
                         "pct_ic95": sm["pct_ocup_sobre_umbral"]["ic95"]})
    guardar_filas(f"{salida}/E4_extension_ocupacion.csv", filas_oc)
    out["E4_ocupacion"] = filas_oc

    # --- E5: escalabilidad -------------------------------------------------------
    filas_esc = []
    for n in ((40, 70) if rapido else (40, 70, 100)):
        dn = generar_dataset(SEMILLA_DATOS, n)
        Pn = Problema(dn, 120)
        rn = correr_ag(Pn, cfg_mejor, semillas[:5])
        fr = fila_ag(f"n={len(dn.clases)}", rn)
        fr["clases_generadas"] = len(dn.clases)
        filas_esc.append(fr)
    guardar_filas(f"{salida}/E5_escalabilidad.csv", filas_esc)
    out["E5_escalabilidad"] = filas_esc

    # --- figuras de simulación ---------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    gaps = [f["escenario"] for f in filas_gap]
    ax[0].bar(gaps, [f["pct_tarde_est"] for f in filas_gap], yerr=[f["pct_tarde_est_ic95"] for f in filas_gap], capsize=4)
    ax[0].set_ylabel("% de traslados de estudiantes tardíos"); ax[0].set_title("Riesgo simulado vs. regla de traslado mínimo"); ax[0].grid(alpha=.3)
    ax[1].bar(gaps, [f["N_b_mejor"] for f in filas_gap], color="C1"); ax[1].set_ylabel("clases en Balsas (mejor solución)"); ax[1].set_title("Efecto de relajar la regla en el objetivo"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{salida}/fig3_gap.png", dpi=110); plt.close(fig)

    # trayectoria CO2 en la sala-día con mayor ocupación relativa
    S6, S3 = SimuladorHorario(P, ParamsSim(ach=6.0)), SimuladorHorario(P, ParamsSim(ach=3.0))
    cand = {}
    for c in datos.clases:
        cand.setdefault((x_ag[c.id], c.dia), []).append(c)
    clave = max(cand, key=lambda kd: max(c.cupo / datos.salones[kd[0]].capacidad for c in cand[kd]))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for S, ach, col in ((S6, 6, "C0"), (S3, 3, "C3")):
        _, tz = S.replica(x_ag, GeneradorU01(SEMILLA_SIM).flujo(0), traza_sala=clave)
        ax.plot([t / 3600 for t, _, _ in tz], [c for _, c, _ in tz], color=col, label=f"CO2 simulado, ACH={ach}")
    ax.axhline(ParamsSim().c_umbral, ls="--", color="k", label="umbral de referencia (supuesto)")
    ax.set_xlabel("hora del día"); ax.set_ylabel("CO2 [ppm]"); ax.grid(alpha=.3); ax.legend()
    ax.set_title(f"Sala {datos.salones[clave[0]].id}, {DIAS[clave[1]]}: híbrido DES + EDO (RK4)")
    fig.tight_layout(); fig.savefig(f"{salida}/fig4_co2_sala.png", dpi=110); plt.close(fig)

    out["segundos_totales"] = time.time() - t_ini
    with open(f"{salida}/resultados.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=str)
    return out
