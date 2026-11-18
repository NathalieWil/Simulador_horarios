"""Servidor local de la GUI (solo biblioteca estándar). Abre http://127.0.0.1:8765 en el navegador."""
from __future__ import annotations
import json, math, mimetypes, os, threading, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .datos import generar_dataset, cargar_csv, DIAS
from .modelo import Problema, Pesos
from .ag import AlgoritmoGenetico, ConfigAG
from .rng import GeneradorU01, auditar
from .simulacion import SimuladorHorario, ParamsSim, rk4, co2_estacionario, co2_analitico, resumen

AQUI = os.path.dirname(os.path.abspath(__file__))
SALIDA = os.path.abspath("resultados")
E = {"P": None, "x": None, "datos": None, "real": None, "exp": "inactivo", "exp_msg": ""}
LOCK = threading.Lock()


def _cl(v, lo, hi, tipo=float):
    return max(lo, min(hi, tipo(v)))


def _params(b) -> ParamsSim:
    return ParamsSim(p_asist=_cl(b.get("asistencia", 85), 1, 100) / 100.0,
                     ach=_cl(b.get("ach", 6), 0.1, 60), g_co2=_cl(b.get("g", 5.2), 0.1, 50) * 1e-6,
                     c_umbral=_cl(b.get("umbral", 1000), 500, 5000),
                     traslado=(_cl(b.get("t_min", 20), 1, 300), _cl(b.get("t_moda", 40), 1, 300), _cl(b.get("t_max", 90), 1, 300)))


def api_optimizar(b):
    if b.get("usar_real") and E["real"] is not None:
        datos = E["real"]
    else:
        datos = generar_dataset(_cl(b.get("semilla_datos", 2026), 1, 2_000_000_000, int), _cl(b.get("n_clases", 70), 10, 150, int))
    P = Problema(datos, _cl(b.get("gap", 120), 0, 600, int), Pesos(w_ocup=_cl(b.get("w_ocup", 0), 0, 100)))
    cfg = ConfigAG(pob=_cl(b.get("pob", 100), 10, 300, int), gens=_cl(b.get("gens", 200), 1, 500, int),
                   seleccion="torneo" if b.get("seleccion") == "torneo" else "ruleta",
                   mut_dirigida=bool(b.get("dirigida", False)))
    r = AlgoritmoGenetico(P, cfg, GeneradorU01(_cl(b.get("semilla_ag", 101), 1, 2_000_000_000, int))).ejecutar()
    E.update(P=P, x=r.x, datos=datos)
    return {"instancia": {"clases": P.n, "salones": len(datos.salones), "sintetico": datos.sintetico},
            "salones": [{"id": s.id, "sede": s.sede, "cap": s.capacidad, "tipo": s.tipo, "acc": s.accesible} for s in datos.salones],
            "clases": [{"id": c.id, "docente": c.docente, "prog": c.programa, "sem": c.semestre, "dia": c.dia,
                        "ini": c.ini, "fin": c.fin, "cupo": c.cupo, "salon": r.x[c.id]} for c in datos.clases],
            "eval": r.eval, "validador": P.validar(r.x)[:20], "segundos": r.segundos, "evaluaciones": r.evaluaciones,
            "historia": [{k: h[k] for k in ("gen", "mejor", "media", "pm", "diversidad", "n_viol")} for h in r.historia]}


def api_simular(b):
    if E["x"] is None:
        raise ValueError("Primero ejecuta el AG en la pestaña «Optimizar».")
    S = SimuladorHorario(E["P"], _params(b))
    reps = S.replicas(E["x"], _cl(b.get("reps", 30), 1, 100, int), GeneradorU01(_cl(b.get("semilla", 777), 1, 2_000_000_000, int)))
    return {"n": len(reps), "resumen": {k: resumen([r[k] for r in reps]) for k in reps[0]}}


def api_traza(b):
    if E["x"] is None:
        raise ValueError("Primero ejecuta el AG en la pestaña «Optimizar».")
    p = _params(b)
    dia = _cl(b.get("dia", 0), 0, 4, int)
    tr = SimuladorHorario(E["P"], p).traza_dia(E["x"], dia, GeneradorU01(_cl(b.get("semilla", 777), 1, 2_000_000_000, int)))
    sal = E["datos"].salones
    return {"dia": dia, "c_ext": p.c_ext, "umbral": p.c_umbral, "lam": p.ach / 3600.0,
            "salas": [{"k": k, "id": sal[k].id, "sede": sal[k].sede, "cap": sal[k].capacidad,
                       "traza": [[round(t, 1), round(c, 1), n] for t, c, n in tz]} for k, tz in tr.items()]}


def api_defensa(b):
    p = _params(b)
    n, cap, horas = _cl(b.get("n", 40), 1, 500, int), _cl(b.get("cap", 40), 1, 500, int), _cl(b.get("horas", 6), 0.5, 24)
    vol = cap * p.vol_por_puesto
    lam, k = p.ach / 3600.0, p.g_co2 * 1e6 * n / vol
    c, t, h = p.c_ext, 0.0, 30.0
    ts, sim, ana = [0.0], [c], [c]
    while t < horas * 3600 - 1e-9:
        c = rk4(lambda C: k + lam * (p.c_ext - C), c, h)
        t += h
        if int(round(t / h)) % 2 == 0:
            ts.append(t / 60.0); sim.append(c)
            ana.append(co2_analitico(p.c_ext, t, n, vol, p.ach, p.g_co2, p.c_ext))
    return {"t_min": ts, "sim": sim, "ana": ana, "css": co2_estacionario(n, vol, p.ach, p.g_co2, p.c_ext),
            "tau_min": 1 / lam / 60.0, "vol": vol, "final_sim": sim[-1], "umbral": p.c_umbral}


def _lanzar_experimentos(rapido: bool):
    from .experimentos import main as run
    try:
        E["exp"], E["exp_msg"] = "corriendo", "Ejecutando escenarios…"
        r = run(SALIDA, rapido)
        E["exp"], E["exp_msg"] = "listo", f"Terminado en {r['segundos_totales']:.0f} s"
    except Exception as e:   # noqa
        E["exp"], E["exp_msg"] = "error", str(e)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        ruta = self.path.split("?")[0]
        if ruta in ("/", "/index.html"):
            return self._send(200, open(os.path.join(AQUI, "web", "index.html"), "rb").read(), "text/html; charset=utf-8")
        if ruta == "/api/auditoria":
            return self._send(200, auditar(GeneradorU01(12345), 50000, 10))
        if ruta == "/api/estado":
            return self._send(200, {"exp": E["exp"], "msg": E["exp_msg"], "real": E["real"] is not None,
                                    "hay_horario": E["x"] is not None})
        if ruta == "/api/resultados":
            fs = sorted(os.listdir(SALIDA)) if os.path.isdir(SALIDA) else []
            return self._send(200, {"archivos": fs})
        if ruta.startswith("/resultados/"):
            f = os.path.join(SALIDA, os.path.basename(ruta))
            if os.path.isfile(f):
                return self._send(200, open(f, "rb").read(), mimetypes.guess_type(f)[0] or "application/octet-stream")
        self._send(404, {"error": "no encontrado"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            b = json.loads(self.rfile.read(n) or b"{}")
            ruta = self.path
            if ruta == "/api/experimentos":
                if E["exp"] == "corriendo":
                    return self._send(200, {"msg": "ya hay una ejecución en curso"})
                threading.Thread(target=_lanzar_experimentos, args=(bool(b.get("rapido")),), daemon=True).start()
                return self._send(200, {"msg": "iniciado"})
            f = {"/api/optimizar": api_optimizar, "/api/simular": api_simular,
                 "/api/traza": api_traza, "/api/defensa": api_defensa}.get(ruta)
            if f is None:
                return self._send(404, {"error": "ruta desconocida"})
            with LOCK:
                self._send(200, f(b))
        except Exception as e:   # noqa
            self._send(400, {"error": str(e)})


def servir(puerto: int = 8765, abrir: bool = True, salones: str | None = None, clases: str | None = None):
    if salones and clases:
        E["real"] = cargar_csv(salones, clases)
        errs = E["real"].validar_entrada()
        print(f"Datos reales cargados: {len(E['real'].clases)} clases, {len(E['real'].salones)} salones. Problemas de entrada: {len(errs)}")
        for e in errs[:10]:
            print("  -", e)
    srv = ThreadingHTTPServer(("127.0.0.1", puerto), H)
    url = f"http://127.0.0.1:{puerto}"
    print(f"GUI en {url}   (Ctrl+C para cerrar)")
    if abrir:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nCerrado.")
