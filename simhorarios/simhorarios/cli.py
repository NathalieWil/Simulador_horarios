"""Uso:
  python -m simhorarios.cli rng                         # auditoría del generador U(0,1)
  python -m simhorarios.cli experimentos [--rapido]     # todos los escenarios -> carpeta resultados/
  python -m simhorarios.cli defensa --ach 3 --n 30 --cap 40 [--g 5.2e-6] [--semilla 7]
  python -m simhorarios.cli gui [--puerto 8765] [--salones s.csv --clases c.csv] [--sin-navegador]
"""
import argparse, json
from .rng import GeneradorU01, auditar
from .simulacion import ParamsSim, rk4, co2_estacionario, co2_analitico


def defensa(a):
    """Defensa «en caliente»: cambia un coeficiente, simula y compara con el valor de equilibrio analítico."""
    p = ParamsSim(ach=a.ach, g_co2=a.g)
    vol = a.cap * p.vol_por_puesto
    lam, k = p.ach / 3600.0, p.g_co2 * 1e6 * a.n / vol
    c, t, h = p.c_ext, 0.0, 30.0
    while t < 6 * 3600:
        c = rk4(lambda C: k + lam * (p.c_ext - C), c, h); t += h
    print(f"ACH={a.ach}  G={a.g:.2e} m3/s  N={a.n}  V={vol:.0f} m3  tau=V/Q={1/lam/60:.1f} min")
    print(f"CO2 simulado a 6 h : {c:8.2f} ppm")
    print(f"CO2 de equilibrio  : {co2_estacionario(a.n, vol, p.ach, p.g_co2, p.c_ext):8.2f} ppm  (C_ext + G*N/Q)")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("rng")
    e = sp.add_parser("experimentos"); e.add_argument("--rapido", action="store_true"); e.add_argument("--salida", default="resultados")
    d = sp.add_parser("defensa"); d.add_argument("--ach", type=float, default=6.0); d.add_argument("--n", type=int, default=30)
    d.add_argument("--cap", type=int, default=40); d.add_argument("--g", type=float, default=5.2e-6); d.add_argument("--semilla", type=int, default=7)
    g = sp.add_parser("gui"); g.add_argument("--puerto", type=int, default=8765); g.add_argument("--salones"); g.add_argument("--clases")
    g.add_argument("--sin-navegador", action="store_true")
    a = ap.parse_args()
    if a.cmd == "gui":
        from .gui import servir
        servir(a.puerto, not a.sin_navegador, a.salones, a.clases)
    elif a.cmd == "rng":
        print(json.dumps(auditar(GeneradorU01(12345)), indent=1))
    elif a.cmd == "experimentos":
        from .experimentos import main as run
        r = run(a.salida, a.rapido); print(f"Listo en {r['segundos_totales']:.1f} s -> {a.salida}/")
    else:
        defensa(a)


if __name__ == "__main__":
    main()
