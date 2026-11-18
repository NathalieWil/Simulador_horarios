import math, unittest
from simhorarios.rng import GeneradorU01, auditar
from simhorarios.datos import generar_dataset
from simhorarios.modelo import Problema, NOASIG
from simhorarios.ag import AlgoritmoGenetico, ConfigAG, base_voraz, base_aleatoria
from simhorarios.simulacion import rk4, co2_analitico, co2_estacionario, ParamsSim, SimuladorHorario


class TestRNG(unittest.TestCase):
    def test_rango_y_reproducibilidad(self):
        a, b = GeneradorU01(5), GeneradorU01(5)
        xs = [a.random() for _ in range(1000)]
        self.assertTrue(all(0 < x < 1 for x in xs))
        self.assertEqual(xs, [b.random() for _ in range(1000)])

    def test_park_miller_valor_conocido(self):
        # x_0=1 -> tras 10 000 pasos x = 1043618065 (valor de verificación de Park & Miller, 1988)
        g = GeneradorU01(1)
        for _ in range(10000): g.random()
        self.assertEqual(g.x, 1043618065)

    def test_uniformidad(self):
        r = auditar(GeneradorU01(12345), 50000, 10)
        self.assertTrue(r["uniforme_no_rechazada"] and r["independencia_no_rechazada"])


class TestModelo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = generar_dataset(2026, 70)
        cls.P = Problema(cls.d, 120)

    def test_entrada_consistente(self):
        self.assertEqual(self.d.validar_entrada(), [])

    def test_evaluador_y_validador_coinciden(self):
        for s in range(15):
            x = base_aleatoria(self.P, GeneradorU01(s + 1))
            self.assertEqual(self.P.evaluar(x)["n_viol"], len(self.P.validar(x)))

    def test_detecta_solape_inyectado(self):
        r = AlgoritmoGenetico(self.P, ConfigAG(pob=60, gens=60), GeneradorU01(3)).ejecutar()
        x = r.x[:]
        i, j = self.P.par_solape[0]
        x[i] = x[j]
        self.assertGreater(len(self.P.validar(x)), len(self.P.validar(r.x)) - 1)
        self.assertTrue(any("[solape]" in e for e in self.P.validar(x)))

    def test_ag_reproducible_y_mejora(self):
        cfg = ConfigAG(pob=40, gens=40)
        r1 = AlgoritmoGenetico(self.P, cfg, GeneradorU01(9)).ejecutar()
        r2 = AlgoritmoGenetico(self.P, cfg, GeneradorU01(9)).ejecutar()
        self.assertEqual(r1.x, r2.x)
        self.assertGreater(r1.eval["fitness"], r1.historia[0]["mejor"] - 1e-9)
        self.assertGreaterEqual(r1.historia[-1]["mejor"], r1.historia[0]["mejor"])

    def test_elitismo_monotono(self):
        r = AlgoritmoGenetico(self.P, ConfigAG(pob=40, gens=40), GeneradorU01(4)).ejecutar()
        m = [h["mejor"] for h in r.historia]
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(m, m[1:])))

    def test_dominios_cumplen_restricciones_4_5_6(self):
        r = AlgoritmoGenetico(self.P, ConfigAG(pob=30, gens=20), GeneradorU01(2)).ejecutar()
        v = self.P.evaluar(r.x)["viol"]
        self.assertEqual((v["cap"], v["tipo"], v["acc"], v["exact"]), (0, 0, 0, 0))


class TestSimulacion(unittest.TestCase):
    def test_rk4_vs_analitico(self):
        p, vol, n = ParamsSim(), 200.0, 30
        c = 500.0
        for _ in range(240):
            c = rk4(lambda C: p.g_co2 * 1e6 * n / vol + (p.ach / 3600) * (p.c_ext - C), c, 30.0)
        self.assertAlmostEqual(c, co2_analitico(500.0, 7200, n, vol, p.ach, p.g_co2, p.c_ext), places=4)

    def test_equilibrio_analitico(self):
        p = ParamsSim()
        css = co2_estacionario(30, 200.0, p.ach, p.g_co2, p.c_ext)
        self.assertAlmostEqual(css, p.c_ext + p.g_co2 * 1e6 * 30 / (p.ach / 3600 * 200.0), places=9)

    def test_monotonia_ventilacion(self):
        d = generar_dataset(2026, 70); P = Problema(d, 120); x = base_voraz(P)
        picos = [SimuladorHorario(P, ParamsSim(ach=a)).replica(x, GeneradorU01(8))["co2_pico_max"] for a in (2, 4, 8)]
        self.assertGreater(picos[0], picos[1]); self.assertGreater(picos[1], picos[2])

    def test_semilla_reproduce_replica(self):
        d = generar_dataset(2026, 70); P = Problema(d, 120); x = base_voraz(P)
        S = SimuladorHorario(P)
        self.assertEqual(S.replica(x, GeneradorU01(8)), S.replica(x, GeneradorU01(8)))


if __name__ == "__main__":
    unittest.main()
