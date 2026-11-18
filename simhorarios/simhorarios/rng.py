"""Fase 1 de la guía: generador de números pseudoaleatorios U(0,1) propio.

Algoritmo: congruencial lineal multiplicativo de Park-Miller (1988),
    x_{n+1} = (a * x_n) mod m,  a = 16807,  m = 2^31 - 1,   u_n = x_n / m.
Es el único generador del proyecto: AG, simulación y pruebas lo usan, de modo
que cambiar la semilla reproduce exactamente toda la ejecución.
"""
from __future__ import annotations
import math
from typing import List, Sequence, TypeVar

T = TypeVar("T")
A = 16807
M = 2_147_483_647  # 2^31 - 1 (primo de Mersenne)


class GeneradorU01:
    def __init__(self, semilla: int = 12345):
        self.semilla_inicial = int(semilla) % M or 1
        self.x = self.semilla_inicial

    # --- núcleo U(0,1) -------------------------------------------------
    def random(self) -> float:
        self.x = (A * self.x) % M
        return self.x / M          # en (0,1): x nunca es 0 ni M

    # --- derivados (transformada inversa / Box-Muller) -------------------
    def uniform(self, a: float, b: float) -> float:
        return a + (b - a) * self.random()

    def randint(self, a: int, b: int) -> int:
        """Entero uniforme en [a, b] (ambos incluidos)."""
        return a + int(self.random() * (b - a + 1))

    def choice(self, seq: Sequence[T]) -> T:
        return seq[int(self.random() * len(seq))]

    def shuffle(self, lst: List[T]) -> None:
        for i in range(len(lst) - 1, 0, -1):      # Fisher-Yates
            j = int(self.random() * (i + 1))
            lst[i], lst[j] = lst[j], lst[i]

    def sample(self, seq: Sequence[T], k: int) -> List[T]:
        pool = list(seq)
        self.shuffle(pool)
        return pool[:k]

    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        u1, u2 = self.random(), self.random()
        return mu + sigma * math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)

    def triangular(self, a: float, c: float, b: float) -> float:
        """Triangular(min=a, moda=c, max=b) por transformada inversa."""
        u = self.random()
        fc = (c - a) / (b - a)
        if u < fc:
            return a + math.sqrt(u * (b - a) * (c - a))
        return b - math.sqrt((1 - u) * (b - a) * (b - c))

    def expovariate(self, tasa: float) -> float:
        return -math.log(self.random()) / tasa

    def bernoulli(self, p: float) -> bool:
        return self.random() < p

    def binomial(self, n: int, p: float) -> int:
        return sum(1 for _ in range(n) if self.random() < p)

    def flujo(self, k: int) -> "GeneradorU01":
        """Sub-flujo independiente k (semillas distintas, reproducibles)."""
        return GeneradorU01((self.semilla_inicial * 48271 + 7919 * (k + 1)) % M)


# --- auditoría estadística del generador ----------------------------------
def auditar(g: GeneradorU01, n: int = 100_000, bins: int = 10) -> dict:
    """Prueba chi-cuadrado de uniformidad, media, varianza y autocorrelación lag-1."""
    xs = [g.random() for _ in range(n)]
    media = sum(xs) / n
    var = sum((x - media) ** 2 for x in xs) / (n - 1)
    cont = [0] * bins
    for x in xs:
        cont[min(int(x * bins), bins - 1)] += 1
    esp = n / bins
    chi2 = sum((o - esp) ** 2 / esp for o in cont)
    num = sum((xs[i] - media) * (xs[i + 1] - media) for i in range(n - 1))
    rho1 = num / sum((x - media) ** 2 for x in xs)
    z_rho = rho1 * math.sqrt(n)                    # ~N(0,1) bajo independencia
    crit = {10: 16.919, 20: 30.144}.get(bins)      # chi2_{0.95, bins-1}
    return {"n": n, "media": media, "media_teorica": 0.5,
            "varianza": var, "varianza_teorica": 1 / 12,
            "chi2": chi2, "chi2_critico_5pct": crit,
            "uniforme_no_rechazada": (chi2 < crit) if crit else None,
            "rho_lag1": rho1, "z_rho": z_rho,
            "independencia_no_rechazada": abs(z_rho) < 1.96}
