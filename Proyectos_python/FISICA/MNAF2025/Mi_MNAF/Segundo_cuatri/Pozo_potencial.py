import numpy as np
import matplotlib.pyplot as plt


class PozoPotencialSolver:
    def __init__(
        self,
        V0=244,
        a=1e-10,
        c=299792458,
        hbar=6.582e-16,
        n=4001,
        umax=4,
    ):
        self.V0 = V0
        self.a = a
        self.c = c
        self.hbar = hbar
        self.n = n
        self.umax = umax

        self.m = 0.511e6 / (self.c ** 2)
        self.k = 2 * self.m * (self.a ** 2) * self.V0 / (self.hbar ** 2)
        # Estimacion teorica del PDF para el estado base: alpha ~ pi^2 / k.
        self.alpha0 = np.pi ** 2 / self.k
        self.alpha_inicial_scan = 0.5 * self.alpha0

        self.u = np.linspace(0, self.umax, self.n) / self.umax
        self.du = self.u[1] - self.u[0]

    def C(self, punto, alpha):
        if np.abs(punto) < 0.5:
            return -self.k * alpha
        return self.k * (1 - alpha)

    def psi_dif(self, alpha, paridad="par"):
        psi, dpsi = (np.zeros(self.n), np.zeros(self.n))

        if paridad == "par":
            psi[0] = 1
            dpsi[0] = 0
        elif paridad == "impar":
            psi[0] = 0
            dpsi[0] = 1
        else:
            raise ValueError("La paridad debe ser 'par' o 'impar'.")

        for i in range(0, self.n - 1):
            psi[i + 1] = psi[i] + dpsi[i] * self.du
            dpsi[i + 1] = dpsi[i] + self.C(self.u[i], alpha) * psi[i] * self.du

        return psi
    
    def _valor_borde(self, alpha, paridad):
        return self.psi_dif(alpha, paridad)[-1]

    def buscar_intervalos_autovalores(self, paridad="par", paso=1e-3):
        a = max(0.0, self.alpha_inicial_scan)
        b = a + paso
        psia = self._valor_borde(a, paridad)
        intervalos = []

        while b <= 1:
            psib = self._valor_borde(b, paridad)

            if psia * psib < 0:
                intervalos.append((a, b))

            a = b
            b = a + paso
            psia = psib

        return intervalos

    def refinar_autovalor_biseccion(self, intervalo, paridad="par", tol=1e-8, max_iter=200):
        a, b = intervalo
        fa = self._valor_borde(a, paridad)
        fb = self._valor_borde(b, paridad)

        if fa * fb > 0:
            raise ValueError("El intervalo no encierra una raiz.")

        for _ in range(max_iter):
            c = 0.5 * (a + b)
            fc = self._valor_borde(c, paridad)

            if abs(fc) < tol or abs(b - a) < tol:
                return c

            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc

        return 0.5 * (a + b)

    def encontrar_autovalores(self, paridad="par", paso_scan=1e-3, tol_biseccion=1e-8):
        intervalos = self.buscar_intervalos_autovalores(paridad=paridad, paso=paso_scan)
        return [
            self.refinar_autovalor_biseccion(itv, paridad=paridad, tol=tol_biseccion)
            for itv in intervalos
        ]

    def onda_completa(self, alpha, paridad="par"):
        psi_pos = self.psi_dif(alpha, paridad)
        u_pos = self.u

        if paridad == "par":
            psi_neg = psi_pos[1:][::-1]
        elif paridad == "impar":
            psi_neg = -psi_pos[1:][::-1]
        else:
            raise ValueError("La paridad debe ser 'par' o 'impar'.")

        u_neg = -u_pos[1:][::-1]
        u_full = np.concatenate([u_neg, u_pos])
        psi_full = np.concatenate([psi_neg, psi_pos])

        max_abs = np.max(np.abs(psi_full))
        if max_abs > 0:
            psi_full = psi_full / max_abs

        return u_full, psi_full

    def resolver_estados_ligados(self, paso_scan=1e-3, tol_biseccion=1e-8):
        pares = self.encontrar_autovalores("par", paso_scan=paso_scan, tol_biseccion=tol_biseccion)
        impares = self.encontrar_autovalores(
            "impar", paso_scan=paso_scan, tol_biseccion=tol_biseccion
        )

        estados = []
        for alpha in pares:
            estados.append({"paridad": "par", "alpha": alpha, "E_eV": alpha * self.V0})
        for alpha in impares:
            estados.append({"paridad": "impar", "alpha": alpha, "E_eV": alpha * self.V0})

        estados.sort(key=lambda x: x["alpha"])
        return estados

    def plot_psi(self, alpha, paridad="par"):
        u_full, psi_full = self.onda_completa(alpha, paridad)
        plt.plot(u_full, psi_full)
        plt.xlabel("u = x/a")
        plt.ylabel("psi(u)")
        plt.title(f"Funcion de onda ({paridad}) para alpha={alpha:.6f}")
        plt.grid(True)
        plt.show()


class NumerovPozoSolver:
    def __init__(self, V0=244, a=1e-10, c=299792458, hbar=6.582e-16, n=4001, umax=4):
        self.V0 = V0
        self.a = a
        self.c = c
        self.hbar = hbar
        self.n = n
        self.umax = umax

        self.m = 0.511e6 / (self.c ** 2)
        self.k = 2 * self.m * (self.a ** 2) * self.V0 / (self.hbar ** 2)
        self.alpha0 = np.pi ** 2 / self.k
        self.alpha_inicial_scan = 0.5 * self.alpha0

        # Malla completa para integrar sin imponer paridad explicita.
        self.u = np.linspace(-self.umax, self.umax, self.n) / self.umax
        self.du = self.u[1] - self.u[0]

    def C(self, punto, alpha):
        if np.abs(punto) < 0.5:
            return -self.k * alpha
        return self.k * (1 - alpha)

    def psi_numerov(self, alpha):
        psi = np.zeros(self.n)
        f = np.array([self.C(ui, alpha) for ui in self.u])
        h2 = self.du ** 2

        psi[0] = 0.0
        psi[1] = 1e-12

        for i in range(1, self.n - 1):
            a = 1 - (h2 * f[i + 1] / 12.0)
            b =(2 - ((2 * h2 * f[i]) / 12.0)+(h2 * f[i])) * psi[i]
            c = (1 - h2 * f[i - 1] / 12.0) * psi[i - 1]
            psi[i + 1] = (b - c) / a

        return psi

    def _valor_borde(self, alpha):
        return self.psi_numerov(alpha)[-1]

    def buscar_intervalos_autovalores(self, paso=1e-3):
        a = max(0.0, self.alpha_inicial_scan)
        b = a + paso
        fa = self._valor_borde(a)
        intervalos = []

        while b <= 1:
            fb = self._valor_borde(b)
            if fa * fb < 0:
                intervalos.append((a, b))
            a, fa = b, fb
            b = a + paso

        return intervalos

    def refinar_autovalor_biseccion(self, intervalo, tol=1e-10, max_iter=300):
        a, b = intervalo
        fa = self._valor_borde(a)
        fb = self._valor_borde(b)

        if fa * fb > 0:
            raise ValueError("El intervalo no encierra una raiz.")

        for _ in range(max_iter):
            c = 0.5 * (a + b)
            fc = self._valor_borde(c)
            if abs(fc) < tol or abs(b - a) < tol:
                return c
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc

        return 0.5 * (a + b)

    def resolver_estados_ligados(self, paso_scan=1e-3, tol_biseccion=1e-10):
        intervalos = self.buscar_intervalos_autovalores(paso=paso_scan)
        alphas = [
            self.refinar_autovalor_biseccion(itv, tol=tol_biseccion)
            for itv in intervalos
        ]
        return [{"alpha": alpha, "E_eV": alpha * self.V0} for alpha in alphas]


class NumerovAtomSolver:
    HARTREE_TO_EV = 27.211386245988

    def __init__(self, r_min=1e-4, r_max=80.0, n=20000):
        self.r_min = r_min
        self.r_max = r_max
        self.n = n
        self.r = np.linspace(r_min, r_max, n)
        self.h = self.r[1] - self.r[0]

    def _potencial_hidrogenoide(self, r, Z=1):
        return -Z / r

    def _potencial_litio_efectivo(self, r):
        # En unidades atomicas: a0 = 1. Continuidad en r=1 -> DeltaV = 2.
        if r < 1.0:
            return -3.0 / r + 2.0
        return -1.0 / r

    def _f_radial(self, r, E, l, tipo="hidrogeno", Z=1):
        if tipo in ("hidrogeno", "hidrogenoide"):
            V = self._potencial_hidrogenoide(r, Z=Z)
        elif tipo == "litio":
            V = self._potencial_litio_efectivo(r)
        else:
            raise ValueError("Tipo no valido: use 'hidrogeno', 'hidrogenoide' o 'litio'.")

        return l * (l + 1) / (r ** 2) + 2 * (V - E)

    def integrar_u(self, E, l=0, tipo="hidrogeno", Z=1):
        u = np.zeros(self.n)
        f = np.array([self._f_radial(ri, E, l, tipo=tipo, Z=Z) for ri in self.r])
        h2 = self.h ** 2

        # Regularidad en el origen: u(r) ~ r^(l+1)
        u[0] = self.r[0] ** (l + 1)
        u[1] = self.r[1] ** (l + 1)

        for i in range(1, self.n - 1):
            a = 1 - (h2 * f[i + 1] / 12.0)
            b = 2 * (1 + 5 * h2 * f[i] / 12.0) * u[i]
            c = (1 - h2 * f[i - 1] / 12.0) * u[i - 1]
            u[i + 1] = (b - c) / a

        return u

    def _valor_borde(self, E, l=0, tipo="hidrogeno", Z=1):
        return self.integrar_u(E, l=l, tipo=tipo, Z=Z)[-1]

    def buscar_intervalos_energia(self, E_min, E_max, paso_E, l=0, tipo="hidrogeno", Z=1):
        E1 = E_min
        E2 = E1 + paso_E
        f1 = self._valor_borde(E1, l=l, tipo=tipo, Z=Z)
        intervalos = []

        while E2 <= E_max:
            f2 = self._valor_borde(E2, l=l, tipo=tipo, Z=Z)
            if f1 * f2 < 0:
                intervalos.append((E1, E2))
            E1, f1 = E2, f2
            E2 = E1 + paso_E

        return intervalos

    def refinar_energia_biseccion(
        self,
        intervalo,
        l=0,
        tipo="hidrogeno",
        Z=1,
        tol=1e-10,
        max_iter=300,
    ):
        a, b = intervalo
        fa = self._valor_borde(a, l=l, tipo=tipo, Z=Z)
        fb = self._valor_borde(b, l=l, tipo=tipo, Z=Z)

        if fa * fb > 0:
            raise ValueError("El intervalo no encierra una raiz.")

        for _ in range(max_iter):
            c = 0.5 * (a + b)
            fc = self._valor_borde(c, l=l, tipo=tipo, Z=Z)
            if abs(fc) < tol or abs(b - a) < tol:
                return c
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc

        return 0.5 * (a + b)

    def encontrar_estados_ligados(
        self,
        n_estados=4,
        E_min=-1.2,
        E_max=-0.01,
        paso_E=2e-3,
        l=0,
        tipo="hidrogeno",
        Z=1,
    ):
        intervalos = self.buscar_intervalos_energia(
            E_min,
            E_max,
            paso_E,
            l=l,
            tipo=tipo,
            Z=Z,
        )
        energias = [
            self.refinar_energia_biseccion(itv, l=l, tipo=tipo, Z=Z)
            for itv in intervalos[:n_estados]
        ]
        return energias

    def energia_analitica_hidrogenoide(self, n, Z=1):
        return -(Z ** 2) / (2 * n ** 2)

    def u_normalizada(self, E, l=0, tipo="hidrogeno", Z=1):
        u = self.integrar_u(E, l=l, tipo=tipo, Z=Z)
        norm = np.sqrt(np.trapezoid(u ** 2, self.r))
        if norm > 0:
            u = u / norm
        return u

    def R_normalizada(self, E, l=0, tipo="hidrogeno", Z=1):
        u = self.u_normalizada(E, l=l, tipo=tipo, Z=Z)
        R = u / self.r
        return R


if __name__ == "__main__":
    print("\033[?25l", end="",flush=True)  # Ocultar cursor]")
    print("\n\033[1m" + "INICIANDO CÁLCULO DE ESTADOS LIGADOS:" + "\033[0m", end="\n\n",)
    print("    Progreso:------------------------- 0%", end="\r",flush=True)
    solver = PozoPotencialSolver()
    estados = solver.resolver_estados_ligados(paso_scan=1e-3, tol_biseccion=1e-10)
    print("    Progreso:\033[32m" + "█████" + "\033[0m" + "-------------------- 20%", end="\r",flush=True)
    solver_num = NumerovPozoSolver()
    estados_num = solver_num.resolver_estados_ligados(paso_scan=1e-3, tol_biseccion=1e-10)
    print("    Progreso:\033[32m" + "██████████" + "\033[0m" + "--------------- 40%", end="\r",flush=True)
    atom_solver = NumerovAtomSolver()
    E_H = atom_solver.encontrar_estados_ligados(
        n_estados=4,
        E_min=-1.2,
        E_max=-0.02,
        paso_E=2e-3,
        l=0,
        tipo="hidrogeno",
        Z=1,
    )
    print("    Progreso:\033[32m" + "███████████████" + "\033[0m" + "---------- 60%", end="\r",flush=True)
    E_He_plus = atom_solver.encontrar_estados_ligados(
        n_estados=4,
        E_min=-4.9,
        E_max=-0.05,
        paso_E=5e-3,
        l=0,
        tipo="hidrogenoide",
        Z=2,
    )
    print("    Progreso:\033[32m" + "█████████████████" + "\033[0m" + "----- 80%", end="\r",flush=True)
    E_Li = atom_solver.encontrar_estados_ligados(
        n_estados=3,
        E_min=-0.6,
        E_max=-0.02,
        paso_E=1e-3,
        l=0,
        tipo="litio",
        Z=1,
    )
    print("    Progreso:\033[32m" + "█████████████████████████" + "\033[0m" + " 100%", end="\n",flush=True)
    print("\n\033[1m" + "PARÁMETROS DEL PROBLEMA:" + "\033[0m")

    print(f"k = {solver.k:.6f}")
    print(f"alpha0 teorico (pi^2/k): {solver.alpha0:.6f}")
    print(f"alpha inicial para scan: {solver.alpha_inicial_scan:.6f}")
    print("\nEstados ligados encontrados:")
    for i, estado in enumerate(estados, start=1):
        print(
            f"n={i:02d} | paridad={estado['paridad']:5s} | "
            f"alpha={estado['alpha']:.10f} | E={estado['E_eV']:.6f} eV"
        )

    print("\nEstados ligados (Numerov, pozo):")
    for i, estado in enumerate(estados_num, start=1):
        print(
            f"n={i:02d} | alpha={estado['alpha']:.10f} | E={estado['E_eV']:.6f} eV"
        )

    print("\nHidrogeno (Numerov radial, l=0):")
    for i, E in enumerate(E_H, start=1):
        E_ana = atom_solver.energia_analitica_hidrogenoide(i, Z=1)
        print(
            f"n={i:02d} | E_num={E:.8f} Ha ({E * atom_solver.HARTREE_TO_EV:.6f} eV) | "
            f"E_ana={E_ana:.8f} Ha"
        )

    print("\nHelio hidrogenoide He+ (Numerov radial, l=0):")
    for i, E in enumerate(E_He_plus, start=1):
        E_ana = atom_solver.energia_analitica_hidrogenoide(i, Z=2)
        print(
            f"n={i:02d} | E_num={E:.8f} Ha ({E * atom_solver.HARTREE_TO_EV:.6f} eV) | "
            f"E_ana={E_ana:.8f} Ha"
        )

    print("\nLitio efectivo (Numerov radial, l=0, potencial por tramos):")
    for i, E in enumerate(E_Li, start=1):
        print(f"n={i:02d} | E_num={E:.8f} Ha ({E * atom_solver.HARTREE_TO_EV:.6f} eV)")


print("\n\033[1m" + "RESULTADOS OBTENIDOS:" + "\033[0m")
print(f"Estados ligados (Pozo potencial, metodo directo): {len(estados)} encontrados.")
print(f"Estados ligados (Pozo potencial, metodo Numerov): {len(estados_num)} encontrados.")
print(f"Estados ligados (Hidrogeno, Numerov radial): {len(E_H)} encontrados.")
print(f"Estados ligados (Helio hidrogenoide, Numerov radial): {len(E_He_plus)} encontrados.")
print(f"Estados ligados (Litio efectivo, Numerov radial): {len(E_Li)} encontrados.")

print("\033[?25h", end="",flush=True)  # Mostrar cursor]