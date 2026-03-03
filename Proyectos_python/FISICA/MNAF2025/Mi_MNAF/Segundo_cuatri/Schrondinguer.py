import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Schrondiguer:
    def __init__(self, 
                max=5, 
                min=-5,
                puntos_mallado=1001,
                pasos_sistema=50000,
                pasos_repr=50,
                sigma=0.5,
                k0=10,
                omega = 4
            ):
        self.max = max
        self.min = min
        self.sigma = sigma
        self.k0 = k0
        self.omega = omega
        self.puntos_mallado = puntos_mallado
        self.pasos_sistema = pasos_sistema
        self.pasos_repr = pasos_repr
        self.delta_x = (self.max - self.min) / (self.puntos_mallado - 1)
        self.delta_t = self.delta_x ** 2 / 2
        self.x = np.linspace(self.min, self.max, self.puntos_mallado)
        self.phi_r = np.exp(-0.5 *(self.x/self.sigma) ** 2) * np.cos(self.k0 * self.x)
        self.phi_i = np.exp(-0.5 *(self.x/self.sigma) ** 2) * np.sin(self.k0 * self.x)
        self.V = 0.5 * self.omega**2 * self.x**2
        self.comprobacion()

    def norma(self):
        return np.sum(self.phi_r ** 2 + self.phi_i ** 2) * self.delta_x
    
    def normalizar(self):
        norma = self.norma()
        self.phi_r /= np.sqrt(norma)
        self.phi_i /= np.sqrt(norma)

    def segunda_derivada(self, phi):
        phi2 = np.zeros_like(phi)
        phi2[1:-1] = (phi[2:]+phi[:-2]-2*phi[1:-1])/self.delta_x**2
        phi2[0] = (phi[1]-2*phi[0])/self.delta_x**2
        phi2[-1] = (phi[-2]-2*phi[-1])/self.delta_x**2
        return phi2

    def comprobacion(self):
        norma = self.norma()
        print(f"Norma sin normalizar: {norma}")

        self.normalizar()
        norma_normalizada = self.norma()
        print(f"Norma normalizada: {norma_normalizada}")
        
        return None
    
    def valor_esperado_momento(self):
        valor_derivada_r = (self.phi_r[2:] - self.phi_r[:-2]) / (2*self.delta_x)
        valor_derivada_i = (self.phi_i[2:] - self.phi_i[:-2]) / (2*self.delta_x)
        valor_esperado = np.sum(self.phi_r[1:-1] * valor_derivada_i - self.phi_i[1:-1] * valor_derivada_r) * self.delta_x
        self.valor_esperado_p = valor_esperado
        return valor_esperado
    
    def valor_esperado_momento2(self):
        valor_derivada2_r = self.segunda_derivada(self.phi_r)
        valor_derivada2_i = self.segunda_derivada(self.phi_i)
        valor_esperado = -np.sum(self.phi_r * valor_derivada2_r + self.phi_i * valor_derivada2_i) * self.delta_x
        return valor_esperado

    def valor_esperado_posicion(self,orden=1):
        if orden == 1:
            valor_esperado = np.sum(self.x * (self.phi_r**2 + self.phi_i**2)) * self.delta_x
        elif orden == 2:
            valor_esperado = np.sum(self.x**2 * (self.phi_r**2 + self.phi_i**2)) * self.delta_x
        else:
            raise ValueError("Orden no soportado. Use 1 o 2.")
        return valor_esperado
    
    def Hermite(self, n, x):
        if n == 0:
            return 1
        elif n == 1:
            return 2 * x
        else:
            H_n_minus_2 = 1
            H_n_minus_1 = 2 * x 
            for i in range(2, n+1):
                H_n = 2 * x * H_n_minus_1 - 2 * (i-1) * H_n_minus_2
                H_n_minus_2, H_n_minus_1 = H_n_minus_1, H_n
            return H_n


    def animacion(self,hermite = True):
        print("Animando...")
        fig=plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(2,1,1)
        real, = ax1.plot([], [], label='Parte Real')
        imag, = ax1.plot([], [], label='Parte Imaginaria')
        valor_abs, = ax1.plot([], [], label='Valor Absoluto', color='red')
        potencial, = ax1.plot(self.x, self.V/np.max(self.V), label='Potencial', color='gray', linestyle='--')

        ax1.set_ylim(-1.2, 1.2)
        ax1.set_xlim(self.min, self.max)

        ax1.legend(loc='upper right',framealpha=0.5)
        ax1.set_xlabel('Posición (m)')
        ax1.set_ylabel('Amplitud')

        ax2 = fig.add_subplot(2,3,4)
        ax2.plot([0], [0], label='Norma')
        ax2.legend(loc='upper left',framealpha=0.5)
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('Norma')
        tiempos = []
        normas = []

        ax3 = fig.add_subplot(2,3,5)
        ax3.plot([], [], label=r'$\left < p \right >$', color='orange')
        ax3.plot([0], [0], label=r'$\left < x \right >$', color='blue')
        ax3.legend(loc='upper right',framealpha=0.5)
        ax3.set_xlabel('Tiempo (s)')
        ax3.set_ylabel(r'$\left < p \right >, \left < x \right >$')
        ax3.set_ylim(-11,11)
        tiempos_p = []
        valores_p = []
        tiempos_x = []
        valores_x = []


        ax4 = fig.add_subplot(2,3,6)
        ax4.plot([0], [0], label=r'$\delta p$', color='orange')
        ax4.plot([0], [0], label=r'$\delta x$', color='blue')
        ax4.legend(loc='center right',framealpha=0.5)
        ax4.set_xlabel('Tiempo (s)')
        delta_p = []
        delta_x = []
        producto_delta = []
        tiempos_delta = []

        if hermite:
            self.phi_r = np.exp(-0.5 * self.omega * self.x**2) * self.Hermite(10, np.sqrt(self.omega) * self.x)
            self.phi_i = np.zeros_like(self.phi_r)
            self.normalizar()


        def update(frame):
            for i in range(frame*self.pasos_repr, (frame+1)*self.pasos_repr): 
                self.phi_i[:] = self.phi_i[:] + self.delta_t/4*self.segunda_derivada(self.phi_r) - self.delta_t/2*self.V[:]*self.phi_r[:]
                self.phi_r[:] = self.phi_r[:] - self.delta_t/2*self.segunda_derivada(self.phi_i) + self.delta_t*self.V[:]*self.phi_i[:]
                self.phi_i[:] = self.phi_i[:] + self.delta_t/4*self.segunda_derivada(self.phi_r) - self.delta_t/2*self.V[:]*self.phi_r[:]
                tiempos.append(i * self.delta_t)
                normas.append(self.norma())

                tiempos_p.append(i * self.delta_t)
                valores_p.append(self.valor_esperado_momento())

                tiempos_x.append(i * self.delta_t)
                valores_x.append(self.valor_esperado_posicion())

                tiempos_delta.append(i * self.delta_t)
                delta_p.append(np.sqrt(self.valor_esperado_momento2() - self.valor_esperado_momento()**2))
                delta_x.append(np.sqrt(self.valor_esperado_posicion(orden=2) - self.valor_esperado_posicion()**2))
                producto_delta.append(delta_p[-1] * delta_x[-1])
                if i % self.pasos_repr == 0:
                    real.set_data(self.x, self.phi_r)
                    imag.set_data(self.x, self.phi_i)
                    valor_abs.set_data(self.x, np.sqrt(self.phi_r**2 + self.phi_i**2))
                    
                    ax2.cla()
                    ax2.plot(tiempos, normas, '-',label='Norma')
                    ax2.legend(loc='upper left',framealpha=0.5)
                    ax2.set_xlabel('Tiempo (s)')
                    ax2.set_ylabel('Norma')

                    ax3.cla()
                    ax3.plot(tiempos_p, valores_p, '-', label=r'$\left < p \right >$',color='orange')
                    ax3.plot(tiempos_x, valores_x, '-', label=r'$\left < x \right >$',color='blue')
                    ax3.legend(loc='upper right',framealpha=0.5)
                    ax3.set_ylim(-11,11)
                    ax3.set_xlabel('Tiempo (s)')
                    ax3.set_ylabel(r'$\left < p \right >, \left < x \right >$')

                    ax4.cla()
                    ax4.plot(tiempos_delta, delta_p, '-', label=r'$\Delta p$',color='orange')
                    ax4.plot(tiempos_delta, delta_x, '-', label=r'$\Delta x$',color='blue')
                    ax4.plot(tiempos_delta, producto_delta, '-', label=r'$\Delta p \cdot \Delta x$',color='green')
                    ax4.hlines(0.5, tiempos_delta[0], tiempos_delta[-1], colors='red', linestyles='--')
                    ax4.legend(loc='center right',framealpha=0.5)
                    ax4.set_xlabel('Tiempo (s)')
                    ax4.set_ylabel(r'$\Delta p, \Delta x$')

            return real, imag, valor_abs
        
        self.ani = FuncAnimation(fig, update, frames=self.pasos_sistema//self.pasos_repr, blit=False, repeat=False,interval=1)

        plt.show()


Schrondiguer1=Schrondiguer()
Schrondiguer1.animacion(hermite=True)

