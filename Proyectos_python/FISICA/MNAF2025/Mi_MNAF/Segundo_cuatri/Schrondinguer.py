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
    

    def animacion(self):
        print("Animando...")
        fig, ax = plt.subplots(2,1, figsize=(10,6))
        real, = ax[0].plot([], [], label='Parte Real')
        imag, = ax[0].plot([], [], label='Parte Imaginaria')
        ax[1].plot([0], [0], label='Norma')

        ax[0].set_ylim(-1.2, 1.2)
        ax[0].set_xlim(self.min, self.max)

        ax[0].legend()
        ax[1].legend()
        ax[1].set_xlabel('Tiempo')
        ax[1].set_ylabel('Norma')
        tiempos = []
        normas = []


        def update(frame):
            for i in range(frame*self.pasos_repr, (frame+1)*self.pasos_repr): 
                self.phi_i[:] = self.phi_i[:] + self.delta_t/4*self.segunda_derivada(self.phi_r) - self.delta_t/2*self.V[:]*self.phi_r[:]
                self.phi_r[:] = self.phi_r[:] - self.delta_t/2*self.segunda_derivada(self.phi_i) + self.delta_t*self.V[:]*self.phi_i[:]
                self.phi_i[:] = self.phi_i[:] + self.delta_t/4*self.segunda_derivada(self.phi_r) - self.delta_t/2*self.V[:]*self.phi_r[:]

                tiempos.append(self.delta_t*i)
                normas.append(self.norma())
                if i % self.pasos_repr == 0:
                    ax[1].cla()
                    real.set_data(self.x, self.phi_r)
                    imag.set_data(self.x, self.phi_i)
                    ax[1].plot(tiempos, normas, '-',label='Norma')
                    ax[1].legend()
                    print(f"Frame: {frame}, Tiempo: {len(tiempos)}, Norma: {len(normas)}")

            return real, imag,
        
        self.ani = FuncAnimation(fig, update, frames=3, blit=False, repeat=False,interval=1)

        plt.show()


Schrondiguer1=Schrondiguer()
Schrondiguer1.animacion()

