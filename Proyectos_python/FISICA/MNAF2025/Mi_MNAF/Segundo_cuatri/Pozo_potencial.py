
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class PozoPotencial:
    def __init__(self, V0, a):
        self.V0 = V0  # Profundidad del pozo
        self.a = a    # Ancho del pozo

    def potencial(self, x):
        if abs(x) < self.a / 2:
            return -self.V0
        else:
            return 0