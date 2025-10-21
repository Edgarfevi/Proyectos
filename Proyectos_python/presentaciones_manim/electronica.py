from manim import *

class ACtoDC(Scene):
    def construct(self):
        # Título
        title = Text("Conversión de Corriente Alterna (CA) a Corriente Continua (CC)", font_size=36)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))

        # Dibujar transformador
        primary_coil = self.create_coil(start=LEFT*4, end=LEFT*2)
        secondary_coil = self.create_coil(start=RIGHT*2, end=RIGHT*4)

        core = Rectangle(height=2.5, width=0.5, color=GRAY).move_to(ORIGIN)
        self.play(Create(primary_coil), Create(secondary_coil), Create(core))
        self.wait(0.5)

        # Señal de entrada (CA)
        ac_wave = self.create_wave(start=-5, end=-2, amplitude=1, color=BLUE)
        ac_label = Text("CA (Entrada)", font_size=24).next_to(ac_wave, DOWN)
        self.play(Create(ac_wave), Write(ac_label))
        self.wait(0.5)

        # Señal de salida antes de rectificar (CA en secundario)
        secondary_wave = self.create_wave(start=2, end=5, amplitude=1, color=YELLOW)
        self.play(Create(secondary_wave))
        self.wait(0.5)

        # Mostrar puente de diodos
        bridge = self.create_diode_bridge().next_to(secondary_coil, RIGHT, buff=1)
        bridge_label = Text("Puente de diodos", font_size=24).next_to(bridge, DOWN)
        self.play(Create(bridge), Write(bridge_label))
        self.wait(0.5)

        # Señal de salida (CC rectificada)
        dc_wave = self.create_rectified_wave(start=6, end=9, amplitude=1, color=RED)
        dc_label = Text("CC (Salida)", font_size=24).next_to(dc_wave, DOWN)
        self.play(Create(dc_wave), Write(dc_label))
        self.wait(1.5)

        # Resumen visual
        self.play(FadeOut(ac_wave, secondary_wave, dc_wave, primary_coil, secondary_coil, bridge, core))
        summary = Tex(r"""
        \textbf{Resumen del proceso:} \\
        1. La CA entra al transformador. \\
        2. Se ajusta el voltaje (más alto o bajo). \\
        3. El puente de diodos convierte la CA en CC pulsante. \\
        4. (Opcional) Un condensador suaviza la señal para obtener CC estable.
        """, font_size=28)
        self.play(Write(summary))
        self.wait(4)

    # Función para crear una bobina
    def create_coil(self, start, end, turns=5, radius=0.2):
        points = []
        length = (end[0] - start[0])
        for i in range(turns * 2 + 1):
            x = start[0] + i * length / (turns * 2)
            y = (radius if i % 2 == 0 else -radius)
            points.append([x, y, 0])
        return VMobject(color=WHITE).set_points_smoothly(points)

    # Onda senoidal (CA)
    def create_wave(self, start=-5, end=5, amplitude=1, color=BLUE, cycles=2):
        return FunctionGraph(
            lambda x: amplitude * np.sin(PI * cycles * (x - start) / (end - start)),
            x_range=[start, end],
            color=color
        )

    # Onda rectificada (CC)
    def create_rectified_wave(self, start=-5, end=5, amplitude=1, color=RED, cycles=2):
        return FunctionGraph(
            lambda x: abs(amplitude * np.sin(PI * cycles * (x - start) / (end - start))),
            x_range=[start, end],
            color=color
        )

    # Puente de diodos (simplificado)
    def create_diode_bridge(self):
        group = VGroup()
        # Diodos (triángulos)
        d1 = Polygon([0,0,0],[0.3,0.2,0],[0.3,-0.2,0], color=ORANGE).shift(UP*0.5+LEFT*0.3)
        d2 = Polygon([0,0,0],[0.3,0.2,0],[0.3,-0.2,0], color=ORANGE).shift(DOWN*0.5+LEFT*0.3)
        d3 = Polygon([0,0,0],[0.3,0.2,0],[0.3,-0.2,0], color=ORANGE).shift(UP*0.5+RIGHT*0.3).rotate(PI)
        d4 = Polygon([0,0,0],[0.3,0.2,0],[0.3,-0.2,0], color=ORANGE).shift(DOWN*0.5+RIGHT*0.3).rotate(PI)
        group.add(d1, d2, d3, d4)
        # Conexiones
        lines = [
            Line(d1.get_left(), d2.get_left(), color=WHITE),
            Line(d1.get_right(), d3.get_left(), color=WHITE),
            Line(d2.get_right(), d4.get_left(), color=WHITE),
            Line(d3.get_right(), d4.get_right(), color=WHITE)
        ]
        group.add(*lines)
        return group
