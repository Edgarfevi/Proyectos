from manim import Scene, Circle, Square, Transform, Rotate, RED, BLUE, GREEN, Text, FadeIn
from manim import PI, DOWN


class CircleToSquare(Scene):
    def construct(self):
        # 1. Creamos un círculo rojo
        circle = Circle(color=RED, fill_opacity=0.5)
        self.play(FadeIn(circle))
        self.wait(0.5)

        # 2. Transformamos el círculo en un cuadrado azul
        square = Square(color=BLUE, fill_opacity=0.5)
        self.play(Transform(circle, square))
        self.wait(0.5)

        # 3. Rotamos el cuadrado y cambiamos de color a verde
        self.play(Rotate(circle, angle=PI/4), circle.animate.set_color(GREEN))
        self.wait(1)

        # 4. Añadimos un texto divertido
        text = Text("¡Manim mola!", font_size=48).next_to(circle, DOWN)
        self.play(FadeIn(text))
        self.wait(2)
