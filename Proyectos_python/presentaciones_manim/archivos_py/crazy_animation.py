from manim import *
import random
import os

class SpiralSquares(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        # Crear un grupo de cuadrados que formen un espiral
        squares = VGroup()
        for i in range(30):
            square = Square(side_length=0.3, color=random.choice([RED, GREEN, BLUE, YELLOW, PURPLE]))
            square.move_to([i*0.1*random.choice([-1,1]), i*0.1*random.choice([-1,1]), 0])
            square.scale(0.5 + i*0.03)
            squares.add(square)

        # Hacerlos aparecer de manera escalonada
        self.play(LaggedStartMap(FadeIn, squares, lag_ratio=0.05))

        # Rotación y crecimiento locos
        for square in squares:
            self.play(
                square.animate.rotate(random.uniform(1, 6)).scale(random.uniform(0.5, 2))
                    .set_color(random.choice([ORANGE, TEAL, PINK, GOLD])),
                run_time=0.3
            )

        # Todos los cuadrados giran alrededor del centro formando un remolino
        self.play(
            Rotate(squares, angle=4*PI),
            squares.animate.shift(UP*0.5 + RIGHT*0.5),
            run_time=2
        )

        # Transformar los cuadrados en texto gigante
        text = Text("¡WHAAAAA!", font_size=96, color=YELLOW)
        self.play(Transform(squares, text))

        # Hacer explotar el texto en colores aleatorios
        colors = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE, PINK]
        self.play(
            text.animate.scale(2).set_color(random.choice(colors)),
            run_time=1.5
        )

        # Desvanecer todo
        self.play(FadeOut(text))


if __name__ == "__main__":
    from manim import config, main

    # Carpeta donde se guardará el video
    output_dir = os.path.join(os.getcwd(), "videos")
    os.makedirs(output_dir, exist_ok=True)

    # Configuración de renderizado
    config.media_dir = output_dir
    config.quality = "low_quality"  # O "high_quality" si quieres mejor resolución

    # Ejecutar Manim y generar el video
    main(["crazy_animation.py", "SpiralSquares"])
