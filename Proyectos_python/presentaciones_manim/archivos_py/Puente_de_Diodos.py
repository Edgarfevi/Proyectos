from manim import *

class Diode(VGroup):
    """Diodo estilizado profesional"""
    def __init__(self, orientation=1, label_text="", **kwargs):
        super().__init__(**kwargs)
        # Triángulo del diodo
        tri = Polygon(ORIGIN, 0.6*UP + 0.3*RIGHT, 0.6*DOWN + 0.3*RIGHT)
        tri.set_fill(GRAY, opacity=0.7)
        tri.set_stroke(BLACK, width=2)
        # Barra
        bar = Line(0.3*RIGHT + 0.6*UP, 0.3*RIGHT + 0.6*DOWN)
        bar.set_stroke(BLACK, width=3)
        group = VGroup(tri, bar)
        if orientation == -1:
            group.rotate(PI)
        self.add(group)
        # Etiqueta opcional
        if label_text:
            lbl = Text(label_text, font_size=20)
            lbl.next_to(group, UP, buff=0.1)
            self.add(lbl)

class DiodeBridgeRhomboidScene(Scene):
    def construct(self):
        # Título
        title = Text("Puente de diodos: CA → CC", font_size=36)
        self.play(Write(title))
        self.wait(0.5)
        self.play(FadeOut(title))

        # Centro del puente
        center = ORIGIN

        # Transformador (entrada AC)
        sec_top = Dot(LEFT*4 + UP*0.5, color=BLUE)
        sec_bottom = Dot(LEFT*4 + DOWN*0.5, color=BLUE)
        sec_label = Text("Secundario\nTransformador", font_size=18).next_to(sec_top, LEFT)

        # Líneas de entrada
        wire_top = Line(sec_top.get_center(), center + UP*1.5)
        wire_bottom = Line(sec_bottom.get_center(), center + DOWN*1.5)

        # Diodos del puente en forma de rombo
        offset = 2  # tamaño del rombo
        D1 = Diode(1, "D1").move_to(center + UP*offset)
        D2 = Diode(-1, "D2").move_to(center + RIGHT*offset)
        D3 = Diode(1, "D3").move_to(center + DOWN*offset)
        D4 = Diode(-1, "D4").move_to(center + LEFT*offset)

        # Conexiones del rombo
        top_right = Line(D1.get_bottom(), D2.get_left())
        right_bottom = Line(D2.get_right(), D3.get_top())
        bottom_left = Line(D3.get_bottom(), D4.get_right())
        left_top = Line(D4.get_left(), D1.get_top())

        # Salida
        out_plus = Dot(center + RIGHT*offset + UP*0.1, color=RED)
        out_minus = Dot(center + LEFT*offset + DOWN*0.1, color=BLUE)
        plus_sign = Text("+", font_size=28, color=RED).next_to(out_plus, UP)
        minus_sign = Text("-", font_size=28, color=BLUE).next_to(out_minus, DOWN)

        # Resistencia de carga
        load = Rectangle(width=0.6, height=1.2, color=ORANGE).next_to(out_plus, RIGHT, buff=0.5)
        load_label = Text("R_L", font_size=20).move_to(load.get_center())
        load_top = Line(out_plus.get_center(), load.get_left())
        load_bottom = Line(out_minus.get_center(), load.get_left() + DOWN*0.6)
        load_connector = Line(load.get_right() + UP*0.6, load.get_right() - UP*0.6)

        # Agrupar todo
        components = VGroup(
            sec_top, sec_bottom, sec_label,
            wire_top, wire_bottom,
            D1,D2,D3,D4,
            top_right, right_bottom, bottom_left, left_top,
            out_plus, out_minus, plus_sign, minus_sign,
            load, load_label, load_top, load_bottom, load_connector
        )
        self.play(Create(components))
        self.wait(0.5)

        # Semiperíodo positivo
        pos_text = Text("Semiperiodo positivo", font_size=24).to_corner(UR)
        self.play(Write(pos_text))

        conducting = VGroup(D1,D3)
        self.play(*(Flash(d, flash_radius=0.5) for d in conducting))

        arrows_pos = VGroup(
            Arrow(sec_top.get_center(), D1.get_bottom(), buff=0.05),
            Arrow(D1.get_bottom(), out_plus.get_center(), buff=0.05),
            Arrow(out_plus.get_center(), load.get_left(), buff=0.05),
            Arrow(load.get_right(), out_minus.get_center(), buff=0.05),
            Arrow(out_minus.get_center(), D3.get_top(), buff=0.05),
            Arrow(D3.get_top(), sec_bottom.get_center(), buff=0.05)
        )
        self.play(*[GrowArrow(a) for a in arrows_pos])
        self.wait(1)
        self.play(*[FadeOut(a) for a in arrows_pos])

        # Semiperíodo negativo
        neg_text = Text("Semiperiodo negativo", font_size=24).to_corner(UR)
        self.play(Transform(pos_text, neg_text))

        conducting = VGroup(D2,D4)
        self.play(*(Flash(d, flash_radius=0.5) for d in conducting))

        arrows_neg = VGroup(
            Arrow(sec_bottom.get_center(), D4.get_right(), buff=0.05),
            Arrow(D4.get_right(), out_plus.get_center(), buff=0.05),
            Arrow(out_plus.get_center(), load.get_left(), buff=0.05),
            Arrow(load.get_right(), out_minus.get_center(), buff=0.05),
            Arrow(out_minus.get_center(), D2.get_left(), buff=0.05),
            Arrow(D2.get_left(), sec_top.get_center(), buff=0.05)
        )
        self.play(*[GrowArrow(a) for a in arrows_neg])
        self.wait(1)
        self.play(*[FadeOut(a) for a in arrows_neg])

        # Conclusión
        conclusion = Text("La corriente atraviesa R_L siempre en la misma dirección.", font_size=22).to_edge(DOWN)
        self.play(Write(conclusion))
        self.wait(2)

        # Finalizar
        bye = Text("Fin de la animación del puente de diodos.", font_size=20).to_corner(UR)
        self.play(FadeIn(bye))
        self.wait(2)
        self.play(FadeOut(*self.mobjects))

if __name__ == '__main__':
    print("Renderízalo con manim:")
    print("manim -pql DiodeBridgeRhomboidScene.py DiodeBridgeRhomboidScene")