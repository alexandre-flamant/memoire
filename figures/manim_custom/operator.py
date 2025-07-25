from manim import *


class GeneralOperator(VGroup):
    def __init__(self, shape_class, shape_kwargs,
                 text_class, text_content, text_kwargs,
                 **kwargs):
        super().__init__(**kwargs)
        self.add(shape_class(**shape_kwargs))
        text = (text_class(text_content, **text_kwargs)
                .move_to(self))
        self.add(text)


class Operator(GeneralOperator):
    def __init__(self, h, w, text, text_class=Tex, stroke_width=2, stroke_color=BLACK, fill_color=None,
                 fill_opacity=1, font_size=25, font_color=BLACK):
        super().__init__(shape_class=Rectangle,
                         shape_kwargs={'height': h, 'width': w, 'stroke_width': stroke_width,
                                       'stroke_color': stroke_color, 'fill_color': fill_color,
                                       'fill_opacity': fill_opacity, },
                         text_class=text_class,
                         text_content=text,
                         text_kwargs={'font_size': font_size, 'color': font_color})
