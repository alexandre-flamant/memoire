from manim import *


class RollingSupport(VMobject):
    def __init__(self,
                 location=(.0, .0, .0),
                 height=1.,
                 direction=UP,
                 color=BLACK,
                 stroke_width=DEFAULT_STROKE_WIDTH,
                 **kwargs):
        super().__init__(**kwargs)
        self.location = location
        self.direction = direction
        self._support = Circle(stroke_width=stroke_width).move_to(location)
        self._support = (self._support
                         .scale(height / self._support.height)
                         .align_to(location, direction)
                         .set_stroke(color))
        self._ground = (Line((0., 0., 0.), 1.5 * self._support.width * direction)
                        .rotate(0.5 * PI)
                        .move_to(self._support.get_center())
                        .align_to(self._support, -direction)
                        .set_stroke(color))
        self.add(self._support, self._ground)

    def get_ref_point(self):
        return self.get_boundary_point(self.direction)


class SimpleSupport(VMobject):
    def __init__(self,
                 location=(.0, .0, .0),
                 height=1.,
                 direction=UP,
                 color=BLACK,
                 stroke_width=DEFAULT_STROKE_WIDTH,
                 **kwargs):
        super().__init__(**kwargs)
        self.location = location
        self.direction = direction
        self._support = Triangle(stroke_width=stroke_width).rotate(angle_of_vector(direction) - PI / 2).move_to(
            location)
        self._support = (self._support
                         .scale(height / self._support.height)
                         .align_to(location, direction)
                         .set_stroke(color))
        self._ground = (Line((0., 0., 0.), 1.5 * self._support.width * direction)
                        .rotate(0.5 * PI)
                        .move_to(self._support.get_center())
                        .align_to(self._support, -direction)
                        .set_stroke(color))
        self.add(self._support, self._ground)

    def get_ref_point(self):
        return self.get_boundary_point(self.direction)


__all__ = ["RollingSupport", "SimpleSupport"]
