from manim import *
from typing import List


class DataTable(VGroup):
    def __init__(self, n_rows=5, n_cols=3, width=1.5, height=2, corner_radius=0.3,
                 stroke_width=2, fill_color=GRAY_A, stroke_color=BLACK, data_fn=None, data_fontsize=30,
                 fill_opacity=1,**kwargs):
        super().__init__(**kwargs)

        self.corner_radius = corner_radius
        self.height = height
        self.width = width
        self.n_rows = n_rows
        self.n_cols = n_cols

        # Background Rounded Rectangle
        self.bg = RoundedRectangle(
            width=width,
            height=height,
            corner_radius=corner_radius,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
        )

        self.row_group = VGroup(name='rows')
        row_height = height / n_rows
        for i in range(1, n_rows):
            line = (Line(self.bg.get_left(), self.bg.get_right(),
                         stroke_color=stroke_color,
                         stroke_width=stroke_width)
                    .align_to(self.bg, UP)
                    .shift(i * row_height * DOWN))
            self.row_group.add(line)

        self.col_group = VGroup(name='cols')
        col_width = width / n_cols
        for i in range(1, n_cols):
            line = (Line(self.bg.get_top(), self.bg.get_bottom(),
                         stroke_color=stroke_color,
                         stroke_width=stroke_width)
                    .align_to(self.bg, LEFT)
                    .shift(i * col_width * RIGHT))
            self.col_group.add(line)

        self.col_fill_group = VGroup(name='col_color')
        for i in range(n_cols):
            if n_cols == 1:
                arc1 = Arc(radius=self.corner_radius, start_angle=0, angle=0.5 * PI)
                arc2 = (Arc(radius=self.corner_radius, start_angle=0.5 * PI, angle=0.5 * PI)
                        .shift((col_width - 2 * self.corner_radius) * LEFT))
                points = [arc1.point_from_proportion(x) for x in np.linspace(0, 1, 100)] + \
                         [arc2.point_from_proportion(x) for x in np.linspace(0, 1, 100)]
                points.append(points[-1] + (row_height - self.corner_radius) * DOWN)
                points.append(points[-1] + col_width * RIGHT)
                fill = Polygon(*points, fill_opacity=fill_opacity, fill_color=fill_color,
                            stroke_width=0, stroke_color=fill_color)
            elif i == 0:
                arc = Arc(radius=self.corner_radius, start_angle=0.5 * PI, angle=0.5 * PI)
                points = [arc.point_from_proportion(x) for x in np.linspace(0, 1, 100)]
                points.append(points[-1] + (row_height - self.corner_radius) * DOWN)
                points.append(points[-1] + col_width * RIGHT)
                points.append(points[-1] + row_height * UP)
                fill = Polygon(*points, fill_opacity=fill_opacity, fill_color=fill_color,
                               stroke_width=0, stroke_color=fill_color)
            elif i == n_cols - 1:
                arc = Arc(radius=self.corner_radius, start_angle=0, angle=0.5 * PI)
                points = [arc.point_from_proportion(x) for x in np.linspace(0, 1, 100)]
                points.append(points[-1] + (col_width - self.corner_radius) * LEFT)
                points.append(points[-1] + row_height * DOWN)
                points.append(points[-1] + col_width * RIGHT)
                fill = Polygon(*points, fill_opacity=fill_opacity, fill_color=fill_color,
                               stroke_width=0, stroke_color=fill_color)
            else:
                fill = Rectangle(height=row_height, width=col_width, fill_color=fill_color,
                                 fill_opacity=fill_opacity, stroke_width=0.1, stroke_color=fill_color)

            fill = fill.align_to(self.bg, UP).align_to(self.bg, LEFT).shift(i * width / n_cols * RIGHT)
            self.col_fill_group.add(fill)

        self.data_group = VGroup(name='data')
        if data_fn is None: data_fn = lambda: f"${np.random.randint(0, 10)}$"
        for row_i in range(1, n_rows):
            for col_i in range(n_cols):
                text = (Tex(f"{data_fn()}", color=BLACK, font_size=data_fontsize)
                        .align_to(self.bg, UP)
                        .align_to(self.bg, LEFT)
                        .shift(row_i * row_height * DOWN)
                        .shift(col_i * col_width * RIGHT))
                (text
                 .shift(0.5 * (col_width - text.width) * RIGHT)
                 .shift(0.5 * (row_height - text.height) * DOWN))
                self.data_group.add(text)

        self.add(self.col_fill_group, self.row_group, self.col_group, self.bg, self.data_group)

    def collapse_col(self, idx: int | List[int]):
        if isinstance(idx, int): idx = [idx]
        for i in idx:
            col = self.col_group[i]
            col.put_start_and_end_on(col.get_start() + (self.height / self.n_rows) * DOWN,
                                     col.get_end())
        return self

    def collapse_cols(self, n=-2):
        if n < 0: n = self.n_cols + n

        for i in range(len(self.col_group)):
            if i >= n: break
            self.collapse_col(i)

        for i in range(n + 1, self.n_cols - 1):
            self.collapse_col(i)
        return self
