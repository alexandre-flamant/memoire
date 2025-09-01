from manim import *
from figures.manim_custom import *

config['background_color'] = WHITE

class BridgeUsage(MovingCameraScene):
    def __init__(self, **kargs):
        super().__init__(**kargs)
        self.camera.frame.scale(2)
        self.camera.frame.move_to(ORIGIN)

    def construct(self):
        nodes = np.array(
            [(0, 0, 0), (5, 0, 0), (10, 0, 0), (15, 0, 0), (20, 0, 0), (25, 0, 0), (30, 0, 0), (35, 0, 0), (40, 0, 0),
             (35, 5, 0), (30, 5, 0), (25, 5, 0), (20, 5, 0), (15, 5, 0), (10, 5, 0), (5, 5, 0)], dtype=float)
        supports = {0: (True, True), 8: (True, True)}
        connectivity_matrix = np.array([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
                                        (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 0),
                                        (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),
                                        (2, 15), (3, 14), (4, 13), (4, 11), (5, 10), (6, 9)])
        loads = {
            2: [0., -500e3],
            3: [0., -1000e3],
            4: [0., -500e3],
        }
        loads = {}

        A = [.1] * len(connectivity_matrix)
        E = [210.e9] * len(connectivity_matrix)


        font_size = 30
        g = Truss(nodes=nodes,
                  connectivity_matrix=connectivity_matrix,
                  supports=supports,
                  loads=loads,
                  member_style={'stroke_color': BLACK, 'stroke_width': 4},
                  node_style={'radius': .15, 'color': DARK_GRAY},
                  support_style={'height': 1.3, 'color': DARK_GRAY},
                  load_style={'scale': 4},
                  tip_style={'tip_length': .75, 'tip_width': .65},
                  deformed_style={'dash_length': .25, 'stroke_width': 8},
                  display_node_labels=False,
                  display_load_labels=False,
                  display_member_labels=False)

        g.move_to(ORIGIN)
        #g.overlap_deformation(100)
        #g.update()

        #for load in g.loads:
        #    self.bring_to_front(load)

        car = SVGMobject("figures/ref_img/car.svg")
        car.scale(2 / car.height)
        self.add( g, car)

        all_group = get_all_vmobjects(self)
        all_group.center().scale(.5)
        path = "test"
        all_group.to_svg(path, crop=True)

        car.next_to(g.members[0], UP, buff=0).shift((1 + car.width)*LEFT)
        car_move = car.animate.next_to(g.members[7].get_center()+(1+car.width)*RIGHT, UP, buff=0.)
        self.wait(1)
        self.play(car_move, run_time=3, rate_func=rate_functions.linear)
        self.wait(1)