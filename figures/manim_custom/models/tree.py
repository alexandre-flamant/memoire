from manim import *


class BinaryTree(VGroup):
    def __init__(self,
                 depth=3,
                 node_radius=.25,
                 node_color=GRAY,
                 node_stroke_color=BLACK,
                 node_stroke_width=4,
                 connection_color=BLACK,
                 connection_stroke_width=4,
                 h_spacing=.5,
                 v_spacing=.5,
                 **kwargs):
        super().__init__(**kwargs)

        self._node_radius = node_radius
        self._node_color = node_color
        self._node_stroke_color = node_stroke_color
        self._node_stroke_width = node_stroke_width
        self._connection_color = connection_color
        self._connection_stroke_width = connection_stroke_width

        if depth <= 0:
            self.add(VGroup(self.create_node(), name='layer_0'))
            return

        last_layer = VGroup(*[self.create_node() for _ in range(2 ** depth)],
                            name=f'layer_{depth}')
        last_layer.arrange(buff=h_spacing)

        self.layers = [last_layer]
        self.connections = []
        for i in range(depth - 1, -1, -1):
            prev_layer = self.layers[-1]
            layer = VGroup(name=f'layer_{i}')
            connection = VGroup(name=f'connection_layer_{i}')
            for j in range(len(prev_layer) // 2):
                parents = prev_layer[2 * j:2 * j + 2]
                node = self.create_node().next_to(parents, UP, v_spacing)
                connection.add(
                    self.create_connection(node, parents[0]),
                    self.create_connection(node, parents[1])
                )
                layer.add(node)

            self.layers.append(layer)
            self.connections.append(connection)

        self.layers = self.layers[::-1]
        self.connections = self.connections[::-1]
        self.add(*self.layers, *self.connections)

    def create_node(self):
        return Dot(radius=self._node_radius, fill_color=self._node_color,
                   stroke_color=self._node_stroke_color, stroke_width=self._node_stroke_width)

    def create_connection(self, node1: Dot, node2: Dot):
        return Line(node1.get_center(), node2.get_center(), buff=self._node_radius,
                    stroke_width=self._connection_stroke_width, stroke_color=self._connection_color)
