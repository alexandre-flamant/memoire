from manim import *
from typing import List


class NeuralNetwork(VGroup):
    def __init__(self,
                 input_size,
                 n_hidden_layers,
                 n_neurons,
                 output_size,
                 max_displayed_neurons=6,
                 neuron_radius=0.15,
                 neuron_stoke_width=5,
                 layer_spacing=1.5,
                 neuron_spacing=0.4,
                 arrow=False,
                 connection_kwargs=None,
                 tip_kwargs=None,
                 labels=True,
                 font_size=20,
                 color_label=BLACK,
                 color_input_label=None,
                 color_hidden_label=None,
                 color_output_label=None,
                 color_neuron=BLUE_A,
                 color_input_neuron=None,
                 color_hidden_neuron=None,
                 color_output_neuron=None,
                 color_connection=BLACK,
                 color_ellipsis=BLACK,
                 stroke_width=4,
                 **kwargs):
        super().__init__(**kwargs)

        if not connection_kwargs: connection_kwargs = {}
        if not tip_kwargs: tip_kwargs = {}

        if not color_input_neuron: color_input_neuron = color_neuron
        if not color_hidden_neuron: color_hidden_neuron = color_neuron
        if not color_output_neuron: color_output_neuron = color_neuron

        if not color_input_label: color_input_label = color_label
        if not color_hidden_label: color_hidden_label = color_label
        if not color_output_label: color_output_label = color_label

        self.arrow = arrow
        self.connection_kwargs = connection_kwargs
        self.tip_kwargs = tip_kwargs
        self.input_size = input_size
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.output_size = output_size
        self.max_displayed_neurons = max_displayed_neurons

        self.input_layer = self._create_layer(input_size, max_displayed_neurons, neuron_radius,
                                              neuron_spacing, color_input_neuron, color_ellipsis,
                                              neuron_stoke_width)

        self.hidden_layers = [self._create_layer(n_neurons, max_displayed_neurons, neuron_radius,
                                                 neuron_spacing, color_hidden_neuron, color_ellipsis,
                                                 neuron_stoke_width)
                              .next_to(self.input_layer, RIGHT, buff=(i + 1) * layer_spacing + 2. * i * neuron_radius)
                              for i in range(n_hidden_layers)]

        self.output_layer = (self._create_layer(output_size, max_displayed_neurons, neuron_radius,
                                                neuron_spacing, color_output_neuron, color_ellipsis,
                                                neuron_stoke_width)
                             .next_to(self.hidden_layers[-1], RIGHT, buff=layer_spacing))

        # Connect neurons
        self.connection = self._connect_layers([self.input_layer] + self.hidden_layers + [self.output_layer],
                                               color_connection, stroke_width)

        self.labels = []
        if labels:
            self.labels.append(self._create_neurons_label(self.input_layer, lambda i: f"$x_{i + 1}$",
                                                          color=color_input_label, font_size=font_size))

            for j, layer in enumerate(self.hidden_layers):
                self.labels.append(
                    self._create_neurons_label(layer,
                                               lambda i: r"$a^{\left(" + str(j + 1) +
                                                         r"\right)}_{" + str(i + 1) + "}$",
                                               color=color_hidden_label, font_size=font_size))

            self.labels.append(self._create_neurons_label(self.output_layer, lambda i: f"$y_{i + 1}$",
                                                          color=color_output_label, font_size=font_size))

        self.add(*self.connection, self.input_layer, self.hidden_layers, self.output_layer, *self.labels)

    @staticmethod
    def _create_neurons_label(neurons, label_f, color, font_size):
        labels = []
        neurons = [n for n in neurons if isinstance(n, Circle)]
        for i, neuron in enumerate(neurons):
            label = Tex(label_f(i), color=color, font_size=font_size)
            label.move_to(neuron)
            labels.append(label)

        return VGroup(*labels)

    @staticmethod
    def _create_layer(size, max_displayed, radius, spacing, neuron_color, ellipsis_color, neuron_stroke_width):
        show_all = size <= max_displayed

        neurons = []
        if show_all:
            for i in range(size):
                neuron = Circle(radius=radius, color=neuron_color, stroke_width=neuron_stroke_width)
                neurons.append(neuron)
        else:
            top = max_displayed // 2
            bottom = max_displayed - top  # Leave one slot for ellipsis

            for i in range(top):
                neurons.append(Circle(radius=radius, color=neuron_color))

            neurons.append(MathTex("\\vdots").set_color(ellipsis_color).scale(radius * 2))

            for i in range(bottom):
                neurons.append(Circle(radius=radius, color=neuron_color))

        neurons = VGroup(*neurons)
        neurons.arrange(DOWN, buff=spacing)

        return neurons

    def _connect_layers(self, layers, connection_color, stroke_width) -> List[VGroup]:
        connections = []
        for l1, l2 in zip(layers[:-1], layers[1:]):
            connections_i = VGroup()
            l1_neurons = [n for n in l1 if isinstance(n, Circle)]
            l2_neurons = [n for n in l2 if isinstance(n, Circle)]

            for n1 in l1_neurons:
                for n2 in l2_neurons:
                    line = Line(n1.get_center(), n2.get_center(), buff=n2.radius,
                                stroke_color=connection_color, stroke_width=stroke_width,
                                **self.connection_kwargs)
                    if self.arrow: line.add_tip(**self.tip_kwargs)
                    connections_i.add(line)
            connections.append(connections_i)
        return connections

    @staticmethod
    def get_layer_bounding_box(layer, buff, stroke_color=BLACK, stroke_opacity=1,
                               fill_color=None, fill_opacity=.2, corner_radius=None):
        h = layer.height
        w = layer.width
        if not corner_radius: corner_radius = .5 * w + buff

        box = RoundedRectangle(height=h + 2. * buff, width=w + 2. * buff, corner_radius=corner_radius,
                               stroke_color=stroke_color, stroke_opacity=stroke_opacity,
                               fill_color=fill_color, fill_opacity=fill_opacity)

        box.move_to(layer)

        return box
