from manim import VGroup, VMobject
import xml.etree.ElementTree as ET


def get_all_vmobjects(o: VMobject):
    group = VGroup()
    for obj in o.mobjects:
        group.add(obj)
    return group


def set_svg_dimensions(svg_path, width="418pt", height=None, output_path=None):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    root.set("width", width)
    if height: root.set("height", height)

    if not output_path: output_path = svg_path
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
