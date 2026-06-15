import graphviz

from micrograd.engine import Node, Value


def create_node(node: "Node"):
    dot = graphviz.Digraph()
    dot.node(str(node.data()), node.label)
    return dot


def draw_node(node: "Node"):

    from IPython.display import Image

    return Image(create_node(node).pipe(format="png"))


if __name__ == "__main__":
    draw_node(Value(3))
