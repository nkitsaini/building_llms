import graphviz

from micrograd.engine import Node


def create_diagraph(node: "Node", rankdir: str = "LR"):
    dot = graphviz.Digraph(graph_attr={"rankdir": rankdir})

    processed = set()

    def add_edge(tail: str, head: str):
        key = (tail, head)
        if key not in processed:
            dot.edge(tail, head)
            processed.add(key)

    def dfs(node: "Node", parent_id: str | None = None):
        node_id = str(id(node))
        if node.calc_grad:
            grad_value = f"{node.grad:.4f}"
        else:
            grad_value = "N/A"
        dot.node(
            node_id,
            f"{{{node.label} | data {node.data:.4f} | grad {grad_value} }}",
            shape="record",
        )
        if parent_id:
            add_edge(node_id, parent_id)
        if not node.op.is_noop():
            op_id = node_id + "_" + node.op.render()
            dot.node(op_id, str(node.op.render()))
            add_edge(op_id, node_id)
            for child in node.children:
                dfs(child, op_id)

    dfs(node)

    return dot


def draw_node(node: "Node", rankdir: str = "LR"):
    from IPython.display import Image

    return Image(create_diagraph(node, rankdir=rankdir).pipe(format="png"))
