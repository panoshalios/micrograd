from micrograd.value import Value
from graphviz import Digraph

def get_graph(root: Value):
    nodes = set()
    edges = set()

    def build(value: Value):
        if value in nodes:
            return

        nodes.add(value)

        for child in value.children:
            edges.add((child,value))
            build(child)

    build(root)
    return nodes, edges

def draw_dot(root: Value):
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})
    nodes, edges = get_graph(root)

    for node in nodes:
        uid = str(id(node))
        dot.node(
            uid,
            label="{ %s | data %.4f | grad %.4f }" % (node.label, node.data, node.gradient),
            shape="record",
        )
        
        if node.operation:
            op_id = uid + node.operation
            dot.node(op_id, label=node.operation)
            dot.edge(op_id, uid)
    
    for child, parent in edges:
        target = str(id(parent)) + parent.operation if parent.operation else str(id(parent))
        dot.edge(str(id(child)), target)

    return dot