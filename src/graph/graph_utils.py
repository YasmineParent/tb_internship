import networkx as nx
import pyagrum as gum


def topic_graph_to_bn(topic_graph: nx.DiGraph, node_names: list) -> gum.BayesNet:
    """Convert a TopologicalCausalMixture topic_graph to a pyAgrum BayesNet for visualization."""
    bn = gum.BayesNet()
    for i in topic_graph.nodes:
        bn.add(gum.LabelizedVariable(str(node_names[i]), str(node_names[i]), 2))
    for i, j in topic_graph.edges:
        bn.addArc(str(node_names[i]), str(node_names[j]))
    return bn
