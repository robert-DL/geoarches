import networkx
import torch


def create_2d_mesh_edges(nx: int, ny: int, periodic: bool = True) -> torch.tensor:
    """
    Create a 2D graph edge index with given number of nodes in X and Y direction.

    If periodic is true, creates a periodic connection in Y dimension
    """
    G = networkx.grid_2d_graph(nx, ny, periodic=[periodic, False])

    G = G.to_directed() if not networkx.is_directed(G) else G

    mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    edge_index = torch.empty((2, G.number_of_edges()), dtype=torch.long)
    for i, (src, dst) in enumerate(G.edges()):
        edge_index[0, i] = mapping[src]
        edge_index[1, i] = mapping[dst]

    return edge_index
