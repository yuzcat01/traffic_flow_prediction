import torch


def ensure_square_graph(graph: torch.Tensor) -> torch.Tensor:
    if graph.dim() != 2:
        raise ValueError(f"graph must be 2D, got shape {tuple(graph.shape)}")
    if graph.size(0) != graph.size(1):
        raise ValueError(f"graph must be square, got shape {tuple(graph.shape)}")
    return graph


def add_self_loops(graph: torch.Tensor) -> torch.Tensor:
    graph = ensure_square_graph(graph)
    eye = torch.eye(graph.size(0), dtype=graph.dtype, device=graph.device)
    return graph + eye


def symmetrize_graph(graph: torch.Tensor) -> torch.Tensor:
    graph = ensure_square_graph(graph)
    return torch.maximum(graph, graph.transpose(0, 1))


def build_binary_attention_mask(graph: torch.Tensor, symmetric: bool = True, add_self_loop: bool = True) -> torch.Tensor:
    graph = ensure_square_graph(graph)
    if symmetric:
        graph = symmetrize_graph(graph)
    if add_self_loop:
        graph = add_self_loops(graph)
    return graph > 0


def symmetric_normalize_adjacency(graph: torch.Tensor, add_self_loop: bool = True) -> torch.Tensor:
    graph = ensure_square_graph(graph)
    if add_self_loop:
        graph = add_self_loops(graph)

    degree = torch.sum(graph, dim=1).clamp_min(1e-8)
    degree_inv_sqrt = degree.pow(-0.5)
    return degree_inv_sqrt.unsqueeze(1) * graph * degree_inv_sqrt.unsqueeze(0)


def scaled_laplacian(graph: torch.Tensor, add_self_loop: bool = True) -> torch.Tensor:
    graph = ensure_square_graph(graph)
    identity = torch.eye(graph.size(0), dtype=graph.dtype, device=graph.device)
    normalized_adjacency = symmetric_normalize_adjacency(graph, add_self_loop=add_self_loop)
    normalized_laplacian = identity - normalized_adjacency
    return normalized_laplacian - identity
