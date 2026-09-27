import torch

from acd_benchmark.cyber.observation_schema import NUM_NODES, NUM_SUBNET_NODES


def batched_edge_index(membership: torch.Tensor, structural: torch.Tensor) -> torch.Tensor:
    """Subnet links plus bidirectional host membership, without cross-batch edges."""
    structural_edges = (structural > 0).nonzero(as_tuple=False)
    membership_edges = (membership > 0).nonzero(as_tuple=False)
    sources = []
    destinations = []
    if structural_edges.numel() > 0:
        offsets = structural_edges[:, 0] * NUM_NODES
        sources.append(offsets + structural_edges[:, 1])
        destinations.append(offsets + structural_edges[:, 2])
    if membership_edges.numel() > 0:
        offsets = membership_edges[:, 0] * NUM_NODES
        host_nodes = offsets + NUM_SUBNET_NODES + membership_edges[:, 1]
        subnet_nodes = offsets + membership_edges[:, 2]
        sources.extend((host_nodes, subnet_nodes))
        destinations.extend((subnet_nodes, host_nodes))
    if not sources:
        return torch.empty((2, 0), dtype=torch.long, device=membership.device)
    return torch.stack((torch.cat(sources), torch.cat(destinations)), dim=0).long()
