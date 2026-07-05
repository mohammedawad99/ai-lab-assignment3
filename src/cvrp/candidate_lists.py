"""k-nearest-neighbor candidate lists for CVRP local search (Stage 11-A).

Large-neighborhood scans (relocate, insertion) look at every position in
every route; on M-n200-k17 that is most of the local-search time. Almost all
improving moves place a customer next to one of its geometrically nearest
nodes, so a precomputed k-nearest list per node lets a search consider only
those positions.

Candidate lists are OPT-IN: every consumer takes neighbors=None and then
behaves exactly like the full-neighborhood original. They only prune the
move set, so feasibility is never affected — a pruned search just improves
less in the worst case.
"""


def build_candidate_lists(distance_matrix, k=None) -> list[list[int]]:
    """k nearest other nodes for every node (depot included as node 0).

    Sorted by (distance, node id) so equal distances break ties on the
    smaller id — deterministic across runs. With k None or k >= n-1 the
    full sorted list is returned (exact-neighborhood fallback).
    """
    n = len(distance_matrix)
    lists = []
    for i in range(n):
        row = distance_matrix[i]
        others = sorted((j for j in range(n) if j != i),
                        key=lambda j: (row[j], j))
        lists.append(others if k is None or k >= n - 1 else others[:k])
    return lists


def neighbor_position_filter(neighbors, customer) -> set[int]:
    """Nodes a customer may be placed next to: its nearest list plus the
    depot, which every route starts and ends at."""
    allowed = set(neighbors[customer])
    allowed.add(0)
    return allowed
