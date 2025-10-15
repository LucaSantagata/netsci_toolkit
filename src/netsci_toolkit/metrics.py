# metrics.py — core algorithms without visualization

import numpy as np
import networkx as nx
from collections import defaultdict


# ---------- 1) CLOSENESS (from scratch) ----------
def closeness_centrality_from_scratch(G):
    """
    Closeness centrality computed from scratch (no nx.closeness_centrality).
    For each node i: C(i) = (N-1) / sum_j d(i, j) over reachable j.
    Isolated nodes get 0.0.
    """
    centrality = {}
    N = G.number_of_nodes()
    for node_i in G.nodes():
        # Shortest-path tree from node_i
        lengths = nx.shortest_path_length(G, source=node_i)
        total_dist = 0
        count = 0
        for _, d in lengths:
            if d > 0:
                total_dist += d
                count += 1
        if total_dist > 0 and N > 1 and count > 0:
            # standard normalization (ignore unreachable)
            centrality[node_i] = (count) / total_dist
        else:
            centrality[node_i] = 0.0
    return centrality


# ---------- 2) EIGENVECTOR (from scratch, power iteration) ----------
def eigenvector_centrality_from_scratch(G, max_iter=100, tol=1e-8):
    """
    Power-iteration implementation of eigenvector centrality on an undirected graph.
    Normalizes by Euclidean norm each iteration; stops when max change < tol.
    """
    if len(G) == 0:
        return {}

    # Use undirected view for symmetry
    H = G.to_undirected()

    x = {u: 1.0 / len(H) for u in H}
    for _ in range(max_iter):
        x_prev = x
        # multiply by adjacency
        x = {u: 0.0 for u in H}
        for u, v in H.edges():
            x[u] += x_prev[v]
            x[v] += x_prev[u]
        # normalize
        norm = np.sqrt(sum(v * v for v in x.values()))
        if norm == 0:
            return x
        x = {u: v / norm for u, v in x.items()}
        # convergence
        diff = max(abs(x[u] - x_prev.get(u, 0.0)) for u in H)
        if diff < tol:
            break
    return x


# ---------- 3) DEGREE-PRESERVING RANDOMIZATION (edge swaps) ----------
def degree_preserving_randomization(G, n_iter=1000, seed=None):
    """
    Randomize edges by performing degree-preserving swaps. Works on simple graphs.
    """
    rng = np.random.default_rng(seed)
    H = G.copy()
    if H.is_directed():
        raise ValueError("This simple swapper assumes an undirected simple graph.")
    edges = list(H.edges())
    for _ in range(n_iter):
        if len(edges) < 2:
            break
        (u, v) = edges[rng.integers(0, len(edges))]
        (x, y) = edges[rng.integers(0, len(edges))]
        # distinct endpoints -> avoid self-loops
        if len({u, v, x, y}) != 4:
            continue
        # choose one rewiring pattern randomly
        if rng.random() > 0.5:
            a, b, c, d = u, y, x, v
        else:
            a, b, c, d = u, x, v, y
        if not (H.has_edge(a, b) or H.has_edge(c, d)):
            H.remove_edge(u, v)
            H.remove_edge(x, y)
            H.add_edge(a, b)
            H.add_edge(c, d)
            edges = list(H.edges())
    return H


# ---------- 4) CONFIGURATION MODEL (from degree sequence) ----------
def configuration_model_from_degree_sequence(degree_sequence, return_simple=True, seed=None):
    """
    Build a (multi)graph from a degree sequence by random stub matching.
    If return_simple=True, converts to a simple Graph removing multiedges/self-loops.
    """
    if sum(degree_sequence) % 2 != 0:
        raise ValueError("The sum of the degree sequence must be even.")

    rng = np.random.default_rng(seed)
    stubs = []
    for node, deg in enumerate(degree_sequence):
        if deg < 0:
            raise ValueError("Degrees must be non-negative.")
        stubs.extend([node] * deg)

    rng.shuffle(stubs)

    M = nx.MultiGraph()
    M.add_nodes_from(range(len(degree_sequence)))
    while stubs:
        u = stubs.pop()
        v = stubs.pop()
        M.add_edge(u, v)

    if return_simple:
        return nx.Graph(M)  # removes multiedges/self-loops by construction
    return M


# ---------- 5) GIRVAN–NEWMAN (greedy edge removal, best modularity) ----------
def girvan_newman(G, output="dict"):
    """
    Greedy Girvan–Newman community detection: remove max edge-betweenness edges,
    track partition with best modularity; return best partition.
    """
    if G.is_directed():
        H = G.to_undirected().copy()
    else:
        H = G.copy()

    best_part = None
    best_Q = -1.0

    # work on a changing copy
    W = H.copy()
    while W.number_of_edges() > 0:
        bc = nx.edge_betweenness_centrality(W)
        e_star = max(bc, key=bc.get)
        W.remove_edge(*e_star)

        comps = list(nx.connected_components(W))
        # compute modularity on ORIGINAL H
        Q = nx.algorithms.community.modularity(H, comps)
        # store best
        if Q > best_Q:
            best_Q = Q
            best_part = {}
            for i, comp in enumerate(comps):
                for u in comp:
                    best_part[u] = i

    # ensure we return labels for all original nodes
    if best_part is None:
        best_part = {u: 0 for u in H.nodes()}
    # shape
    if output == "dict":
        return best_part
    if output == "list":
        return [best_part[u] for u in H.nodes()]
    if output == "lists":
        k = sorted(set(best_part.values()))
        out = [[] for _ in k]
        for u, c in best_part.items():
            out[c].append(u)
        return out
    raise ValueError("output must be one of {'dict','list','lists'}")


# ---------- Louvain helpers ----------
def get_modularity(G, partition):
    """
    Modularity of a partition on graph G. If edges have no 'weight', assume weight=1.
    partition: dict node -> community_id
    """
    H = G.copy()
    # ensure weights
    if H.number_of_edges() > 0:
        if "weight" not in next(iter(H.edges(data=True)))[-1]:
            nx.set_edge_attributes(H, 1.0, "weight")

    E_c = defaultdict(float)  # internal community weight
    k_c = defaultdict(float)  # sum of degrees per community
    M = 0.0                   # total edge weight (sum w_ij)

    for u, v, w in H.edges(data=True):
        M += w["weight"]
        if partition[u] == partition[v]:
            E_c[partition[u]] += w["weight"]

    deg = H.degree(weight="weight")
    for u in H.nodes():
        k_c[partition[u]] += deg[u]

    if M == 0:
        return 0.0
    return sum((E_c[c] / M) - (k_c[c] / (2.0 * M)) ** 2 for c in k_c)


def local_optimization_step(G, Qmax, verbose=False):
    """
    One Louvain local-moving phase: try moving each node to neighbor communities.
    Updates node attribute 'community' in-place. Returns new Qmax.
    """
    nodes = list(G.nodes())
    rng = np.random.default_rng()
    rng.shuffle(nodes)

    changed = True
    it = 0
    while changed:
        changed = False
        it += 1
        if verbose:
            print(f"[Louvain] local-move iter {it}")

        for u in nodes:
            part = nx.get_node_attributes(G, "community")
            cur = part[u]
            best = cur
            best_dQ = 0.0

            # try neighbors' communities
            for v in G.neighbors(u):
                c = G.nodes[v]["community"]
                if c == cur:
                    continue
                part[u] = c
                dQ = get_modularity(G, part) - Qmax
                if dQ > best_dQ:
                    best_dQ = dQ
                    best = c
            if best != cur:
                G.nodes[u]["community"] = best
                Qmax += best_dQ
                changed = True
        rng.shuffle(nodes)
    return Qmax


def network_aggregation_step(G):
    """
    Collapse communities into super-nodes and aggregate edge weights between them.
    Returns a new weighted graph where each node is a community.
    """
    edges = defaultdict(float)
    for u, v, w in G.edges(data=True):
        c1 = G.nodes[u]["community"]
        c2 = G.nodes[v]["community"]
        a, b = sorted((c1, c2))
        edges[(a, b)] += w.get("weight", 1.0)

    H = nx.Graph()
    for (a, b), w in edges.items():
        H.add_edge(a, b, weight=w)
    for c in H.nodes():
        H.nodes[c]["community"] = c
    return H


def reindex_communities(partition):
    """
    Reindex community labels to 0..K-1 (stable order).
    """
    mapping = {c: i for i, c in enumerate(sorted(set(partition.values())))}
    return {u: mapping[c] for u, c in partition.items()}


# ---------- 10) LOUVAIN ----------
def louvain_method(G, init=None, verbose=False):
    """
    Minimal Louvain implementation:
    - local moving to improve modularity
    - network aggregation
    - repeat until no change in number of communities
    Returns: dict node -> community_id
    """
    # ensure undirected weighted graph
    H = G.to_undirected().copy()
    if H.number_of_edges() > 0:
        if "weight" not in next(iter(H.edges(data=True)))[-1]:
            nx.set_edge_attributes(H, 1.0, "weight")

    # initial communities
    if init is None:
        part = {u: i for i, u in enumerate(H.nodes())}
    else:
        part = init.copy()
    nx.set_node_attributes(H, part, "community")

    # initial modularity
    Qmax = get_modularity(H, part)

    N_prev = -1
    while True:
        # local optimization
        Qmax = local_optimization_step(H, Qmax, verbose=verbose)
        part = nx.get_node_attributes(H, "community")

        # aggregate
        H = network_aggregation_step(H)

        # number of communities
        N = H.number_of_nodes()
        if N == N_prev:
            break
        N_prev = N

    # map back: after final agg, node labels == community ids
    final_part = nx.get_node_attributes(H, "community")
    # make labels compact
    return reindex_communities(final_part)
