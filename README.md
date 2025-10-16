# 🧠 netsci_toolkit

This package contains **core algorithms for network analysis**, implemented entirely from scratch and inspired by educational materials from the online textbook **"Network Science" by Brennan Klein** → [https://brennanklein.com/network-science-data-textbook](https://brennanklein.com/network-science-data-textbook).

It includes:
- Custom implementations of **centrality measures** (closeness, eigenvector)
- **Random graph generation** models (degree-preserving randomization, configuration model)
- **Community detection algorithms** (Girvan–Newman and Louvain)
- Helper functions for **modularity computation** and **hierarchical aggregation**

---

## ⚙️ Installation

Install directly from GitHub (recommended):
```bash
pip install "git+https://github.com/LucaSantagata/netsci_toolkit.git"
```

---

## 📚 Functions Overview

### 1. **Closeness Centrality** (`closeness_centrality_from_scratch`)

Computes closeness centrality for each node in the graph from scratch, without using NetworkX's built-in function.

**Formula:** For each node *i*, closeness centrality is defined as:
```
C(i) = (N-1) / Σ d(i,j)
```
where *d(i,j)* is the shortest path distance from *i* to *j*, summed over all reachable nodes *j*.

**Parameters:**
- `G` (networkx.Graph): Input graph
- `wf_improved` (bool, default=True): If True, applies Wasserman–Faust correction for disconnected graphs

**Example:**
```python
import networkx as nx
from netsci_toolkit.metrics import closeness_centrality_from_scratch

# Create a simple graph
G = nx.karate_club_graph()

# Compute closeness centrality
closeness = closeness_centrality_from_scratch(G, wf_improved=True)

# Display top 5 nodes by closeness
top_nodes = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 nodes by closeness centrality:")
for node, score in top_nodes:
    print(f"Node {node}: {score:.4f}")
```

---

### 2. **Eigenvector Centrality** (`eigenvector_centrality_from_scratch`)

Computes eigenvector centrality using the power iteration method. A node's centrality is proportional to the sum of the centralities of its neighbors.

**Parameters:**
- `G` (networkx.Graph): Input graph
- `max_iter` (int, default=100): Maximum number of iterations
- `tol` (float, default=1e-8): Convergence tolerance

**Example:**
```python
import networkx as nx
from netsci_toolkit.metrics import eigenvector_centrality_from_scratch

# Create a graph
G = nx.karate_club_graph()

# Compute eigenvector centrality
eigenvector = eigenvector_centrality_from_scratch(G, max_iter=100, tol=1e-8)

# Display results
print("Eigenvector centrality for first 5 nodes:")
for node in list(G.nodes())[:5]:
    print(f"Node {node}: {eigenvector[node]:.4f}")
```

---

### 3. **Degree-Preserving Randomization** (`degree_preserving_randomization`)

Randomizes the edges of a graph while preserving the degree sequence of all nodes. Uses edge-swap method.

**Parameters:**
- `G` (networkx.Graph): Input undirected graph
- `n_iter` (int, default=1000): Number of edge swap attempts
- `seed` (int, optional): Random seed for reproducibility

**Example:**
```python
import networkx as nx
from netsci_toolkit.metrics import degree_preserving_randomization

# Create original graph
G = nx.barabasi_albert_graph(100, 3, seed=42)

# Randomize while preserving degrees
G_random = degree_preserving_randomization(G, n_iter=1000, seed=42)

# Verify degree sequences match
print("Original degree sequence:", sorted(dict(G.degree()).values(), reverse=True)[:10])
print("Randomized degree sequence:", sorted(dict(G_random.degree()).values(), reverse=True)[:10])
print("Degrees preserved:", sorted(dict(G.degree()).values()) == sorted(dict(G_random.degree()).values()))
```

---

### 4. **Configuration Model** (`configuration_model_from_degree_sequence`)

Generates a random graph from a given degree sequence using the stub-matching method.

**Parameters:**
- `degree_sequence` (list): List of node degrees (must sum to an even number)
- `return_simple` (bool, default=True): If True, removes multi-edges and self-loops
- `seed` (int, optional): Random seed for reproducibility

**Example:**
```python
import networkx as nx
from netsci_toolkit.metrics import configuration_model_from_degree_sequence

# Extract degree sequence from existing graph
G_original = nx.barabasi_albert_graph(50, 3, seed=42)
degree_seq = [deg for node, deg in G_original.degree()]

# Generate new graph with same degree sequence
G_config = configuration_model_from_degree_sequence(degree_seq, return_simple=True, seed=42)

print(f"Original graph: {G_original.number_of_nodes()} nodes, {G_original.number_of_edges()} edges")
print(f"Config model: {G_config.number_of_nodes()} nodes, {G_config.number_of_edges()} edges")
print(f"Degree sequences match: {sorted(dict(G_original.degree()).values()) == sorted(dict(G_config.degree()).values())}")
```

---

### 5. **Girvan–Newman Community Detection** (`girvan_newman`)

Implements the Girvan–Newman algorithm for community detection by iteratively removing edges with highest betweenness and selecting the partition with maximum modularity.

**Parameters:**
- `G` (networkx.Graph): Input graph
- `output` (str, default="dict"): Output format
  - `"dict"`: Returns `{node: community_id}`
  - `"list"`: Returns list of community IDs in node order
  - `"lists"`: Returns list of lists, each containing nodes in a community

**Example:**
```python
import networkx as nx
from netsci_toolkit.metrics import girvan_newman

# Create graph with community structure
G = nx.karate_club_graph()

# Detect communities
communities_dict = girvan_newman(G, output="dict")
communities_lists = girvan_newman(G, output="lists")

print(f"Number of communities detected: {len(communities_lists)}")
print(f"Community sizes: {[len(c) for c in communities_lists]}")

# Visualize first community
print(f"Nodes in community 0: {communities_lists[0]}")
```

---

### 6. **Louvain Method** (`louvain_method`)

Implements the Louvain algorithm for community detection, which maximizes modularity through local optimization and hierarchical aggregation.

**Parameters:**
- `G` (networkx.Graph): Input graph
- `init` (dict, optional): Initial partition `{node: community_id}`. If None, each node starts in its own community
- `verbose` (bool, default=False): Print progress information

**Example:**
```python
import networkx as nx
from netsci_toolkit.metrics import louvain_method, get_modularity

# Create graph
G = nx.karate_club_graph()

# Detect communities
partition = louvain_method(G, verbose=True)

# Calculate modularity
communities_list = [[] for _ in range(max(partition.values()) + 1)]
for node, comm in partition.items():
    communities_list[comm].append(node)

Q = get_modularity(G, partition)

print(f"Number of communities: {len(communities_list)}")
print(f"Modularity: {Q:.4f}")
print(f"Community sizes: {[len(c) for c in communities_list]}")
```

---

### 7. **Helper Functions**

#### `get_modularity(G, partition)`

Computes the modularity of a given partition on graph G.

**Parameters:**
- `G` (networkx.Graph): Input graph
- `partition` (dict): Node-to-community mapping `{node: community_id}`

**Returns:** float (modularity value, range: -0.5 to 1.0)

**Example:**
```python
import networkx as nx
from netsci_toolkit.metrics import get_modularity

G = nx.karate_club_graph()
# Simple partition: split by node index
partition = {node: 0 if node < 17 else 1 for node in G.nodes()}

Q = get_modularity(G, partition)
print(f"Modularity: {Q:.4f}")
```

---

## 🔬 Complete Workflow Example

Here's a complete example combining multiple functions:
```python
import networkx as nx
from netsci_toolkit.metrics import (
    closeness_centrality_from_scratch,
    eigenvector_centrality_from_scratch,
    degree_preserving_randomization,
    louvain_method,
    get_modularity
)

# 1. Load a real network
G = nx.karate_club_graph()
print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# 2. Compute centrality measures
closeness = closeness_centrality_from_scratch(G)
eigenvector = eigenvector_centrality_from_scratch(G)

most_central_closeness = max(closeness, key=closeness.get)
most_central_eigenvector = max(eigenvector, key=eigenvector.get)

print(f"\nMost central node (closeness): {most_central_closeness}")
print(f"Most central node (eigenvector): {most_central_eigenvector}")

# 3. Detect communities
communities = louvain_method(G, verbose=False)
Q = get_modularity(G, communities)

print(f"\nCommunities detected: {max(communities.values()) + 1}")
print(f"Modularity: {Q:.4f}")

# 4. Create null model
G_random = degree_preserving_randomization(G, n_iter=1000, seed=42)
communities_random = louvain_method(G_random, verbose=False)
Q_random = get_modularity(G_random, communities_random)

print(f"\nNull model modularity: {Q_random:.4f}")
print(f"Difference from original: {Q - Q_random:.4f}")
```

---

## 📖 References

This toolkit is inspired by the **"Network Science"** textbook by Brennan Klein:
- 📘 Online textbook: [https://brennanklein.com/network-science-data-textbook](https://brennanklein.com/network-science-data-textbook)

---

## 📝 License

MIT License

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests on GitHub.

---

## 👨‍💻 Author

Luca Santagata - [GitHub](https://github.com/LucaSantagata)
