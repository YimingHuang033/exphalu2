"""SeSE baseline (Semantic Structural Entropy; UAI 2026, arXiv:2511.16275).

Short-form pipeline adapted from the official repository
(https://github.com/SELGroup/SeSE, commit 8d4c6c5,
`sentence_structural_entropy/src/uncertainty_measures/structural_entropy.py`
and `construct_semantic_graph.py`). The coding-tree structural entropy math
(`PartitionTree`, `compute_se`) is ported as-is; only the numba-jitted
`cut_volume` is replaced by an equivalent numpy reduction and visualization
helpers are dropped.

Deviations from the official protocol (recorded, never silent):
- Answer enhancement uses GPT-4o in the official code; no API key is available
  on this machine, so responses are used verbatim and the run record carries
  `answer_enhancement: "none:blocked-no-api"`.
- Requires an NLI model (deberta-v2-xlarge-mnli) and a sentence-embedding
  model; if either is missing the method returns an explicit invalid status.

Direction contract: larger structural entropy = more semantic uncertainty =
higher hallucination risk (official usage; SeSE generalizes Semantic Entropy).
"""
from __future__ import annotations

import copy
import heapq
import itertools
import math
from typing import Optional

import numpy as np

from ..backends.base import BackendError

VERSION = "sese-v1-official-port-8d4c6c5"


# ---------------------------------------------------------------------------
# Partition tree: ported from the official repository (see module docstring).
# ---------------------------------------------------------------------------
def _cut_volume(adj_matrix: np.ndarray, p1, p2) -> float:
    if len(p1) == 1 and len(p2) == 1:
        return float(adj_matrix[p1[0], p2[0]])
    return float(adj_matrix[np.ix_(p1, p2)].sum())


class _PartitionTreeNode:
    __slots__ = ("ID", "partition", "parent", "children", "vol", "g",
                 "merged", "child_h", "child_cut")

    def __init__(self, ID, partition, vol, g, children=None, parent=None,
                 child_h=0, child_cut=0):
        self.ID = ID
        self.partition = partition
        self.parent = parent
        self.children = children
        self.vol = vol
        self.g = g
        self.merged = False
        self.child_h = child_h
        self.child_cut = child_cut


def _graph_parse(adj_matrix: np.ndarray):
    n = adj_matrix.shape[0]
    adj_table = {}
    vol_total = 0.0
    node_vol = []
    for i in range(n):
        n_v = 0.0
        adj = set()
        for j in range(n):
            if adj_matrix[i, j] != 0:
                n_v += adj_matrix[i, j]
                vol_total += adj_matrix[i, j]
                adj.add(j)
        adj_table[i] = adj
        node_vol.append(n_v)
    return n, vol_total, node_vol, adj_table


class _PartitionTree:
    def __init__(self, adj_matrix: np.ndarray):
        self.adj_matrix = np.asarray(adj_matrix, dtype=np.float64)
        self.tree_node = {}
        self.g_num_nodes, self.VOL, self.node_vol, self.adj_table = _graph_parse(
            self.adj_matrix)
        if self.VOL <= 0:
            raise BackendError("sese: semantic graph has zero total volume")
        self._id = 0
        self.leaves = []
        self.root_id = None
        for vertex in range(self.g_num_nodes):
            v = self.node_vol[vertex]
            self.tree_node[self._id] = _PartitionTreeNode(
                ID=self._id, partition=[vertex], g=v, vol=v)
            self.leaves.append(self._id)
            self._id += 1

    def _next_id(self):
        i = self._id
        self._id += 1
        return i

    @staticmethod
    def _merge(new_id, id1, id2, cut_v, node_dict):
        new_partition = node_dict[id1].partition + node_dict[id2].partition
        v = node_dict[id1].vol + node_dict[id2].vol
        g = node_dict[id1].g + node_dict[id2].g - 2 * cut_v
        child_h = max(node_dict[id1].child_h, node_dict[id2].child_h) + 1
        node_dict[new_id] = _PartitionTreeNode(
            ID=new_id, partition=new_partition, children={id1, id2},
            g=g, vol=v, child_h=child_h, child_cut=cut_v)
        node_dict[id1].parent = new_id
        node_dict[id2].parent = new_id

    @staticmethod
    def _compress_node(node_dict, node_id, parent_id):
        p_child_h = node_dict[parent_id].child_h
        node_children = node_dict[node_id].children
        node_dict[parent_id].child_cut += node_dict[node_id].child_cut
        node_dict[parent_id].children.remove(node_id)
        node_dict[parent_id].children = node_dict[parent_id].children.union(node_children)
        for c in node_children:
            node_dict[c].parent = parent_id
        com_node_child_h = node_dict[node_id].child_h
        node_dict.pop(node_id)
        if (p_child_h - com_node_child_h) == 1:
            while True:
                max_child_h = max(node_dict[f_c].child_h
                                  for f_c in node_dict[parent_id].children)
                if node_dict[parent_id].child_h == (max_child_h + 1):
                    break
                node_dict[parent_id].child_h = max_child_h + 1
                parent_id = node_dict[parent_id].parent
                if parent_id is None:
                    break

    @staticmethod
    def _child_tree_depth(node_dict, nid):
        node = node_dict[nid]
        depth = 0
        while node.parent is not None:
            node = node_dict[node.parent]
            depth += 1
        return depth + node_dict[nid].child_h

    @staticmethod
    def _compress_delta(node1, p_node):
        return node1.child_cut * math.log(p_node.vol / node1.vol)

    @staticmethod
    def _combine_delta(node1, node2, cut_v, g_vol):
        v1, v2 = node1.vol, node2.vol
        g1, g2 = node1.g, node2.g
        v12 = v1 + v2
        return ((v1 - g1) * math.log(v12 / v1, 2)
                + (v2 - g2) * math.log(v12 / v2, 2)
                - 2 * cut_v * math.log(g_vol / v12, 2)) / g_vol

    def _build_k_tree(self, g_vol, nodes_dict, k=None):
        min_heap, cmp_heap = [], []
        nodes_ids = list(nodes_dict.keys())
        new_id = None
        for i in nodes_ids:
            for j in self.adj_table[i]:
                if j > i:
                    n1, n2 = nodes_dict[i], nodes_dict[j]
                    cut_v = _cut_volume(self.adj_matrix, n1.partition, n2.partition)
                    diff = self._combine_delta(n1, n2, cut_v, g_vol)
                    heapq.heappush(min_heap, (diff, i, j, cut_v))
        unmerged_count = len(nodes_ids)
        while unmerged_count > 1:
            if len(min_heap) == 0:
                break
            diff, id1, id2, cut_v = heapq.heappop(min_heap)
            if nodes_dict[id1].merged or nodes_dict[id2].merged:
                continue
            nodes_dict[id1].merged = True
            nodes_dict[id2].merged = True
            new_id = self._next_id()
            self._merge(new_id, id1, id2, cut_v, nodes_dict)
            self.adj_table[new_id] = self.adj_table[id1].union(self.adj_table[id2])
            for i in self.adj_table[new_id]:
                self.adj_table[i].add(new_id)
            if nodes_dict[id1].child_h > 0:
                heapq.heappush(cmp_heap,
                               [self._compress_delta(nodes_dict[id1], nodes_dict[new_id]),
                                id1, new_id])
            if nodes_dict[id2].child_h > 0:
                heapq.heappush(cmp_heap,
                               [self._compress_delta(nodes_dict[id2], nodes_dict[new_id]),
                                id2, new_id])
            unmerged_count -= 1
            for ID in list(self.adj_table[new_id]):
                if not nodes_dict[ID].merged:
                    n1, n2 = nodes_dict[ID], nodes_dict[new_id]
                    cut_v = _cut_volume(self.adj_matrix, n1.partition, n2.partition)
                    new_diff = self._combine_delta(n1, n2, cut_v, g_vol)
                    heapq.heappush(min_heap, (new_diff, ID, new_id, cut_v))
        root = new_id
        if unmerged_count > 1:
            assert len(min_heap) == 0
            unmerged_nodes = {i for i, j in nodes_dict.items() if not j.merged}
            new_child_h = max(nodes_dict[i].child_h for i in unmerged_nodes) + 1
            new_id = self._next_id()
            nodes_dict[new_id] = _PartitionTreeNode(
                ID=new_id, partition=list(nodes_ids), children=unmerged_nodes,
                vol=g_vol, g=0, child_h=new_child_h)
            for i in unmerged_nodes:
                nodes_dict[i].merged = True
                nodes_dict[i].parent = new_id
                if nodes_dict[i].child_h > 0:
                    heapq.heappush(cmp_heap,
                                   [self._compress_delta(nodes_dict[i], nodes_dict[new_id]),
                                    i, new_id])
            root = new_id
        if k is not None:
            while nodes_dict[root].child_h > k:
                diff, node_id, p_id = heapq.heappop(cmp_heap)
                if self._child_tree_depth(nodes_dict, node_id) <= k:
                    continue
                children = nodes_dict[node_id].children
                self._compress_node(nodes_dict, node_id, p_id)
                if nodes_dict[root].child_h == k:
                    break
                for e in cmp_heap:
                    if e[1] == p_id:
                        if self._child_tree_depth(nodes_dict, p_id) > k:
                            e[0] = self._compress_delta(nodes_dict[e[1]], nodes_dict[e[2]])
                    if e[1] in children:
                        if nodes_dict[e[1]].child_h == 0:
                            continue
                        if self._child_tree_depth(nodes_dict, e[1]) > k:
                            e[2] = p_id
                            e[0] = self._compress_delta(nodes_dict[e[1]], nodes_dict[p_id])
                heapq.heapify(cmp_heap)
        return root

    def _check_balance(self, node_dict, root_id):
        root_c = copy.deepcopy(node_dict[root_id].children)
        for c in root_c:
            if node_dict[c].child_h == 0:
                self._single_up(node_dict, c)

    def _single_up(self, node_dict, node_id):
        new_id = self._next_id()
        p_id = node_dict[node_id].parent
        grow_node = _PartitionTreeNode(
            ID=new_id, partition=node_dict[node_id].partition, parent=p_id,
            children={node_id}, vol=node_dict[node_id].vol, g=node_dict[node_id].g)
        node_dict[node_id].parent = new_id
        node_dict[p_id].children.remove(node_id)
        node_dict[p_id].children.add(new_id)
        node_dict[new_id] = grow_node
        node_dict[new_id].child_h = node_dict[node_id].child_h + 1
        self.adj_table[new_id] = self.adj_table[node_id]
        for i in self.adj_table[node_id]:
            self.adj_table[i].add(new_id)

    def _build_sub_leaves(self, node_list, p_vol):
        subgraph_node_dict = {}
        ori_ent = 0.0
        for vertex in node_list:
            ori_ent += -(self.tree_node[vertex].g / self.VOL) * math.log2(
                self.tree_node[vertex].vol / p_vol)
            sub_n = set()
            vol = 0.0
            for vertex_n in node_list:
                c = self.adj_matrix[vertex, vertex_n]
                if c != 0:
                    vol += c
                    sub_n.add(vertex_n)
            subgraph_node_dict[vertex] = _PartitionTreeNode(
                ID=vertex, partition=[vertex], g=vol, vol=vol)
            self.adj_table[vertex] = sub_n
        return subgraph_node_dict, ori_ent

    def _build_root_down(self):
        root_child = self.tree_node[self.root_id].children
        subgraph_node_dict = {}
        ori_en = 0.0
        g_vol = self.tree_node[self.root_id].vol
        for node_id in root_child:
            node = self.tree_node[node_id]
            ori_en += -(node.g / g_vol) * math.log2(node.vol / g_vol)
            new_n = set()
            for nei in self.adj_table[node_id]:
                if nei in root_child:
                    new_n.add(nei)
            self.adj_table[node_id] = new_n
            subgraph_node_dict[node_id] = _PartitionTreeNode(
                ID=node_id, partition=node.partition, vol=node.vol, g=node.g,
                children=node.children)
        return subgraph_node_dict, ori_en

    def entropy(self, node_dict=None) -> float:
        if node_dict is None:
            node_dict = self.tree_node
        ent = 0.0
        for node in node_dict.values():
            if node.parent is not None:
                node_p = node_dict[node.parent]
                ent += -(node.g / self.VOL) * math.log2(node.vol / node_p.vol)
        return -ent

    def _leaf_up_entropy(self, sub_node_dict, sub_root_id, node_id):
        ent = 0.0
        stack = [sub_root_id]
        order = []
        while stack:
            nid = stack.pop(0)
            order.append(nid)
            if sub_node_dict[nid].children:
                stack.extend(sub_node_dict[nid].children)
        for sub_node_id in order:
            if sub_node_id == sub_root_id:
                sub_node_dict[sub_root_id].vol = self.tree_node[node_id].vol
                sub_node_dict[sub_root_id].g = self.tree_node[node_id].g
            elif sub_node_dict[sub_node_id].child_h == 1:
                node = sub_node_dict[sub_node_id]
                inner_vol = node.vol - node.g
                partition = node.partition
                ori_vol = sum(self.tree_node[i].vol for i in partition)
                ori_g = ori_vol - inner_vol
                node.vol = ori_vol
                node.g = ori_g
                node_p = sub_node_dict[node.parent]
                ent += -(node.g / self.VOL) * math.log2(node.vol / node_p.vol)
            else:
                node = sub_node_dict[sub_node_id]
                node.g = self.tree_node[sub_node_id].g
                node.vol = self.tree_node[sub_node_id].vol
                node_p = sub_node_dict[node.parent]
                ent += -(node.g / self.VOL) * math.log2(node.vol / node_p.vol)
        return ent

    def _leaf_up(self):
        h1_id = set()
        for l in self.leaves:
            h1_id.add(self.tree_node[l].parent)
        delta = 0.0
        id_mapping = {}
        h1_new_child_tree = {}
        for node_id in h1_id:
            candidate_node = self.tree_node[node_id]
            sub_nodes = candidate_node.partition
            if len(sub_nodes) in (1, 2):
                id_mapping[node_id] = None
            else:
                sub_g_vol = candidate_node.vol - candidate_node.g
                subgraph_node_dict, ori_ent = self._build_sub_leaves(
                    sub_nodes, candidate_node.vol)
                sub_root = self._build_k_tree(
                    g_vol=sub_g_vol, nodes_dict=subgraph_node_dict, k=2)
                self._check_balance(subgraph_node_dict, sub_root)
                new_ent = self._leaf_up_entropy(subgraph_node_dict, sub_root, node_id)
                delta += ori_ent - new_ent
                h1_new_child_tree[node_id] = subgraph_node_dict
                id_mapping[node_id] = sub_root
        delta = delta / self.g_num_nodes
        return delta, id_mapping, h1_new_child_tree

    def _leaf_up_update(self, id_mapping, leaf_up_dict):
        for node_id, h1_root in id_mapping.items():
            if h1_root is None:
                children = copy.deepcopy(self.tree_node[node_id].children)
                for i in children:
                    self._single_up(self.tree_node, i)
            else:
                h1_dict = leaf_up_dict[node_id]
                self.tree_node[node_id].children = h1_dict[h1_root].children
                for h1_c in h1_dict[h1_root].children:
                    h1_dict[h1_c].parent = node_id
                h1_dict.pop(h1_root)
                self.tree_node.update(h1_dict)
        self.tree_node[self.root_id].child_h += 1

    def _root_down_delta(self):
        if len(self.tree_node[self.root_id].children) < 3:
            return 0.0, None, None
        subgraph_node_dict, ori_entropy = self._build_root_down()
        g_vol = self.tree_node[self.root_id].vol
        new_root = self._build_k_tree(g_vol=g_vol, nodes_dict=subgraph_node_dict, k=2)
        self._check_balance(subgraph_node_dict, new_root)
        new_entropy = self.entropy(subgraph_node_dict)
        delta = (ori_entropy - new_entropy) / len(self.tree_node[self.root_id].children)
        return delta, new_root, subgraph_node_dict

    def _root_down_update(self, new_id, root_down_dict):
        self.tree_node[self.root_id].children = root_down_dict[new_id].children
        for node_id in root_down_dict[new_id].children:
            root_down_dict[node_id].parent = self.root_id
        root_down_dict.pop(new_id)
        self.tree_node.update(root_down_dict)
        self.tree_node[self.root_id].child_h += 1

    def build_coding_tree(self, k=2, mode="v2"):
        if k == 1:
            return
        if mode == "v1" or k is None:
            self.root_id = self._build_k_tree(self.VOL, self.tree_node, k=k)
        elif mode == "v2":
            self.root_id = self._build_k_tree(self.VOL, self.tree_node, k=2)
            self._check_balance(self.tree_node, self.root_id)
            if self.tree_node[self.root_id].child_h < 2:
                self.tree_node[self.root_id].child_h = 2
            flag = 0
            root_down_dict = None
            while self.tree_node[self.root_id].child_h < k:
                if flag == 0:
                    leaf_up_delta, id_mapping, leaf_up_dict = self._leaf_up()
                    root_down_delta, new_id, root_down_dict = self._root_down_delta()
                elif flag == 1:
                    leaf_up_delta, id_mapping, leaf_up_dict = self._leaf_up()
                elif flag == 2:
                    root_down_delta, new_id, root_down_dict = self._root_down_delta()
                else:
                    raise ValueError
                if leaf_up_delta < root_down_delta:
                    flag = 2
                    self._root_down_update(new_id, root_down_dict)
                else:
                    flag = 1
                    self._leaf_up_update(id_mapping, leaf_up_dict)
                    if root_down_delta != 0:
                        for rd_id, rd_node in root_down_dict.items():
                            if rd_node.child_h == 0:
                                rd_node.children = self.tree_node[rd_id].children


def structural_entropy(semantic_matrix: np.ndarray, tree_depth: int = 2) -> float:
    """Port of the official `compute_se`: coding-tree structural entropy of the
    semantic graph. Higher = more spread semantic structure = more uncertainty."""
    adj = np.asarray(semantic_matrix, dtype=np.float64)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1] or adj.shape[0] < 1:
        raise BackendError(f"sese: bad semantic matrix shape {adj.shape}")
    tree = _PartitionTree(adj)
    tree.build_coding_tree(tree_depth)
    return float(tree.entropy())

# ---------------------------------------------------------------------------
# Semantic graph construction (official construct_semantic_graph adaptation).
# ---------------------------------------------------------------------------
class DebertaNLI:
    """NLI wrapper returning [entailment, neutral, contradiction] probabilities.

    Label order is resolved from the model config's label2id, not hardcoded.
    """

    def __init__(self, model_path: str, device: Optional[str] = None,
                 dtype: str = "float16"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        torch_dtype = {"float16": torch.float16, "float32": torch.float32,
                       "bfloat16": torch.bfloat16}[dtype]
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, torch_dtype=torch_dtype).eval()
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.dtype = dtype
        id2label = getattr(self.model.config, "id2label", {}) or {}
        label_of = {str(v).upper(): int(k) for k, v in id2label.items()}
        missing = {"ENTAILMENT", "NEUTRAL", "CONTRADICTION"} - set(label_of)
        if missing:
            raise BackendError(
                f"sese: NLI model at {model_path} lacks labels {sorted(missing)}")
        self._idx = {name: label_of[name]
                     for name in ("ENTAILMENT", "NEUTRAL", "CONTRADICTION")}

    def probs(self, pairs) -> list[list[float]]:
        torch = self._torch
        inputs = self.tokenizer(
            list(pairs), padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        p = torch.softmax(logits, dim=1).cpu().tolist()
        # official SeSE order: [entail, neutral, contradiction]
        return [[q[self._idx["ENTAILMENT"]], q[self._idx["NEUTRAL"]],
                 q[self._idx["CONTRADICTION"]]] for q in p]


class SentenceEmbedder:
    def __init__(self, model_path: str):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise BackendError(
                f"sese: sentence-transformers unavailable ({e}); "
                "install it or register the method as blocked") from e
        try:
            self.model = SentenceTransformer(model_path)
        except Exception as e:
            raise BackendError(
                f"sese: failed to load sentence-embedding model {model_path}: {e}") from e

    def encode(self, texts: list[str]) -> np.ndarray:
        emb = self.model.encode(texts, convert_to_numpy=True,
                                normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float64)


def _page_rank(adjacency_matrix: np.ndarray, damping_factor: float = 0.85,
               max_iterations: int = 100, tol: float = 1e-6) -> np.ndarray:
    n = adjacency_matrix.shape[0]
    out_degree = np.array(adjacency_matrix.sum(axis=0)).flatten()
    out_degree[out_degree == 0] = 1
    P = adjacency_matrix / out_degree
    rank = np.full(n, 1.0 / n)
    for _ in range(max_iterations):
        new_rank = (1 - damping_factor) / n + damping_factor * P.T @ rank
        if np.linalg.norm(new_rank - rank, 1) < tol:
            break
        rank = new_rank
    return rank


def _connected_components(adjacency_matrix: np.ndarray):
    n = adjacency_matrix.shape[0]
    parent = list(range(n))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    for u, v in np.argwhere(adjacency_matrix > 0):
        ru, rv = find(int(u)), find(int(v))
        if ru != rv:
            parent[ru] = rv
    comp: dict[int, list[int]] = {}
    for i in range(n):
        comp.setdefault(find(i), []).append(i)
    return list(comp.values())


def _make_connected(adjacency_matrix: np.ndarray, entail_fn) -> np.ndarray:
    """Official repair: connect components via maximum-entailment representative
    pairs (PageRank representative per component + Kruskal-style bridging)."""
    components = _connected_components(adjacency_matrix)
    if len(components) <= 1:
        return adjacency_matrix
    representatives = []
    for comp in components:
        pr = _page_rank(adjacency_matrix[np.ix_(comp, comp)])
        representatives.append(comp[int(np.argmax(pr))])
    rep_pairs = list(itertools.combinations(representatives, 2))
    if not rep_pairs:
        return adjacency_matrix
    weights = entail_fn([(0, 0)] if False else rep_pairs)
    heap = []
    for (u, v), w in zip(rep_pairs, weights):
        heapq.heappush(heap, (-w[0], u, v))
    parent = list(range(adjacency_matrix.shape[0]))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    added = 0
    while heap:
        w, u, v = heapq.heappop(heap)
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            adjacency_matrix[u, v] = adjacency_matrix[v, u] = -w
            added += 1
    return adjacency_matrix


def sese_risk(responses: list[str], nli: DebertaNLI, embedder: SentenceEmbedder,
              tree_depth: int = 2, w_entail: float = 0.65,
              similarity_threshold: float = 0.3) -> tuple[float, dict]:
    """SeSE short-form score over the sampled-answer pool.

    responses: K sampled answer texts (shared generation pool; cost accounted
    separately from generation per DESIGN section 8.1). Official GPT-4o answer
    enhancement is NOT applied (no API key on this machine) - recorded in info.
    """
    from sklearn.cluster import AgglomerativeClustering

    K = len(responses)
    if K < 2:
        raise BackendError(f"sese needs K>=2 sampled answers, got {K}")
    if any(not str(r).strip() for r in responses):
        raise BackendError("sese: empty response text in the sample pool")

    emb = embedder.encode(responses)
    cos_sim = emb @ emb.T

    pairs = list(itertools.combinations(range(K), 2))
    entail_sim = np.zeros((K, K), dtype=np.float64)
    for i in range(0, len(pairs), 32):
        batch = pairs[i:i + 32]
        probs = nli.probs([(responses[a], responses[b]) for a, b in batch])
        for (a, b), p in zip(batch, probs):
            entail_sim[a, b] = entail_sim[b, a] = p[0]

    sim = np.clip(w_entail * entail_sim + (1 - w_entail) * cos_sim, 0.0, 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)

    non_diag = dist[~np.eye(K, dtype=bool)]
    if np.all(non_diag == 0.0):
        cluster_ids = [0] * K
    elif np.all(non_diag == 1.0):
        cluster_ids = list(range(K))
    else:
        clusterer = AgglomerativeClustering(
            n_clusters=None, distance_threshold=1.0 - similarity_threshold,
            metric="precomputed", linkage="average")
        cluster_ids = clusterer.fit_predict(dist).tolist()

    # Official graph: entailment-probability edges INSIDE clusters only.
    adj = np.zeros((K, K), dtype=np.float64)
    inner_pairs = [(i, j) for i, j in pairs if cluster_ids[i] == cluster_ids[j]]
    if inner_pairs:
        # entailment probs for inner pairs were already computed above
        for (a, b) in inner_pairs:
            adj[a, b] = adj[b, a] = entail_sim[a, b]

    adj = _make_connected(adj, lambda prs: nli.probs(
        [(responses[a], responses[b]) for a, b in prs]))

    se = structural_entropy(adj, tree_depth=tree_depth)
    info = {
        "version": VERSION,
        "answer_enhancement": "none:blocked-no-api",
        "n_clusters": int(len(set(cluster_ids))),
        "cluster_ids": cluster_ids,
        "n_edges_inner": int(len(inner_pairs)),
        "n_nli_calls": int(len(pairs) + len([
            (a, b) for a, b in itertools.combinations(
                _representatives(adj), 2)])),
        "tree_depth": tree_depth,
        "w_entail": w_entail,
        "similarity_threshold": similarity_threshold,
        "structural_entropy": se,
    }
    return float(se), info


def _representatives(adj: np.ndarray) -> list[int]:
    comps = _connected_components(adj)
    return [c[int(np.argmax(_page_rank(adj[np.ix_(c, c)])))] for c in comps]



