"""
Graph data structure implementation for PageRank calculations.

This module provides a directed graph implementation for PageRank algorithm calculations.

Example:
    >>> graph = Graph.from_edge_list([[1, 2], [2], [0]])  # Creates a circular graph
    >>> print(graph.get_outgoing_edges(0))  # Shows nodes that node points to
    {1, 2}
"""

# Standard Library Imports
from collections import defaultdict
from typing import Dict, Set, List, Iterator


class Graph:
    """
    A directed graph implementation using an adjacency list.

    Nodes are identified by unique integer IDs starting from zero.
    Provides efficient storage and fast access for sparse graphs.
    """

    def __init__(self) -> None:
        """Initialize an empty graph."""
        self._outgoing: Dict[int, Set[int]] = defaultdict(set)
        self._incoming: Dict[int, Set[int]] = defaultdict(set)
        self._nodes: List[int] = []

    def add_node(self, node_id: int) -> int:
        """
        Add a node to the graph if it doesn't already exist.

        Args:
            node_id (int): The ID of the node to add.

        Returns:
            int: The ID of the added node.

        Raises:
            ValueError: If node_id is negative.
        """
        if node_id < 0:
            raise ValueError("Node IDs must  be non-negative integers")

        if not self.has_node(node_id):
            self._nodes.append(node_id)
        return node_id

    def add_edge(self, from_id: int, to_id: int) -> None:
        """
        Add a directed edge between two nodes.

        Creates nodes if they don't exist and adds a directed edge from the source node to the target node.

        Args:
            from_id: Source node ID
            to_id: Target node ID
        """
        from_node = self.add_node(from_id)
        to_node = self.add_node(to_id)

        self._outgoing[from_node].add(to_node)
        self._incoming[to_node].add(from_node)

    def get_outgoing_edges(self, node_id: int) -> Set[int]:
        """Get all nodes that the specified node points to."""
        return self._outgoing[node_id] if self.has_node(node_id) else set()

    def get_incoming_edges(self, node_id: int) -> Set[int]:
        """Get all nodes that point to the specified node."""
        return self._incoming[node_id] if self.has_node(node_id) else set()

    def get_out_degree(self, node_id: int) -> int:
        """Get the number of outgoing edges for a node."""
        return len(self.get_outgoing_edges(node_id))

    def get_in_degree(self, node_id: int) -> int:
        """Get the number of incoming edges for a node."""
        return len(self.get_incoming_edges(node_id))

    def iter_nodes(self) -> Iterator[int]:
        """Iterate over all nodes in the graph."""
        return iter(self._nodes)

    @property
    def node_count(self) -> int:
        """Get the total number of nodes in the graph."""
        return len(self._nodes)

    def has_node(self, node_id: int) -> bool:
        """Check if a node exists in the graph."""
        return node_id in self._nodes

    def has_edge(self, from_id: int, to_id: int) -> bool:
        """Check if an edge exists between two nodes."""
        from_node_exists = self.has_node(from_id)
        to_node_exists = self.has_node(to_id)
        return bool(from_node_exists and to_node_exists and to_id in self._outgoing[from_id])

    @classmethod
    def from_edge_list(cls, edges: List[List[int]]) -> 'Graph':
        """
        Create a graph from a list of edge lists.

        Each index in the edge list represents a node ID, and the corresponding
        list contains the IDs of nodes that it points to.

        Args:
            edges: List where each element is a list of node IDs that the index node points to

        Returns:
            Graph: A new Graph instance with the specified edges
        """
        graph: Graph = cls()
        for from_id, to_ids in enumerate(edges):
            for to_id in to_ids:
                if to_id >= 0:
                    graph.add_edge(from_id, to_id)
        return graph

    def __str__(self):
        """
        Create a string representation of the graph showing adjacency lists.

        Returns:
            str: A string showing each node and its outgoing edges
        """
        adjacency_representation: str = ""
        for node in self._nodes:
            adjacency_representation += f"{node}: {sorted(self.get_outgoing_edges(node))}\n"
        return adjacency_representation